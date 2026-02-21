import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from .default_policy import ARDroidRoboarenaPolicy, broadcast_to_workers
from .reward_utils import get_progress_predictions

logger = logging.getLogger(__name__)


class BestOfNARDroidRoboarenaPolicy(ARDroidRoboarenaPolicy):
    """AR_droid policy with best-of-N sampling via a reward model.

    Generates num_candidates action chunks in parallel, decodes the predicted
    videos, scores them with an external reward model (robometer), and returns
    the action chunk from the highest-scoring candidate.
    """

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        num_candidates: int = 4,
        rm_host: str | None = None,
        rm_port: int | None = None,
    ):
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
        )
        self.num_candidates = num_candidates
        self._rm_url: str | None = None
        if rm_host is not None and rm_port is not None:
            self._rm_url = f"http://{rm_host}:{rm_port}"

    def repeat_observation(self, obs: dict) -> dict:
        """Repeat observation tensors to create a batch of size num_candidates."""
        repeated_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                repeated_obs[k] = np.repeat(v[None], self.num_candidates, axis=0)
            else:
                repeated_obs[k] = [v for _ in range(self.num_candidates)]
        return repeated_obs

    def _select_kv_cache(self, best_idx: int) -> None:
        """Replace the action head's KV caches with the best candidate's entry,
        expanded back to ``num_candidates`` so the next call sees a consistent
        batch dimension.

        Each cache is a list[Tensor] of length ``num_layers``.
        KV caches have shape ``[2, B, seq, num_heads, head_dim]``;
        cross-attn caches have shape ``[2, B, 512, num_heads, head_dim]``.
        ``clip_feas`` and ``ys`` have leading dim ``B``.
        """
        ah = self._policy.trained_model.action_head
        B = self.num_candidates

        def _select_and_expand(cache: list[torch.Tensor], dim: int = 1) -> list[torch.Tensor]:
            return [t.select(dim, best_idx).unsqueeze(dim).expand_as(t).contiguous()
                    for t in cache]

        if ah.kv_cache1 is not None:
            ah.kv_cache1 = _select_and_expand(ah.kv_cache1)
        if ah.kv_cache_neg is not None:
            ah.kv_cache_neg = _select_and_expand(ah.kv_cache_neg)
        if ah.crossattn_cache is not None:
            ah.crossattn_cache = _select_and_expand(ah.crossattn_cache)
        if ah.crossattn_cache_neg is not None:
            ah.crossattn_cache_neg = _select_and_expand(ah.crossattn_cache_neg)

        if ah.clip_feas is not None:
            ah.clip_feas = ah.clip_feas[best_idx:best_idx + 1].expand(B, *ah.clip_feas.shape[1:]).contiguous()
        if ah.ys is not None:
            ah.ys = ah.ys[best_idx:best_idx + 1].expand(B, *ah.ys.shape[1:]).contiguous()

    def _decode_video_latents(self, video_pred: torch.Tensor) -> np.ndarray:
        """Decode VAE video latents to uint8 pixel frames.

        Args:
            video_pred: (B, C, T, H, W) latent tensor

        Returns:
            (B, T, H, W, C) uint8 numpy array
        """
        ah = self._policy.trained_model.action_head
        frames = ah.vae.decode(
            video_pred,
            tiled=ah.tiled,
            tile_size=(ah.tile_size_height, ah.tile_size_width),
            tile_stride=(ah.tile_stride_height, ah.tile_stride_width),
        )
        frames = rearrange(frames, "B C T H W -> B T H W C")
        return ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

    def _score_single(self, video: np.ndarray, task: str, idx: int) -> tuple[int, float]:
        """Score a single candidate. Returns (index, score)."""
        try:
            progress = get_progress_predictions(
                video_input=video,
                task=task,
                eval_server_url=self._rm_url,
            )
            return idx, (progress[-1] if len(progress) > 0 else 0.0)
        except Exception as e:
            logger.warning(f"Reward scoring failed for candidate {idx}: {e}")
            return idx, 0.0

    def _score_candidates(self, videos: np.ndarray, tasks: list[str]) -> np.ndarray:
        """Score each candidate video via the reward model server (parallel).

        Args:
            videos: (B, T, H, W, C) uint8 frames
            tasks: language task instructions (length B)

        Returns:
            (B,) array of final progress scores
        """
        if self._rm_url is None:
            raise RuntimeError(
                "Reward model URL not configured. "
                "Pass rm_host and rm_port to BestOfNARDroidRoboarenaPolicy."
            )

        B = videos.shape[0]
        scores = np.zeros(B, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=B) as pool:
            futures = [
                pool.submit(self._score_single, videos[i], tasks[i], i)
                for i in range(B)
            ]
            for future in as_completed(futures):
                idx, score = future.result()
                scores[idx] = score

        return scores

    def infer(self, obs: dict) -> np.ndarray:
        """Infer with best-of-N selection."""
        # Session change detection
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        self._call_count += 1

        converted_obs = self._convert_observation(obs)
        repeated_obs = self.repeat_observation(converted_obs)
        tasks = repeated_obs.get("annotation.language.action_text", "")

        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # Broadcast repeated obs to workers
        broadcast_to_workers(repeated_obs)
        batch = Batch(obs=repeated_obs)

        # Distributed forward pass — all num_candidates processed in one batch
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # Decode video latents: (B, C, T, H, W) -> (B, T, H, W, C) uint8
        t0 = time.perf_counter()
        with torch.no_grad():
            videos = self._decode_video_latents(video_pred)
        logger.info(f"BoN: decoded {videos.shape[0]} candidates in {time.perf_counter() - t0:.2f}s")

        # Score candidates via reward model
        t0 = time.perf_counter()
        scores = self._score_candidates(videos, tasks)
        best_idx = int(np.argmax(scores))
        logger.info(
            f"BoN: best={best_idx}, score={scores[best_idx]:.4f}, "
            f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"scoring took {time.perf_counter() - t0:.2f}s"
        )

        # Replace KV caches with the best candidate's, expanded to batch size
        self._select_kv_cache(best_idx)

        # Save best candidate's video latent for visualization
        if self._video_saver is not None:
            self._video_saver.append(video_pred[best_idx : best_idx + 1])

        # Extract best candidate's actions (index out the batch dim)
        action_chunk_dict = result_batch.act
        best_action_dict = {}
        for k, v in action_chunk_dict.items():
            if k.startswith("action."):
                best_action_dict[k] = v[best_idx]

        action = self._convert_action(best_action_dict)

        if self._is_first_call:
            self._is_first_call = False

        return action
