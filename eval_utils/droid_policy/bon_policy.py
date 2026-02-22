import datetime
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from tianshou.data import Batch

matplotlib.use("Agg")

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from .default_policy import ARDroidRoboarenaPolicy, broadcast_to_workers
from .reward_utils import get_progress_predictions

logger = logging.getLogger(__name__)


class BestOfNARDroidRoboarenaPolicy(ARDroidRoboarenaPolicy):
    """AR_droid policy with best-of-N sampling via a reward model.

    Generates num_candidates action chunks in parallel, decodes the predicted
    videos, scores them with an external reward model (robometer), and returns
    the action chunk from the highest-scoring candidate.

    Visualization bookkeeping (when ``output_dir`` is set):
        {output_dir}/ep_{i:04d}_pred.mp4     — best candidate's decoded view,
                                                concatenated across all inference
                                                steps within the episode.
        {output_dir}/ep_{i:04d}_bon/
            step_{j:04d}.mp4                  — per-step BoN comparison video.
    """

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
        num_candidates: int = 4,
        rm_host: str | None = None,
        rm_port: int | None = None,
        reward_view: str = "left_exterior",
    ):
        super().__init__(
            groot_policy=groot_policy,
            signal_group=signal_group,
            output_dir=output_dir,
        )

        self._num_candidates = num_candidates
        self._rm_url: str | None = None
        if rm_host is not None and rm_port is not None:
            self._rm_url = f"http://{rm_host}:{rm_port}"

        if reward_view not in ("wrist", "left_exterior", "right_exterior", "full_grid"):
            raise ValueError(f"Unknown reward_view: {reward_view!r}")
        self._reward_view = reward_view

        # Episode / step bookkeeping
        self._episode_idx = 0
        self._step_idx = 0
        self._episode_pred_frames: list[np.ndarray] = []
        
    # ------------------------------------------------------------------
    # Observation / action helpers (inherited)
    # ------------------------------------------------------------------

    def repeat_observation(self, obs: dict) -> dict:
        """Repeat observation tensors to create a batch of size num_candidates."""
        repeated_obs = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                repeated_obs[k] = np.repeat(v[None], self._num_candidates, axis=0)
            else:
                repeated_obs[k] = [v for _ in range(self._num_candidates)]
        return repeated_obs

    # ------------------------------------------------------------------
    # KV-cache management
    # ------------------------------------------------------------------

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
        B = self._num_candidates

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

    # ------------------------------------------------------------------
    # Video decode
    # ------------------------------------------------------------------

    def _decode_video_latents(self, video_pred: torch.Tensor) -> np.ndarray:
        """Decode VAE video latents and extract a single camera view.

        The model produces a 2x2 grid of views with shape (B, C, T, 2H, 2W):
            Top half  (rows :H):   wrist view, pixel-doubled to 2W wide
            Bottom-left  (H:, :W): left exterior
            Bottom-right (H:, W:): right exterior

        Which view is extracted is controlled by ``self._reward_view``:
            "wrist"           — top half, subsampled to (H, W)
            "left_exterior"   — bottom-left quadrant (H, W)
            "right_exterior"  — bottom-right quadrant (H, W)
            "full_grid"       — entire (2H, 2W) grid, no cropping

        Args:
            video_pred: (B, C, T, H_latent, W_latent) latent tensor

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
        # frames: (B, C, T, 2H, 2W)
        H = frames.shape[3] // 2
        W = frames.shape[4] // 2

        if self._reward_view == "wrist":
            view = frames[:, :, :, :H, ::2]       # undo pixel-doubling
        elif self._reward_view == "left_exterior":
            view = frames[:, :, :, H:, :W]
        elif self._reward_view == "right_exterior":
            view = frames[:, :, :, H:, W:]
        else:  # full_grid
            view = frames

        view = rearrange(view, "B C T H W -> B T H W C")
        return ((view.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

    # ------------------------------------------------------------------
    # Reward model scoring
    # ------------------------------------------------------------------

    def _score_single(self, video: np.ndarray, task: str, idx: int) -> tuple[int, float, np.ndarray]:
        """Score a single candidate. Returns (index, final_score, progress_curve)."""
        try:
            progress = get_progress_predictions(
                video_input=video,
                task=task,
                eval_server_url=self._rm_url,
            )
            score = progress[-1] if len(progress) > 0 else 0.0
            return idx, score, np.asarray(progress, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Reward scoring failed for candidate {idx}: {e}")
            return idx, 0.0, np.zeros(video.shape[0], dtype=np.float32)

    def _score_candidates(
        self, videos: np.ndarray, tasks: list[str],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Score each candidate video via the reward model server (parallel).

        Args:
            videos: (B, T, H, W, C) uint8 frames
            tasks: language task instructions (length B)

        Returns:
            scores: (B,) array of final progress scores
            progress_curves: list of B arrays, each (T,) per-frame progress
        """
        if self._rm_url is None:
            raise RuntimeError(
                "Reward model URL not configured. "
                "Pass rm_host and rm_port to BestOfNARDroidRoboarenaPolicy."
            )

        B, T = videos.shape[0], videos.shape[1]
        scores = np.zeros(B, dtype=np.float32)
        progress_curves: list[np.ndarray] = [np.zeros(T, dtype=np.float32) for _ in range(B)]

        with ThreadPoolExecutor(max_workers=B) as pool:
            futures = [
                pool.submit(self._score_single, videos[i], tasks[i], i)
                for i in range(B)
            ]
            for future in as_completed(futures):
                idx, score, curve = future.result()
                scores[idx] = score
                progress_curves[idx] = curve

        return scores, progress_curves

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_bon_visualization(
        self,
        videos: np.ndarray,
        progress_curves: list[np.ndarray],
        scores: np.ndarray,
        best_idx: int,
    ) -> str | None:
        """Save a two-column video: candidate rollouts (left) + progress plots (right).

        Saved to ``{output_dir}/ep_{i}_bon/step_{j:04d}.mp4``.
        """
        B, T, H, W, C = videos.shape
        plot_h, plot_w = H, W

        # -- Pre-render base plots (one matplotlib call per candidate) ----
        base_plots: list[np.ndarray] = []
        dot_coords: list[np.ndarray] = []  # (Tp, 2) pixel coords per candidate

        for i in range(B):
            is_best = i == best_idx
            progress = progress_curves[i]
            Tp = len(progress)

            fig, ax = plt.subplots(figsize=(plot_w / 100, plot_h / 100), dpi=100)
            color = "#2ecc71" if is_best else "#6495ed"
            ax.plot(range(Tp), progress, color=color, linewidth=2)
            ax.set_xlim(-0.5, max(Tp - 0.5, 0.5))
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("Frame", fontsize=7)
            ax.set_ylabel("Progress", fontsize=7)

            title = f"#{i}  {scores[i]:.3f}"
            if is_best:
                title = f"* #{i} BEST  {scores[i]:.3f}"
                for spine in ax.spines.values():
                    spine.set_color("#2ecc71")
                    spine.set_linewidth(2.5)
            ax.set_title(
                title, fontsize=8,
                fontweight="bold" if is_best else "normal",
                color="#2ecc71" if is_best else "#333333",
            )
            fig.tight_layout(pad=0.4)
            fig.canvas.draw()

            # Map data coords → pixel coords in the rendered image
            fig_h_px = int(fig.get_figheight() * fig.dpi)
            coords = np.zeros((Tp, 2), dtype=np.int32)
            for t in range(Tp):
                dx, dy = ax.transData.transform((t, progress[t]))
                coords[t, 0] = int(np.clip(dx, 0, plot_w - 1))
                coords[t, 1] = int(np.clip(fig_h_px - dy, 0, plot_h - 1))

            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)

            if buf.shape[0] != plot_h or buf.shape[1] != plot_w:
                from PIL import Image as _PILImage
                buf = np.array(_PILImage.fromarray(buf).resize((plot_w, plot_h), _PILImage.LANCZOS))

            base_plots.append(buf)
            dot_coords.append(coords)

        # -- Precompute circle mask for the red dot -----------------------
        R = 5
        yy, xx = np.ogrid[-R : R + 1, -R : R + 1]
        circle_mask = (xx * xx + yy * yy) <= R * R

        def _stamp_dot(img: np.ndarray, cx: int, cy: int, color=(255, 50, 50)):
            h, w = img.shape[:2]
            y0, y1 = max(cy - R, 0), min(cy + R + 1, h)
            x0, x1 = max(cx - R, 0), min(cx + R + 1, w)
            my0, mx0 = y0 - (cy - R), x0 - (cx - R)
            sub = circle_mask[my0 : my0 + y1 - y0, mx0 : mx0 + x1 - x0]
            img[y0:y1, x0:x1][sub] = color

        # -- Compose per-frame output ------------------------------------
        output_frames: list[np.ndarray] = []
        border = 3

        for t in range(T):
            rows = []
            for i in range(B):
                # Video panel
                vid = videos[i, t].copy()
                if i == best_idx:
                    vid[:border, :] = [46, 204, 113]
                    vid[-border:, :] = [46, 204, 113]
                    vid[:, :border] = [46, 204, 113]
                    vid[:, -border:] = [46, 204, 113]

                # Plot panel with red dot
                plot = base_plots[i].copy()
                tp = min(t, len(dot_coords[i]) - 1)
                _stamp_dot(plot, int(dot_coords[i][tp, 0]), int(dot_coords[i][tp, 1]))

                rows.append(np.hstack([vid, plot]))

            output_frames.append(np.vstack(rows))

        # -- Save to episode BoN folder ----------------------------------
        bon_dir = os.path.join(self._output_dir, f"ep_{self._episode_idx:04d}_bon")
        os.makedirs(bon_dir, exist_ok=True)
        path = os.path.join(bon_dir, f"step_{self._step_idx:04d}.mp4")
        imageio.mimsave(path, output_frames, fps=3, codec="libx264")
        logger.info(f"BoN visualization: {path}")
        return path


    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_state(self, save_video: bool = True) -> None:
        if save_video and self._output_dir is not None and self._episode_pred_frames:
            path = os.path.join(self._output_dir, f"ep_{self._episode_idx:04d}_pred.mp4")
            imageio.mimsave(path, self._episode_pred_frames, fps=5, codec="libx264")
            logger.info(f"Episode prediction video ({len(self._episode_pred_frames)} frames): {path}")

        self._episode_pred_frames = []
        self._episode_idx += 1
        self._step_idx = 0

        # Clear the inherited VideoSaver (we manage saves ourselves)
        if self._video_saver is not None:
            self._video_saver.clear()

        self._frame_buffer.clear()
        self._call_count = 0
        self._is_first_call = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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
        scores, progress_curves = self._score_candidates(videos, tasks)
        best_idx = int(np.argmax(scores))
        logger.info(
            f"BoN: best={best_idx}, score={scores[best_idx]:.4f}, "
            f"mean={scores.mean():.4f}, std={scores.std():.4f}, "
            f"scoring took {time.perf_counter() - t0:.2f}s"
        )

        # Replace KV caches with the best candidate's, expanded to batch size
        self._select_kv_cache(best_idx)

        # Accumulate best candidate's frames and save per-step BoN visualization
        if self._output_dir is not None:
            self._episode_pred_frames.extend(list(videos[best_idx]))
            self._save_bon_visualization(videos, progress_curves, scores, best_idx)

        self._step_idx += 1

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
