import asyncio
import dataclasses
import datetime
import logging
import os
import socket

import torch
import torch.distributed as dist
import tyro
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

# Modular components
from eval_utils.droid_policy import ARDroidRoboarenaPolicy, DistributedWorkerLoop
from eval_utils.policy_server import PolicyServerConfig
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    port: int = 8000
    timeout_seconds: int = 50000  # 10 hours default, configurable
    model_path: str = "/mnt/aws-lfs-01/shared/seonghyeony/checkpoints/dreamzero/1105/wan_action_train_i2v_multiview_agibot_diverse_subtask_subsampling_action_OTJ_1104_steps100000_gpus128_bs128_per_device1_shared_time_multiview/copy-ckpt-26000"
    enable_dit_cache: bool = False
    max_chunk_size: int | None = None  # If None, use config value. Otherwise override max_chunk_size for inference.


def init_mesh() -> DeviceMesh:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh


def main(args: Args) -> None:
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    embodiment_tag = "oxe_droid"
    model_path = args.model_path

    device_mesh = init_mesh()
    rank = dist.get_rank()

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        parent_dir = os.path.dirname(model_path)
        datetime_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{datetime_suffix}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    if rank == 0:
        wrapper_policy = ARDroidRoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            output_dir=output_dir,
        )

        server_config = PolicyServerConfig(
            image_resolution=(180, 320),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )

        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        worker = DistributedWorkerLoop(policy, signal_group)
        asyncio.run(worker.run())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)
