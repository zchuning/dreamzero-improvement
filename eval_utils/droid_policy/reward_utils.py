import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests


def get_progress_predictions(
    video_input: Union[str, np.ndarray],
    task: str,
    eval_server_url: str = "http://localhost:8000",
    port: Optional[int] = None,
    fps: float = 1.0,
    max_frames: int = 16,
    timeout_s: float = 120.0,
) -> np.ndarray:
    """
    Get progress predictions for a video or frame array from an evaluation server.
    
    Args:
        video_input: Either a path to video file (str), URL (str), .npy/.npz file (str),
                     or a numpy array of frames with shape (T, H, W, C) or (T, C, H, W)
        task: Task instruction string (e.g., "pick up the cup")
        eval_server_url: Base URL of evaluation server (default: "http://localhost:8000")
        port: Optional port number. If provided, overrides port in eval_server_url
        fps: Frames per second to extract if video_input is a video file (default: 1.0)
        max_frames: Maximum frames to send to server (default: 16)
        timeout_s: HTTP request timeout in seconds (default: 120.0)
    
    Returns:
        numpy array of progress predictions with shape (T,) where T is the number of frames
        
    Example:
        # With video path
        progress = get_progress_predictions(
            video_input="/path/to/video.mp4",
            task="pick up the cup",
            eval_server_url="http://40.119.56.66",
            port=8000
        )
        
        # With frame array
        frames = np.random.randint(0, 255, (100, 224, 224, 3), dtype=np.uint8)
        progress = get_progress_predictions(
            video_input=frames,
            task="pick up the cup"
        )
    """
    # Handle port override
    if port is not None:
        # Extract base URL without port
        if "://" in eval_server_url:
            protocol, rest = eval_server_url.split("://", 1)
            base = rest.split(":")[0].split("/")[0]
            eval_server_url = f"{protocol}://{base}:{port}"
        else:
            eval_server_url = f"{eval_server_url}:{port}"
    
    # Load frames
    if isinstance(video_input, np.ndarray):
        frames = video_input
        # Ensure uint8
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
    else:
        frames = _load_frames_input(video_input, fps=fps)
    
    if frames is None or frames.size == 0:
        raise ValueError("Could not load frames from input")
    
    # Create sample
    T = frames.shape[0]
    sample = {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": "0",
            "metadata": {"subsequence_length": int(T)},
            "video_embeddings": None,
        },
    }
    
    # Send request
    outputs = _post_evaluate_batch_npy(eval_server_url, [sample], timeout_s=timeout_s)
    
    # Extract progress predictions
    progress_pred = _extract_progress_from_output(outputs)
    
    return progress_pred


def _load_frames_input(video_or_array_path: str, fps: float = 1.0) -> np.ndarray:
    """Load frames from video file or numpy array file."""
    if video_or_array_path.endswith('.npy'):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith('.npz'):
        npz = np.load(video_or_array_path)
        if 'arr_0' in npz:
            frames_array = npz['arr_0']
        else:
            frames_array = next(iter(npz.values()))
    else:
        frames_array = _extract_frames(video_or_array_path, fps=fps)
    
    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    
    return frames_array


def _extract_frames(video_path: str, fps: float = 1.0) -> np.ndarray:
    """Extract frames from video file."""
    import decord
    
    vr = decord.VideoReader(video_path, num_threads=1)
    total_frames = len(vr)
    
    try:
        native_fps = float(vr.get_avg_fps())
    except Exception:
        native_fps = 1.0
    
    if fps is None or fps <= 0:
        fps = native_fps
    
    if native_fps > 0:
        desired_frames = int(round(total_frames * (fps / native_fps)))
    else:
        desired_frames = total_frames
    
    desired_frames = max(1, min(desired_frames, total_frames))
    
    if desired_frames == total_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()
    
    frames_array = vr.get_batch(frame_indices).asnumpy()
    del vr
    return frames_array


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    """Convert numpy array to file tuple for multipart upload."""
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def _build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build multipart payload for server request."""
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]
    
    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})
        
        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue
            
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()
            
            if isinstance(val, np.ndarray):
                file_key = f"sample_{i}_trajectory_{field}"
                files[file_key] = _numpy_to_npy_file_tuple(val, f"{file_key}.npy")
                traj_copy[field] = {"__numpy_file__": file_key}
            else:
                traj_copy[field] = val
        
        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]
        
        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)
    
    return files, data


def _post_evaluate_batch_npy(
    eval_server_url: str,
    samples: List[Dict[str, Any]],
    timeout_s: float = 120.0,
) -> Dict[str, Any]:
    """Send evaluation request to server."""
    files, data = _build_multipart_payload(samples)
    url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def _extract_progress_from_output(outputs: Dict[str, Any]) -> np.ndarray:
    """Extract progress predictions from server response."""
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    
    progress_pred = outputs_progress.get("progress_pred", [])
    if progress_pred and len(progress_pred) > 0:
        return np.array(progress_pred[0])
    else:
        return np.array([])


if __name__ == "__main__":
    # Example 1: With video file and port
    progress = get_progress_predictions(
        video_input="video_0_toaster_50fps_fixedrate_4x.mp4",
        task="put both pieces of bread on the plate",
        eval_server_url="http://0.0.0.0",
        port=8001,
        fps=2.0
    )

    print(f"Video 1 progress port 8001: {progress}")
    # Example 1: With video file and port
    progress = get_progress_predictions(
        video_input="video_0_toaster_50fps_fixedrate_4x.mp4",
        task="put both pieces of bread on the plate",
        eval_server_url="http://localhost",
        port=8004,
        fps=2.0
    )
    print(f"Video 1 progress port 8004: {progress}")

    # Example 2: With numpy array of frames
    frames = np.random.randint(0, 255, (12, 224, 224, 3), dtype=np.uint8)
    progress = get_progress_predictions(
        video_input=frames,
        task="close the drawer",
        eval_server_url="http://localhost",
        port=8004,
    )

    print(f"Video 2 Progress shape: {progress.shape}")
    print(f"Video 2 Total reward: {progress}")
    print(f"Video 2 Final reward: {progress[-1]}")
