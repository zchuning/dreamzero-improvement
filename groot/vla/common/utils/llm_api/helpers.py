import base64
import glob
import os
import shutil
import tempfile

import cv2

# Common OpenAI Model names:
# "gpt-4o"
# "gpt-4.1"
# "gpt-4.1-mini"
# "gpt-4.1-nano"
# "gpt-4.5-preview"
# "o3"
# "o3-mini"
# "o4-mini"

# Common Claude Model names:
# us.anthropic.claude-opus-4-20250514-v1:0
# us.anthropic.claude-sonnet-4-20250514-v1:0

# The specific models you have access to depend on your LLMGateway token.


def get_media_type(file_path):
    """Get the MIME media type for a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: MIME media type string (e.g., "image/jpeg", "image/png").
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif file_ext == ".png":
        return "image/png"
    elif file_ext == ".gif":
        return "image/gif"
    elif file_ext == ".webp":
        return "image/webp"
    else:
        return "image/jpeg"


def ask(
    client,
    message,
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    temperature=0.1,
    max_tokens=1000,
):
    """Send a text message to an LLM client and get a response.

    This function provides a simple interface to interact with various LLM models
    through a client object (e.g., OpenAI, Anthropic, etc.).

    Args:
        client: The LLM client object that implements the chat completion interface.
            Should have a callable interface that accepts messages, model, temperature,
            and max_tokens parameters.
        message (str): The user message to send to the model.
        model (str, optional): The model name to use. Defaults to "gpt-4o".
            Common options include:
            - OpenAI: "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.5-preview"
            - Anthropic: "us.anthropic.claude-opus-4-20250514-v1:0", "us.anthropic.claude-sonnet-4-20250514-v1:0"
        system_prompt (str, optional): The system prompt to set the model's behavior.
            Defaults to "You are a helpful assistant.".
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        str or None: The model's response text, or None if the API call failed.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = client(
        messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1
    )
    if response and response.choices:
        return response.choices[0].message.content
    return None


def ask_multiple(
    client,
    message,
    n=3,
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    temperature=0.1,
    max_tokens=1000,
):
    """Send a text message to an LLM client and get multiple responses.

    Args:
        client: The LLM client object that implements the chat completion interface.
        message (str): The user message to send to the model.
        n (int, optional): Number of completions to generate. Defaults to 3.
        model (str, optional): The model name to use. Defaults to "gpt-4o".
        system_prompt (str, optional): The system prompt to set the model's behavior.
            Defaults to "You are a helpful assistant.".
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        list or None: List of response strings, or None if the API call failed.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = client(
        messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n
    )
    if response and response.choices:
        return [choice.message.content for choice in response.choices]
    return None


def ask_about_image(client, image_path, message, model, temperature=0.1, max_tokens=1000):
    """Send a message with one or more images to an LLM client for analysis.

    This function allows you to ask questions about images using vision-capable
    LLM models. It supports both single images and directories containing multiple images.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        image_path (str): Path to a single image file or a directory containing images.
            If a directory is provided, all supported image files will be included.
        message (str): The question or prompt about the image(s).
        model (str): The vision-capable model name to use (e.g., "gpt-4o").
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        str: The model's response analyzing the image(s), or an error message if
            the operation failed.
    """
    if os.path.isdir(image_path):
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]
        found_images = []
        for ext in image_extensions:
            pattern = os.path.join(image_path, f"*{ext}")
            found_images.extend(glob.glob(pattern))
            pattern = os.path.join(image_path, f"*{ext.upper()}")
            found_images.extend(glob.glob(pattern))
        image_paths = sorted(list(set(found_images)))
        if not image_paths:
            return f"❌ Error: No image files found in directory '{image_path}'. Supported formats: {', '.join(image_extensions)}"
    else:
        image_paths = [image_path]

    content = [{"type": "text", "text": message}]
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        media_type = get_media_type(image_path)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}}
        )
    messages = [{"role": "user", "content": content}]
    response = client(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1)
    if response is None:
        return f"Error: API call failed for model '{model}'. Check logs for details."
    return response.choices[0].message.content


def ask_about_image_multiple(
    client, image_path, message, model, n=3, temperature=0.1, max_tokens=1000
):
    """Send a message with one or more images to an LLM client and get multiple responses.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        image_path (str): Path to a single image file or a directory containing images.
        message (str): The question or prompt about the image(s).
        model (str): The vision-capable model name to use (e.g., "gpt-4o").
        n (int, optional): Number of completions to generate. Defaults to 3.
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        list or None: List of response strings, or None if the API call failed.
    """
    if os.path.isdir(image_path):
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]
        found_images = []
        for ext in image_extensions:
            pattern = os.path.join(image_path, f"*{ext}")
            found_images.extend(glob.glob(pattern))
            pattern = os.path.join(image_path, f"*{ext.upper()}")
            found_images.extend(glob.glob(pattern))
        image_paths = sorted(list(set(found_images)))
        if not image_paths:
            return f"❌ Error: No image files found in directory '{image_path}'. Supported formats: {', '.join(image_extensions)}"
    else:
        image_paths = [image_path]

    content = [{"type": "text", "text": message}]
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        media_type = get_media_type(image_path)
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}}
        )
    messages = [{"role": "user", "content": content}]
    response = client(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n)
    if response is None:
        return f"Error: API call failed for model '{model}'. Check logs for details."
    return [choice.message.content for choice in response.choices]


def extract_frames_from_video(video_path, num_frames=20, output_dir=None, cleanup=True):
    """Extract frames from a video file for analysis.

    This function extracts a specified number of frames from a video file,
    evenly distributed throughout the video duration. The frames are saved
    as JPEG images in the specified output directory.

    Args:
        video_path (str): Path to the input video file.
        num_frames (int, optional): Number of frames to extract. Defaults to 20.
            If the video has fewer frames than requested, all available frames
            will be extracted.
        output_dir (str, optional): Directory to save the extracted frames.
            If None, a temporary directory will be created. Defaults to None.
        cleanup (bool, optional): Whether to clean up the output directory
            after extraction. Only applies when output_dir is None.
            Defaults to True.

    Returns:
        tuple: (frame_paths, output_dir) where:
            - frame_paths (list): List of paths to the extracted frame images
            - output_dir (str): Path to the directory containing the frames

    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Error: Could not open video file '{video_path}'")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames > total_frames:
        num_frames = total_frames
    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        step = (total_frames - 1) / (num_frames - 1)
        frame_indices = [int(i * step) for i in range(num_frames)]
    frame_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_filename = f"frame_{i:03d}_{frame_idx:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(frame_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
    cap.release()
    return frame_paths, output_dir


def ask_about_video(
    client,
    video_path,
    message,
    model="gpt-4o",
    num_frames=20,
    cleanup=True,
    temperature=0.1,
    max_tokens=1000,
):
    """Analyze a video by extracting frames and asking an LLM about them.

    This function combines video frame extraction with LLM analysis. It extracts
    frames from a video file and sends them along with a message to a vision-capable
    LLM for analysis.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        video_path (str): Path to the video file to analyze.
        message (str): The question or prompt about the video content.
        model (str, optional): The vision-capable model name to use.
            Defaults to "gpt-4o".
        num_frames (int, optional): Number of frames to extract from the video
            for analysis. Defaults to 20.
        cleanup (bool, optional): Whether to clean up temporary frame files
            after analysis. Defaults to True.
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        str: The model's response analyzing the video content, or an error message
            if the operation failed.
    """
    try:
        frame_paths, temp_dir = extract_frames_from_video(
            video_path, num_frames=num_frames, cleanup=False
        )
        if not frame_paths:
            return "❌ Error: No frames could be extracted from the video"
        result = ask_about_image(
            client, temp_dir, message, model, temperature=temperature, max_tokens=max_tokens
        )
        return result
    except Exception as e:
        return f"❌ Error analyzing video: {str(e)}"
    finally:
        if cleanup and "temp_dir" in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def ask_about_video_multiple(
    client,
    video_path,
    message,
    model="gpt-4o",
    num_frames=20,
    cleanup=True,
    n=3,
    temperature=0.1,
    max_tokens=1000,
):
    """Analyze a video by extracting frames and getting multiple LLM responses.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        video_path (str): Path to the video file to analyze.
        message (str): The question or prompt about the video content.
        model (str, optional): The vision-capable model name to use.
            Defaults to "gpt-4o".
        num_frames (int, optional): Number of frames to extract from the video
            for analysis. Defaults to 20.
        cleanup (bool, optional): Whether to clean up temporary frame files
            after analysis. Defaults to True.
        n (int, optional): Number of completions to generate. Defaults to 3.
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        list or None: List of response strings, or None if the operation failed.
    """
    try:
        frame_paths, temp_dir = extract_frames_from_video(
            video_path, num_frames=num_frames, cleanup=False
        )
        if not frame_paths:
            return "❌ Error: No frames could be extracted from the video"
        result = ask_about_image_multiple(
            client, temp_dir, message, model, n=n, temperature=temperature, max_tokens=max_tokens
        )
        return result
    except Exception as e:
        return f"❌ Error analyzing video: {str(e)}"
    finally:
        if cleanup and "temp_dir" in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def analyze_simulation_with_code_and_config(
    client,
    python_file,
    yaml_config,
    video_path=None,
    prompt_file=None,
    message="Analyze this robot simulation code, configuration, and execution results.",
    model="gpt-4o",
    num_frames=20,
    cleanup=True,
    temperature=0.1,
    max_tokens=1000,
):
    """Comprehensive analysis of simulation with code, config, and video results.

    This function performs a comprehensive analysis of a robot simulation by combining:
    - Python simulation code
    - YAML configuration files
    - Optional video results
    - Optional custom analysis prompt

    It's intended to be used for providing the LLM with a comprehensive context of the simulation,
    and then asking it to analyze the simulation.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        python_file (str): Path to the Python simulation file to analyze.
        yaml_config (str): Path to the YAML configuration file used in the simulation.
        video_path (str, optional): Path to a video file showing simulation results.
            If provided, frames will be extracted and included in the analysis.
            Defaults to None.
        prompt_file (str, optional): Path to a text file containing a custom
            analysis prompt. If provided, this will override the default message.
            Defaults to None.
        message (str, optional): Default analysis prompt to use if no prompt_file
            is provided. Defaults to "Analyze this robot simulation code, configuration, and execution results.".
        model (str, optional): The vision-capable model name to use.
            Defaults to "gpt-4o".
        num_frames (int, optional): Number of frames to extract from the video
            if video_path is provided. Defaults to 20.
        cleanup (bool, optional): Whether to clean up temporary frame files
            after analysis. Defaults to True.
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        str: The model's comprehensive analysis of the simulation, or an error
            message if the operation failed.
    """
    if not os.path.exists(python_file):
        return f"❌ Error: Python file '{python_file}' not found"
    if not os.path.exists(yaml_config):
        return f"❌ Error: YAML config file '{yaml_config}' not found"
    if video_path and not os.path.exists(video_path):
        return f"❌ Error: Video file '{video_path}' not found"
    if prompt_file and not os.path.exists(prompt_file):
        return f"❌ Error: Prompt file '{prompt_file}' not found"
    temp_dir = None
    try:
        if prompt_file:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            analysis_message = prompt_content
        else:
            analysis_message = message
        with open(python_file, "r", encoding="utf-8") as f:
            python_content = f.read()
        with open(yaml_config, "r", encoding="utf-8") as f:
            yaml_content = f.read()
        video_frames_text = ""
        if video_path:
            try:
                frame_paths, temp_dir = extract_frames_from_video(
                    video_path, num_frames=num_frames, cleanup=False
                )
                if frame_paths:
                    video_frames_text = f"\n\n**VIDEO FRAMES:**\nExtracted {len(frame_paths)} frames from the simulation video showing the execution results."
                else:
                    video_frames_text = (
                        "\n\n**VIDEO:** No frames could be extracted from the provided video."
                    )
            except Exception as e:
                video_frames_text = f"\n\n**VIDEO ERROR:** Could not process video: {e}"
                frame_paths = []
        else:
            frame_paths = []
        code_section = f"""**PYTHON CODE - {os.path.basename(python_file)}:**\n```python\n{python_content}\n```\n\n**YAML CONFIGURATION - {os.path.basename(yaml_config)}:**\n```yaml\n{yaml_content}\n```{video_frames_text}"""
        content = [{"type": "text", "text": f"{analysis_message}\n\n{code_section}"}]
        if frame_paths:
            for frame_path in frame_paths:
                with open(frame_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                media_type = get_media_type(frame_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                    }
                )
        messages = [{"role": "user", "content": content}]
        response = client(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, n=1
        )
        if response is None:
            return f"Error: API call failed for model '{model}'. Check logs for details."
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"
    finally:
        if cleanup and temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass


def analyze_simulation_with_code_and_config_multiple(
    client,
    python_file,
    yaml_config,
    video_path=None,
    prompt_file=None,
    message="Analyze this robot simulation code, configuration, and execution results.",
    model="gpt-4o",
    num_frames=20,
    cleanup=True,
    n=3,
    temperature=0.1,
    max_tokens=1000,
):
    """Comprehensive analysis of simulation with multiple perspectives.

    This function performs a comprehensive analysis of a robot simulation and generates
    multiple responses, useful for getting different analytical perspectives.

    Args:
        client: The LLM client object that implements the chat completion interface
            with vision capabilities.
        python_file (str): Path to the Python simulation file to analyze.
        yaml_config (str): Path to the YAML configuration file used in the simulation.
        video_path (str, optional): Path to a video file showing simulation results.
            If provided, frames will be extracted and included in the analysis.
            Defaults to None.
        prompt_file (str, optional): Path to a text file containing a custom
            analysis prompt. If provided, this will override the default message.
            Defaults to None.
        message (str, optional): Default analysis prompt to use if no prompt_file
            is provided. Defaults to "Analyze this robot simulation code, configuration, and execution results.".
        model (str, optional): The vision-capable model name to use.
            Defaults to "gpt-4o".
        num_frames (int, optional): Number of frames to extract from the video
            if video_path is provided. Defaults to 20.
        cleanup (bool, optional): Whether to clean up temporary frame files
            after analysis. Defaults to True.
        n (int, optional): Number of completions to generate. Defaults to 3.
        temperature (float, optional): Controls randomness in the response.
            0.0 = deterministic, higher values = more random. Defaults to 0.1.
        max_tokens (int, optional): Maximum number of tokens in the response.
            Defaults to 1000.

    Returns:
        list or None: List of analysis responses, or None if the operation failed.
    """
    if not os.path.exists(python_file):
        return f"❌ Error: Python file '{python_file}' not found"
    if not os.path.exists(yaml_config):
        return f"❌ Error: YAML config file '{yaml_config}' not found"
    if video_path and not os.path.exists(video_path):
        return f"❌ Error: Video file '{video_path}' not found"
    if prompt_file and not os.path.exists(prompt_file):
        return f"❌ Error: Prompt file '{prompt_file}' not found"
    temp_dir = None
    try:
        if prompt_file:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_content = f.read().strip()
            analysis_message = prompt_content
        else:
            analysis_message = message
        with open(python_file, "r", encoding="utf-8") as f:
            python_content = f.read()
        with open(yaml_config, "r", encoding="utf-8") as f:
            yaml_content = f.read()
        video_frames_text = ""
        if video_path:
            try:
                frame_paths, temp_dir = extract_frames_from_video(
                    video_path, num_frames=num_frames, cleanup=False
                )
                if frame_paths:
                    video_frames_text = f"\n\n**VIDEO FRAMES:**\nExtracted {len(frame_paths)} frames from the simulation video showing the execution results."
                else:
                    video_frames_text = (
                        "\n\n**VIDEO:** No frames could be extracted from the provided video."
                    )
            except Exception as e:
                video_frames_text = f"\n\n**VIDEO ERROR:** Could not process video: {e}"
                frame_paths = []
        else:
            frame_paths = []
        code_section = f"""**PYTHON CODE - {os.path.basename(python_file)}:**\n```python\n{python_content}\n```\n\n**YAML CONFIGURATION - {os.path.basename(yaml_config)}:**\n```yaml\n{yaml_content}\n```{video_frames_text}"""
        content = [{"type": "text", "text": f"{analysis_message}\n\n{code_section}"}]
        if frame_paths:
            for frame_path in frame_paths:
                with open(frame_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                media_type = get_media_type(frame_path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{base64_image}"},
                    }
                )
        messages = [{"role": "user", "content": content}]
        response = client(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n
        )
        if response is None:
            return f"Error: API call failed for model '{model}'. Check logs for details."
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        return f"❌ Error during analysis: {str(e)}"
    finally:
        if cleanup and temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
