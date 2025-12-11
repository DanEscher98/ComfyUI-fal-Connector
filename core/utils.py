"""
Utility functions for tensor/media conversion.

Handles conversion between:
- PyTorch tensors (ComfyUI format)
- PIL Images
- fal.ai URLs
"""

from __future__ import annotations

import io
import tempfile
from typing import Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

import fal_client


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a ComfyUI image tensor to PIL Image.

    ComfyUI tensors are [B, H, W, C] with values in [0, 1].

    Args:
        tensor: Image tensor, shape [B, H, W, C] or [H, W, C]

    Returns:
        PIL Image (first image if batch)
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image in batch

    # Convert to numpy and scale to 0-255
    np_image = tensor.cpu().numpy()
    np_image = (np_image * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(np_image)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI image tensor.

    Args:
        image: PIL Image

    Returns:
        Tensor with shape [1, H, W, C] and values in [0, 1]
    """
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy and normalize
    np_image = np.array(image).astype(np.float32) / 255.0

    # Add batch dimension [H, W, C] -> [1, H, W, C]
    tensor = torch.from_numpy(np_image).unsqueeze(0)

    return tensor


def upload_image_tensor(tensor: torch.Tensor) -> str:
    """
    Upload a ComfyUI image tensor to fal.ai storage.

    Args:
        tensor: Image tensor [B, H, W, C]

    Returns:
        URL of uploaded image
    """
    pil_image = tensor_to_pil(tensor)

    # Save to temporary PNG file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_image.save(f.name, format="PNG")
        temp_path = f.name

    # Upload to fal.ai
    url = fal_client.upload_file(temp_path)

    return url


def download_image(url: str, timeout: float = 60.0) -> torch.Tensor:
    """
    Download an image from URL and convert to ComfyUI tensor.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        Image tensor [1, H, W, C]
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    image = Image.open(io.BytesIO(response.content))
    return pil_to_tensor(image)


def download_video(url: str, timeout: float = 120.0) -> Tuple[str, bytes]:
    """
    Download a video from URL.

    Args:
        url: Video URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (content_type, video_bytes)
    """
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "video/mp4")

    # Read full content
    video_bytes = response.content

    return content_type, video_bytes


def save_video_temp(video_bytes: bytes, suffix: str = ".mp4") -> str:
    """
    Save video bytes to a temporary file.

    Args:
        video_bytes: Raw video data
        suffix: File extension

    Returns:
        Path to temporary file
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        return f.name


def get_image_dimensions(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    Get image dimensions from tensor.

    Args:
        tensor: Image tensor [B, H, W, C]

    Returns:
        (width, height)
    """
    if tensor.dim() == 4:
        _, h, w, _ = tensor.shape
    else:
        h, w, _ = tensor.shape

    return w, h


def resize_image_tensor(
    tensor: torch.Tensor,
    max_size: int = 1920,
    min_size: int = 64,
) -> torch.Tensor:
    """
    Resize image tensor if needed to fit within constraints.

    Args:
        tensor: Image tensor [B, H, W, C]
        max_size: Maximum dimension
        min_size: Minimum dimension

    Returns:
        Resized tensor
    """
    pil_image = tensor_to_pil(tensor)

    w, h = pil_image.size
    max_dim = max(w, h)
    min_dim = min(w, h)

    # Check if resize needed
    if max_dim <= max_size and min_dim >= min_size:
        return tensor

    # Calculate new size
    if max_dim > max_size:
        scale = max_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
    elif min_dim < min_size:
        scale = min_size / min_dim
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        return tensor

    # Resize
    resized = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return pil_to_tensor(resized)
