"""
Video generation nodes for fal.ai models.

Provides nodes for:
- Luma Dream Machine (image-to-video)
- Kling (image-to-video)
- MiniMax (image-to-video)
- WAN (text-to-video)
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional, Tuple

import folder_paths
import torch

from .base import FalVideoModelNode
from ..core.utils import (
    upload_image_tensor,
    download_video,
    save_video_temp,
)


class LumaImageToVideo_fal(FalVideoModelNode):
    """
    Luma Dream Machine image-to-video generation.

    Supports start/end frames for controlled video generation.
    Uses fal-ai/luma-dream-machine/image-to-video endpoint.
    """

    ENDPOINT = "fal-ai/luma-dream-machine/image-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the video motion/action"
                }),
            },
            "optional": {
                "first_image": ("IMAGE", {
                    "tooltip": "Starting frame for the video"
                }),
                "last_image": ("IMAGE", {
                    "tooltip": "Ending frame for the video"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {
                    "default": "16:9"
                }),
                "duration": (["5s", "9s"], {
                    "default": "5s",
                    "tooltip": "Video duration"
                }),
                "loop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Create seamless loop"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed (0 for random)"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        first_image: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        aspect_ratio: str = "16:9",
        duration: str = "5s",
        loop: bool = False,
        seed: int = 0,
    ) -> Tuple[Any, str]:
        """Generate video from images using Luma Dream Machine."""

        # Build payload
        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "loop": loop,
        }

        # Upload images if provided
        if first_image is not None:
            payload["image_url"] = upload_image_tensor(first_image)

        if last_image is not None:
            payload["end_image_url"] = upload_image_tensor(last_image)

        # Add seed if non-zero
        if seed > 0:
            payload["seed"] = seed

        # Run the model
        result = self.run_model(payload)

        # Get video URL from result
        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise ValueError("No video URL in response")

        # Download video and save to output folder
        _, video_bytes = download_video(video_url)
        video_path = self._save_video(video_bytes, "luma")

        # Return video path and URL
        return (video_path, video_url)

    def _save_video(self, video_bytes: bytes, prefix: str) -> str:
        """Save video bytes to output folder."""
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "fal_videos")
        os.makedirs(video_dir, exist_ok=True)

        # Generate unique filename
        import time
        filename = f"{prefix}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        return video_path


class KlingImageToVideo_fal(FalVideoModelNode):
    """
    Kling image-to-video generation.

    High quality video generation with start/end frame support.
    Uses fal-ai/kling-video/v1.5/pro/image-to-video endpoint.
    """

    ENDPOINT = "fal-ai/kling-video/v1.5/pro/image-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the video motion/action"
                }),
                "start_image": ("IMAGE", {
                    "tooltip": "Starting frame for the video"
                }),
            },
            "optional": {
                "end_image": ("IMAGE", {
                    "tooltip": "Optional ending frame"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "What to avoid in the video"
                }),
                "duration": (["5", "10"], {
                    "default": "5",
                    "tooltip": "Video duration in seconds"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed (0 for random)"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        start_image: torch.Tensor,
        end_image: Optional[torch.Tensor] = None,
        negative_prompt: str = "",
        duration: str = "5",
        aspect_ratio: str = "16:9",
        seed: int = 0,
    ) -> Tuple[Any, str]:
        """Generate video using Kling."""

        # Upload start image
        image_url = upload_image_tensor(start_image)

        # Build payload
        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
        }

        # Add optional parameters
        if end_image is not None:
            payload["tail_image_url"] = upload_image_tensor(end_image)

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if seed > 0:
            payload["seed"] = seed

        # Run the model
        result = self.run_model(payload)

        # Get video URL
        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise ValueError("No video URL in response")

        # Download and save
        _, video_bytes = download_video(video_url)
        video_path = self._save_video(video_bytes, "kling")

        return (video_path, video_url)

    def _save_video(self, video_bytes: bytes, prefix: str) -> str:
        """Save video bytes to output folder."""
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "fal_videos")
        os.makedirs(video_dir, exist_ok=True)

        import time
        filename = f"{prefix}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        return video_path


class MinimaxImageToVideo_fal(FalVideoModelNode):
    """
    MiniMax image-to-video generation.

    Fast video generation from a single image.
    Uses fal-ai/minimax-video/image-to-video endpoint.
    """

    ENDPOINT = "fal-ai/minimax-video/image-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the video motion/action"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Source image for video generation"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        image: torch.Tensor,
        seed: int = 0,
    ) -> Tuple[Any, str]:
        """Generate video using MiniMax."""

        image_url = upload_image_tensor(image)

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_url": image_url,
        }

        if seed > 0:
            payload["seed"] = seed

        result = self.run_model(payload)

        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise ValueError("No video URL in response")

        _, video_bytes = download_video(video_url)
        video_path = self._save_video(video_bytes, "minimax")

        return (video_path, video_url)

    def _save_video(self, video_bytes: bytes, prefix: str) -> str:
        """Save video bytes to output folder."""
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "fal_videos")
        os.makedirs(video_dir, exist_ok=True)

        import time
        filename = f"{prefix}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        return video_path


class WanTextToVideo_fal(FalVideoModelNode):
    """
    WAN text-to-video generation.

    Generate videos from text prompts without input images.
    Uses fal-ai/wan/v2.1/1.3b/text-to-video endpoint.
    """

    ENDPOINT = "fal-ai/wan/v2.1/1.3b/text-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the video to generate"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "aspect_ratio": (["16:9", "9:16"], {
                    "default": "16:9"
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 2,
                    "max": 40,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "16:9",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 0,
    ) -> Tuple[Any, str]:
        """Generate video from text using WAN."""

        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if seed > 0:
            payload["seed"] = seed

        result = self.run_model(payload)

        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise ValueError("No video URL in response")

        _, video_bytes = download_video(video_url)
        video_path = self._save_video(video_bytes, "wan")

        return (video_path, video_url)

    def _save_video(self, video_bytes: bytes, prefix: str) -> str:
        """Save video bytes to output folder."""
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "fal_videos")
        os.makedirs(video_dir, exist_ok=True)

        import time
        filename = f"{prefix}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        return video_path


class ViduReferenceToVideo_fal(FalVideoModelNode):
    """
    Vidu Q1 Reference-to-Video generation.

    Generate videos with consistent subjects from reference images.
    Supports up to 7 reference images for character/object consistency.
    Uses fal-ai/vidu/q1/reference-to-video endpoint.
    """

    ENDPOINT = "fal-ai/vidu/q1/reference-to-video"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Description of the video to generate (max 1500 chars)"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Primary reference image for subject consistency"
                }),
            },
            "optional": {
                "reference_image_2": ("IMAGE", {
                    "tooltip": "Additional reference image (optional)"
                }),
                "reference_image_3": ("IMAGE", {
                    "tooltip": "Additional reference image (optional)"
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1"], {
                    "default": "16:9"
                }),
                "movement_amplitude": (["auto", "small", "medium", "large"], {
                    "default": "auto",
                    "tooltip": "Movement intensity of objects in the video"
                }),
                "bgm": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Add background music to the video"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed (0 for random)"
                }),
            }
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_url")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def generate(
        self,
        prompt: str,
        reference_image: torch.Tensor,
        reference_image_2: Optional[torch.Tensor] = None,
        reference_image_3: Optional[torch.Tensor] = None,
        aspect_ratio: str = "16:9",
        movement_amplitude: str = "auto",
        bgm: bool = False,
        seed: int = 0,
    ) -> Tuple[Any, str]:
        """Generate video with consistent subjects using Vidu Q1."""

        # Upload reference images
        reference_urls = [upload_image_tensor(reference_image)]

        if reference_image_2 is not None:
            reference_urls.append(upload_image_tensor(reference_image_2))

        if reference_image_3 is not None:
            reference_urls.append(upload_image_tensor(reference_image_3))

        # Build payload
        payload: dict[str, Any] = {
            "prompt": prompt[:1500],  # Max 1500 chars
            "reference_image_urls": reference_urls,
            "aspect_ratio": aspect_ratio,
            "movement_amplitude": movement_amplitude,
            "bgm": bgm,
        }

        if seed > 0:
            payload["seed"] = seed

        # Run the model
        result = self.run_model(payload)

        # Get video URL
        video_data = result.get("video", {})
        video_url = video_data.get("url", "")

        if not video_url:
            raise ValueError("No video URL in response")

        # Download and save
        _, video_bytes = download_video(video_url)
        video_path = self._save_video(video_bytes, "vidu_ref")

        return (video_path, video_url)

    def _save_video(self, video_bytes: bytes, prefix: str) -> str:
        """Save video bytes to output folder."""
        output_dir = folder_paths.get_output_directory()
        video_dir = os.path.join(output_dir, "fal_videos")
        os.makedirs(video_dir, exist_ok=True)

        import time
        filename = f"{prefix}_{int(time.time())}.mp4"
        video_path = os.path.join(video_dir, filename)

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        return video_path


# Node mappings
FAL_VIDEO_NODES = {
    "LumaImageToVideo_fal": LumaImageToVideo_fal,
    "KlingImageToVideo_fal": KlingImageToVideo_fal,
    "MinimaxImageToVideo_fal": MinimaxImageToVideo_fal,
    "WanTextToVideo_fal": WanTextToVideo_fal,
    "ViduReferenceToVideo_fal": ViduReferenceToVideo_fal,
}

FAL_VIDEO_DISPLAY_NAMES = {
    "LumaImageToVideo_fal": "Luma Image to Video (fal.ai)",
    "KlingImageToVideo_fal": "Kling Image to Video (fal.ai)",
    "MinimaxImageToVideo_fal": "MiniMax Image to Video (fal.ai)",
    "WanTextToVideo_fal": "WAN Text to Video (fal.ai)",
    "ViduReferenceToVideo_fal": "Vidu Reference to Video (fal.ai)",
}
