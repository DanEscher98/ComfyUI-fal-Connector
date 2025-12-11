"""
Image generation nodes for fal.ai models.

Provides nodes for:
- FLUX.1 Dev (high quality)
- FLUX.1 Schnell (fast)
- FLUX Pro Ultra (2K resolution)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .base import FalImageModelNode
from ..core.utils import download_image


class FluxDev_fal(FalImageModelNode):
    """
    FLUX.1 [dev] text-to-image generation.

    High-quality 12B parameter model for detailed image generation.
    Uses fal-ai/flux/dev endpoint.
    """

    ENDPOINT = "fal-ai/flux/dev"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Text description of the image to generate"
                }),
            },
            "optional": {
                "image_size": ([
                    "square_hd",      # 1024x1024
                    "square",         # 512x512
                    "portrait_4_3",   # 768x1024
                    "portrait_16_9",  # 576x1024
                    "landscape_4_3",  # 1024x768
                    "landscape_16_9", # 1024x576
                ], {
                    "default": "landscape_16_9",
                    "tooltip": "Output image size preset"
                }),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 1,
                    "max": 50,
                    "tooltip": "Number of denoising steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "How closely to follow the prompt"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "tooltip": "Number of images to generate"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed (0 for random)"
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"

    def generate(
        self,
        prompt: str,
        image_size: str = "landscape_16_9",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images: int = 1,
        seed: int = 0,
        enable_safety_checker: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate images using FLUX.1 Dev."""

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed > 0:
            payload["seed"] = seed

        result = self.run_model(payload)

        # Get images from result
        images_data = result.get("images", [])

        if not images_data:
            raise ValueError("No images in response")

        # Download and stack images
        tensors = []
        for img_data in images_data:
            url = img_data.get("url", "")
            if url:
                tensor = download_image(url)
                tensors.append(tensor)

        if not tensors:
            raise ValueError("Failed to download any images")

        # Stack into batch [B, H, W, C]
        output = torch.cat(tensors, dim=0)

        return (output,)


class FluxSchnell_fal(FalImageModelNode):
    """
    FLUX.1 [schnell] fast text-to-image generation.

    Optimized for speed (1-4 steps).
    Uses fal-ai/flux/schnell endpoint.
    """

    ENDPOINT = "fal-ai/flux/schnell"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
            "optional": {
                "image_size": ([
                    "square_hd",
                    "square",
                    "portrait_4_3",
                    "portrait_16_9",
                    "landscape_4_3",
                    "landscape_16_9",
                ], {
                    "default": "landscape_16_9",
                }),
                "num_inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 12,
                    "tooltip": "1-4 steps recommended for speed"
                }),
                "num_images": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"

    def generate(
        self,
        prompt: str,
        image_size: str = "landscape_16_9",
        num_inference_steps: int = 4,
        num_images: int = 1,
        seed: int = 0,
        enable_safety_checker: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Generate images using FLUX.1 Schnell."""

        payload: dict[str, Any] = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }

        if seed > 0:
            payload["seed"] = seed

        result = self.run_model(payload)

        images_data = result.get("images", [])

        if not images_data:
            raise ValueError("No images in response")

        tensors = []
        for img_data in images_data:
            url = img_data.get("url", "")
            if url:
                tensor = download_image(url)
                tensors.append(tensor)

        if not tensors:
            raise ValueError("Failed to download any images")

        output = torch.cat(tensors, dim=0)

        return (output,)


class FluxProUltra_fal(FalImageModelNode):
    """
    FLUX1.1 [pro] ultra - high resolution up to 2K.

    Professional grade with improved photo realism.
    Uses fal-ai/flux-pro/v1.1-ultra endpoint.
    """

    ENDPOINT = "fal-ai/flux-pro/v1.1-ultra"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
            "optional": {
                "aspect_ratio": ([
                    "21:9",  # Ultra-wide
                    "16:9",
                    "4:3",
                    "1:1",
                    "3:4",
                    "9:16",
                    "9:21",
                ], {
                    "default": "16:9",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2147483647,
                }),
                "enable_safety_checker": ("BOOLEAN", {
                    "default": True,
                }),
                "output_format": (["jpeg", "png"], {
                    "default": "jpeg",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate"

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        seed: int = 0,
        enable_safety_checker: bool = True,
        output_format: str = "jpeg",
    ) -> Tuple[torch.Tensor]:
        """Generate high-resolution images using FLUX Pro Ultra."""

        payload: dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "enable_safety_checker": enable_safety_checker,
            "output_format": output_format,
        }

        if seed > 0:
            payload["seed"] = seed

        result = self.run_model(payload)

        images_data = result.get("images", [])

        if not images_data:
            raise ValueError("No images in response")

        tensors = []
        for img_data in images_data:
            url = img_data.get("url", "")
            if url:
                tensor = download_image(url)
                tensors.append(tensor)

        if not tensors:
            raise ValueError("Failed to download any images")

        output = torch.cat(tensors, dim=0)

        return (output,)


# Node mappings
FAL_IMAGE_NODES = {
    "FluxDev_fal": FluxDev_fal,
    "FluxSchnell_fal": FluxSchnell_fal,
    "FluxProUltra_fal": FluxProUltra_fal,
}

FAL_IMAGE_DISPLAY_NAMES = {
    "FluxDev_fal": "FLUX.1 Dev (fal.ai)",
    "FluxSchnell_fal": "FLUX.1 Schnell (fal.ai)",
    "FluxProUltra_fal": "FLUX Pro Ultra 2K (fal.ai)",
}
