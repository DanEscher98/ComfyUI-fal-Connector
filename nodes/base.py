"""
Base class for fal.ai model nodes.

Provides common functionality for all fal.ai nodes:
- Client access
- Error handling
- Progress reporting
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from ..core.client import FalClient, FalClientError, get_client


class FalModelNode:
    """
    Base class for fal.ai model nodes.

    Subclasses should define:
    - ENDPOINT: The fal.ai model endpoint
    - INPUT_TYPES(): Input specification
    - RETURN_TYPES: Output types tuple
    - FUNCTION: Method name to call
    - generate() or other method: The actual implementation
    """

    # Override in subclass
    ENDPOINT: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = "fal.ai"

    # Default timeout for model calls (5 minutes)
    DEFAULT_TIMEOUT: ClassVar[float] = 300.0

    # Default poll interval (2 seconds)
    DEFAULT_POLL_INTERVAL: ClassVar[float] = 2.0

    @classmethod
    def get_client(cls) -> FalClient:
        """Get the fal.ai client instance."""
        return get_client()

    @classmethod
    def run_model(
        cls,
        payload: dict[str, Any],
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Run the model with the given payload.

        Args:
            payload: Request payload for the model
            timeout: Optional timeout override
            poll_interval: Optional poll interval override

        Returns:
            Model result dict

        Raises:
            FalClientError: On API errors
        """
        if not cls.ENDPOINT:
            raise ValueError(f"{cls.__name__} has no ENDPOINT defined")

        client = cls.get_client()

        return client.run_sync(
            endpoint=cls.ENDPOINT,
            payload=payload,
            timeout=timeout or cls.DEFAULT_TIMEOUT,
            poll_interval=poll_interval or cls.DEFAULT_POLL_INTERVAL,
        )


class FalImageModelNode(FalModelNode):
    """Base class for image generation nodes."""

    CATEGORY = "fal.ai/image"
    RETURN_TYPES = ("IMAGE",)


class FalVideoModelNode(FalModelNode):
    """Base class for video generation nodes."""

    CATEGORY = "fal.ai/video"
    RETURN_TYPES = ("VIDEO",)

    # Videos take longer
    DEFAULT_TIMEOUT: ClassVar[float] = 600.0


class FalAudioModelNode(FalModelNode):
    """Base class for audio nodes."""

    CATEGORY = "fal.ai/audio"
