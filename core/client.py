"""
fal.ai API client with queue-based polling support.

Uses the queue.fal.run endpoint for reliable long-running operations.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Optional

import fal_client
import httpx

from ..config import get_fal_api_key


class FalClientError(Exception):
    """Exception raised for fal.ai API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class FalClient:
    """
    Async fal.ai client with queue-based polling support.

    Uses queue.fal.run for reliable execution of long-running models.
    """

    QUEUE_BASE_URL = "https://queue.fal.run"

    def __init__(self):
        self._api_key: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def api_key(self) -> str:
        """Get cached API key."""
        if self._api_key is None:
            self._api_key = get_fal_api_key()
        return self._api_key

    @property
    def headers(self) -> dict[str, str]:
        """Get authorization headers."""
        return {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=120.0)
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        """
        Submit a request to the queue.

        Args:
            endpoint: Model endpoint (e.g., "fal-ai/flux/dev")
            payload: Request payload

        Returns:
            request_id for polling
        """
        client = await self._get_client()
        url = f"{self.QUEUE_BASE_URL}/{endpoint}"

        response = await client.post(url, json=payload, headers=self.headers)

        if response.status_code != 200:
            raise FalClientError(
                f"Failed to submit request: {response.text}",
                status_code=response.status_code,
            )

        data = response.json()
        request_id = data.get("request_id")

        if not request_id:
            raise FalClientError("No request_id in response", details=data)

        return request_id

    async def get_status(self, endpoint: str, request_id: str) -> dict[str, Any]:
        """
        Get the status of a queued request.

        Args:
            endpoint: Model endpoint
            request_id: Request ID from submit()

        Returns:
            Status dict with 'status' field
        """
        client = await self._get_client()
        url = f"{self.QUEUE_BASE_URL}/{endpoint}/requests/{request_id}/status"

        response = await client.get(url, headers=self.headers)

        if response.status_code != 200:
            raise FalClientError(
                f"Failed to get status: {response.text}",
                status_code=response.status_code,
            )

        return response.json()

    async def get_result(self, endpoint: str, request_id: str) -> dict[str, Any]:
        """
        Get the result of a completed request.

        Args:
            endpoint: Model endpoint
            request_id: Request ID from submit()

        Returns:
            Result dict from the model
        """
        client = await self._get_client()
        url = f"{self.QUEUE_BASE_URL}/{endpoint}/requests/{request_id}"

        response = await client.get(url, headers=self.headers)

        if response.status_code != 200:
            raise FalClientError(
                f"Failed to get result: {response.text}",
                status_code=response.status_code,
            )

        return response.json()

    async def poll(
        self,
        endpoint: str,
        request_id: str,
        timeout: float = 300.0,
        interval: float = 2.0,
        on_status: Optional[callable] = None,
    ) -> dict[str, Any]:
        """
        Poll until request completes.

        Args:
            endpoint: Model endpoint
            request_id: Request ID from submit()
            timeout: Maximum time to wait in seconds
            interval: Time between polls in seconds
            on_status: Optional callback for status updates

        Returns:
            Final result from the model
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout:
                raise FalClientError(
                    f"Request timed out after {timeout}s",
                    details={"request_id": request_id, "elapsed": elapsed},
                )

            status_data = await self.get_status(endpoint, request_id)
            status = status_data.get("status", "").upper()

            if on_status:
                on_status(status, elapsed, status_data)

            if status == "COMPLETED":
                return await self.get_result(endpoint, request_id)

            if status in ("FAILED", "CANCELLED"):
                error_msg = status_data.get("error", "Unknown error")
                raise FalClientError(
                    f"Request {status.lower()}: {error_msg}",
                    details=status_data,
                )

            # Status is IN_QUEUE or IN_PROGRESS, keep polling
            await asyncio.sleep(interval)

    async def run(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float = 300.0,
        poll_interval: float = 2.0,
        on_status: Optional[callable] = None,
    ) -> dict[str, Any]:
        """
        Submit and poll until completion.

        This is the main method for running models.

        Args:
            endpoint: Model endpoint (e.g., "fal-ai/flux/dev")
            payload: Request payload
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks
            on_status: Optional callback for status updates

        Returns:
            Result dict from the model
        """
        request_id = await self.submit(endpoint, payload)
        return await self.poll(
            endpoint,
            request_id,
            timeout=timeout,
            interval=poll_interval,
            on_status=on_status,
        )

    def run_sync(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """
        Synchronous version of run() for ComfyUI nodes.

        ComfyUI nodes run in a thread pool, so we need a sync interface.
        """
        return asyncio.run(
            self.run(endpoint, payload, timeout=timeout, poll_interval=poll_interval)
        )

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file to fal.ai storage.

        Args:
            file_path: Local file path

        Returns:
            URL of uploaded file
        """
        return fal_client.upload_file(file_path)


# Global client instance (lazy initialized)
_client: Optional[FalClient] = None


def get_client() -> FalClient:
    """Get the global FalClient instance."""
    global _client
    if _client is None:
        _client = FalClient()
    return _client
