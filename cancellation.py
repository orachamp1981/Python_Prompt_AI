"""Thread-safe cancellation signal for a running prompt."""

import threading
from typing import Callable, Optional


class PromptCancelled(Exception):
    """Raised when a user stops an active prompt."""


class CancellationToken:
    """Cancellation event that can interrupt a currently blocking operation."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callback: Optional[Callable[[], None]] = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self.cancelled:
            raise PromptCancelled("Prompt cancelled.")

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callback = self._callback
        if callback is not None:
            callback()

    def attach_cancel_callback(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._event.is_set():
                cancel_now = True
            else:
                self._callback = callback
                cancel_now = False
        if cancel_now:
            callback()

    def clear_cancel_callback(self) -> None:
        with self._lock:
            self._callback = None
