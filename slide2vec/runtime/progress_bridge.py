from contextlib import contextmanager

from hs2p import progress as hs2p_progress

from slide2vec.progress import (
    NullProgressReporter,
    ProgressEvent as Slide2VecProgressEvent,
    get_progress_reporter,
)

_BRIDGED_HS2P_PROGRESS_KINDS = {
    "backend.selected",
    "tissue.started",
    "tissue.progress",
    "tissue.finished",
    "tiling.progress",
    "tiling.finished",
    "preview.started",
    "preview.progress",
    "preview.finished",
}


class _Hs2pProgressBridge:
    def __init__(self, downstream) -> None:
        self._downstream = downstream

    def emit(self, event) -> None:
        if event.kind not in _BRIDGED_HS2P_PROGRESS_KINDS:
            return
        self._downstream.emit(
            Slide2VecProgressEvent(kind=event.kind, payload=dict(event.payload))
        )

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        if hasattr(self._downstream, "write_log"):
            self._downstream.write_log(message, stream=stream)


@contextmanager
def bridge_hs2p_progress_to_slide2vec():
    downstream = get_progress_reporter()
    if isinstance(downstream, NullProgressReporter):
        yield
        return
    bridge = _Hs2pProgressBridge(downstream)
    with hs2p_progress.activate_progress_reporter(bridge):
        yield

