"""Live streaming: FastAPI hub + WebSocket + browser overlay.

The overlay page reads ``/frame.jpg`` (annotated preview) and ``/ws`` (world
state JSON at ~20 Hz) and draws boxes / HP bars / velocity arrows on top.
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

OVERLAY_HTML = Path(__file__).resolve().parent.parent / "overlay" / "index.html"


class Hub:
    """Thread-safe holder of the latest pipeline output."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state: dict[str, Any] = {}
        self.jpg: Optional[bytes] = None
        self.minimap_jpg: Optional[bytes] = None
        self.clients: set[WebSocket] = set()
        self.frame_count = 0

    def update(
        self,
        state: dict[str, Any],
        frame_bgr: np.ndarray,
        minimap_bgr: Optional[np.ndarray] = None,
    ) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
        data = buf.tobytes() if ok else None
        mm_data = None
        if minimap_bgr is not None and minimap_bgr.size:
            ok2, buf2 = cv2.imencode(".jpg", minimap_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            mm_data = buf2.tobytes() if ok2 else None
        with self._lock:
            self.state = state
            if data is not None:
                self.jpg = data
            if mm_data is not None:
                self.minimap_jpg = mm_data
            self.frame_count += 1

    def snapshot(self) -> tuple[dict[str, Any], Optional[bytes]]:
        with self._lock:
            return self.state, self.jpg

    def minimap_snapshot(self) -> Optional[bytes]:
        with self._lock:
            return self.minimap_jpg


def create_app(hub: Hub) -> FastAPI:
    app = FastAPI(title="GenshinVision")

    @app.get("/")
    def index():
        return FileResponse(str(OVERLAY_HTML))

    @app.get("/state")
    def state():
        return JSONResponse(hub.snapshot()[0])

    @app.get("/frame.jpg")
    def frame():
        _, jpg = hub.snapshot()
        if jpg is None:
            return Response(status_code=503)
        return Response(content=jpg, media_type="image/jpeg")

    @app.get("/minimap.jpg")
    def minimap_img():
        jpg = hub.minimap_snapshot()
        if jpg is None:
            return Response(status_code=503)
        return Response(content=jpg, media_type="image/jpeg")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        try:
            while True:
                state, _ = hub.snapshot()
                await ws.send_json(state)
                await asyncio.sleep(0.05)  # ~20 Hz
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(ws)

    return app


def start_server(hub: Hub, host: str = "0.0.0.0", port: int = 8000, log_level: str = "warning"):
    """Runs uvicorn in a daemon thread; returns the (started) server object."""
    import uvicorn

    app = create_app(hub)
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, ws_ping_interval=20)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="genshinvision-uvicorn")
    thread.start()
    deadline = time.time() + 15.0
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"Не удалось запустить сервер на {host}:{port}")
    return server


def pump(pipeline, hub: Hub, target_fps: float = 30.0, max_frames: int = 0, warmup: int = 3) -> int:
    """Run the pipeline forever, feeding the hub at ``target_fps``.

    ``warmup`` lets the tracker accumulate hits before the first broadcast.
    """
    interval = 1.0 / max(target_fps, 1.0)
    n = 0
    while True:
        t0 = time.perf_counter()
        out = pipeline.step()
        if out is None:
            break
        n += 1
        if n > warmup:
            hub.update(out.state.to_dict(), out.frame, minimap_bgr=out.minimap_crop)
        if max_frames and n >= max_frames:
            break
        elapsed = time.perf_counter() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
    return n
