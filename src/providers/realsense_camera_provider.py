# realsense_camera_provider.py

from __future__ import annotations

import time
import logging
import threading
import numpy as np
import pyrealsense2 as rs

from dataclasses import dataclass
from typing import Optional

from .singleton import singleton


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    fx, fy : color stream 초점 거리 (픽셀)
    cx, cy : color stream 주점 (픽셀)
    """
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraFrame:
    """
    t_monotonic  : wait_for_frames 반환 직후 time.monotonic() 값 — sync 기준 시각
    bgr          : (H, W, 3) uint8   — BGR 컬러 이미지
    depth        : (H, W)    float32 — depth (meter), color frame에 align됨
    intrinsics   : color stream 내부 파라미터
    camera_fps   : 카메라 FPS
    frame_cnt    : 프레임 순서 카운터 (0부터 단조 증가) — 동기화 기준
    """
    t_monotonic: float
    bgr: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    camera_fps: float
    frame_cnt: int


@singleton
class RealSenseCameraProvider:
    """
    컬러(BGR) + depth(컬러 align) stream을 background thread에서 지속 수신하며,
    최신 프레임을 data property로 노출
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        self._data: Optional[CameraFrame] = None
        self._lock = threading.Lock()

        self.running = False
        self._thread: Optional[threading.Thread] = None

        self._pipeline = None
        self._config = None
        self._profile = None
        self._align = None
        self._depth_scale: Optional[float] = None
        self._intrinsics_cache: Optional[CameraIntrinsics] = None
        self._frame_cnt: int = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logging.warning("RealSenseCameraProvider already running")
            return
        self._start_realsense_sdk()
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # 첫 프레임 도착까지 대기 — 반환 후 data가 항상 CameraFrame을 보장
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with self._lock:
                if self._data is not None:
                    break
            time.sleep(0.01)
        else:
            raise RuntimeError("RealSenseCameraProvider: timed out waiting for first frame")
        
        logging.info("RealSenseCameraProvider started")

    def stop(self) -> None:
        self.running = False

        if self._pipeline is not None:  # _stop_realsense_sdk
            try:
                self._pipeline.stop()
            except Exception as e:
                logging.warning(f"RealSenseCameraProvider pipeline stop error: {e}")

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

        self._cleanup_realsense_sdk()
        logging.info("RealSenseCameraProvider stopped")

    def _start_realsense_sdk(self) -> None:
        if self._pipeline is not None:
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense device connected")
        if not (0 <= self.camera_index < len(devices)):
            raise RuntimeError(
                f"camera_index={self.camera_index} out of range "
                f"(detected devices: {len(devices)})"
            )

        serial = devices[self.camera_index].get_info(rs.camera_info.serial_number)

        self._config = rs.config()
        self._config.enable_device(serial)
        self._config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self._config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )

        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(self._config)

        depth_sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        
        self._align = rs.align(rs.stream.color)

        # warm-up: auto-exposure 안정화
        for _ in range(100):
            self._pipeline.wait_for_frames()

    def _cleanup_realsense_sdk(self) -> None:
        self._pipeline = None
        self._config = None
        self._profile = None
        self._align = None
        self._depth_scale = None
        self._intrinsics_cache = None
        self._frame_cnt = 0

    def _read_frame(self) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError("Pipeline not started. Call start() first")

        frames = self._pipeline.wait_for_frames(timeout_ms=100)
        timestamp = time.monotonic()  # 프레임 도착 후 캡처 — sync 기준 시각

        frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Color or depth frame missing")

        bgr: np.ndarray = np.asanyarray(color_frame.get_data()).copy()

        depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m *= float(self._depth_scale)

        camera_fps = float(color_frame.get_frame_metadata(rs.frame_metadata_value.actual_fps)) * 0.001  # raw는 mili-fps 단위
        self._frame_cnt += 1

        return CameraFrame(
            t_monotonic=timestamp,
            bgr=bgr,
            depth=depth_m,
            intrinsics=self._get_intrinsics(),
            camera_fps=camera_fps,
            frame_cnt=self._frame_cnt,
        )

    def _get_intrinsics(self) -> CameraIntrinsics:
        """처음 호출 시 color stream intrinsics를 캐싱"""
        if self._intrinsics_cache is not None:
            return self._intrinsics_cache
        try:
            stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
            camera_intrinsics = stream.get_intrinsics()
            self._intrinsics_cache = CameraIntrinsics(
                fx=float(camera_intrinsics.fx),
                fy=float(camera_intrinsics.fy),
                cx=float(camera_intrinsics.ppx),
                cy=float(camera_intrinsics.ppy),
            )
        except Exception as e:
            logging.warning(f"RealSenseCameraProvider: failed to read intrinsics: {e}")
            self._intrinsics_cache = CameraIntrinsics(fx=0.0, fy=0.0, cx=0.0, cy=0.0)
        return self._intrinsics_cache

    @property
    def data(self) -> Optional[CameraFrame]:
        """최신 CameraFrame. 첫 프레임 도착 전에는 None."""
        with self._lock:
            return self._data

    def _run(self) -> None:
        while self.running:
            try:
                frame = self._read_frame()
                with self._lock:
                    self._data = frame
            except Exception as e:
                logging.error(f"RealSenseCameraProvider: run loop error: {e}")
                with self._lock:
                    self._data = None