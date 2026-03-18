# src/data_collector/segmentation_data_collector.py

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from providers.realsense_camera_provider import CameraFrame, RealSenseCameraProvider


class SegmentationDataCollector:
    """
    RealSenseCameraProvider로부터 최신 프레임을 받아
    semantic segmentation 학습용 raw dataset을 저장하는 수집기.

    저장 구조 예시:
        data/indoor_semantic_segmentation/
          raw/
            images/
              000001.png
            depth/
              000001.npy
            meta/
              000001.json
            manifest.csv

    Notes
    -----
    - RGB 학습용 이미지: PNG/JPG로 저장
    - Depth: meter 단위 float32 그대로 .npy 저장
    - Meta: 학습에 필요한 최소 정보만 JSON 저장
    - 동일 frame_cnt 중복 저장 방지 (내부 로직)
    - sample_period_sec로 샘플링 간격 제어 가능 (내부 로직)
    - scene, building, floor 등의 공통 메타데이터를 모든 샘플에 자동 기록 가능
    """

    def __init__(
        self,
        output_dir: str | Path,
        camera_provider: RealSenseCameraProvider,
        sample_period_sec: float = 0.5,
        save_depth: bool = True,
        image_ext: str = ".png",
        common_metadata: Optional[dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.camera_provider = camera_provider
        self.sample_period_sec = float(sample_period_sec)
        self.save_depth = bool(save_depth)
        self.image_ext = image_ext.lower()
        self.common_metadata = dict(common_metadata) if common_metadata is not None else {}

        if self.image_ext not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image_ext: {self.image_ext}")

        self.raw_dir = self.output_dir / "raw"
        self.images_dir = self.raw_dir / "images"
        self.depth_dir = self.raw_dir / "depth"
        self.meta_dir = self.raw_dir / "meta"
        self.manifest_path = self.raw_dir / "manifest.csv"

        self._stop_requested = False
        self._last_saved_frame_cnt: Optional[int] = None
        self._last_saved_t_monotonic: Optional[float] = None
        self._next_sample_index: int = 1

        self._prepare_dirs()
        self._next_sample_index = self._discover_next_sample_index()
        self._ensure_manifest_header()

    def _prepare_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        if self.save_depth:
            self.depth_dir.mkdir(parents=True, exist_ok=True)

    def _discover_next_sample_index(self) -> int:
        """
        images 디렉터리의 기존 파일명을 보고 다음 sample index를 계산한다.
        파일 stem이 숫자인 경우만 반영한다.
        """
        max_index = 0
        for path in self.images_dir.iterdir():
            if not path.is_file():
                continue
            try:
                max_index = max(max_index, int(path.stem))
            except ValueError:
                continue
        return max_index + 1

    def _ensure_manifest_header(self) -> None:
        if self.manifest_path.exists() and self.manifest_path.stat().st_size > 0:
            return

        with self.manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "image_path",
                    "depth_path",
                    "meta_path",
                    "width",
                    "height",
                ],
            )
            writer.writeheader()

    def request_stop(self) -> None:
        self._stop_requested = True

    def _format_sample_id(self, sample_index: int) -> str:
        return f"{sample_index:06d}"

    def _should_save(self, frame: CameraFrame) -> bool:
        """
        내부 로직용:
        - 동일 frame_cnt 중복 저장 방지
        - sample_period_sec 간격 유지
        """
        if self._last_saved_frame_cnt is not None and frame.frame_cnt == self._last_saved_frame_cnt:
            return False

        if self.sample_period_sec <= 0.0:
            return True

        if self._last_saved_t_monotonic is None:
            return True

        dt = frame.t_monotonic - self._last_saved_t_monotonic
        return dt >= self.sample_period_sec

    def _build_metadata(
        self,
        sample_id: str,
        image_rel_path: str,
        depth_rel_path: Optional[str],
        frame: CameraFrame,
    ) -> dict[str, Any]:
        height, width = frame.bgr.shape[:2]

        metadata: dict[str, Any] = {
            "sample_id": sample_id,
            "image_path": image_rel_path,
            "depth_path": depth_rel_path,
            "image_height": int(height),
            "image_width": int(width),
        }

        metadata.update(self.common_metadata)
        return metadata

    def _append_manifest_row(
        self,
        sample_id: str,
        image_rel_path: str,
        depth_rel_path: Optional[str],
        meta_rel_path: str,
        frame: CameraFrame,
    ) -> None:
        height, width = frame.bgr.shape[:2]

        with self.manifest_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "image_path",
                    "depth_path",
                    "meta_path",
                    "width",
                    "height",
                ],
            )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "image_path": image_rel_path,
                    "depth_path": depth_rel_path or "",
                    "meta_path": meta_rel_path,
                    "width": width,
                    "height": height,
                }
            )

    def save_frame(self, frame: CameraFrame) -> str:
        """
        단일 프레임을 즉시 저장하고 sample_id를 반환한다.
        """
        sample_id = self._format_sample_id(self._next_sample_index)

        image_filename = f"{sample_id}{self.image_ext}"
        depth_filename = f"{sample_id}.npy"
        meta_filename = f"{sample_id}.json"

        image_path = self.images_dir / image_filename
        depth_path = self.depth_dir / depth_filename
        meta_path = self.meta_dir / meta_filename

        image_rel_path = str(image_path.relative_to(self.output_dir))
        depth_rel_path = str(depth_path.relative_to(self.output_dir)) if self.save_depth else None
        meta_rel_path = str(meta_path.relative_to(self.output_dir))

        success = cv2.imwrite(str(image_path), frame.bgr)
        if not success:
            raise RuntimeError(f"Failed to write image: {image_path}")

        if self.save_depth:
            np.save(depth_path, frame.depth.astype(np.float32), allow_pickle=False)

        metadata = self._build_metadata(
            sample_id=sample_id,
            image_rel_path=image_rel_path,
            depth_rel_path=depth_rel_path,
            frame=frame,
        )
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self._append_manifest_row(
            sample_id=sample_id,
            image_rel_path=image_rel_path,
            depth_rel_path=depth_rel_path,
            meta_rel_path=meta_rel_path,
            frame=frame,
        )

        self._last_saved_frame_cnt = frame.frame_cnt
        self._last_saved_t_monotonic = frame.t_monotonic
        self._next_sample_index += 1

        logging.info(
            "Saved sample_id=%s image=%s depth=%s",
            sample_id,
            image_path.name,
            depth_path.name if self.save_depth else "disabled",
        )
        return sample_id

    def collect(
        self,
        max_samples: Optional[int] = None,
        max_duration_sec: Optional[float] = None,
        poll_interval_sec: float = 0.005,
    ) -> int:
        """
        provider에서 최신 프레임을 계속 읽으면서 조건에 맞는 프레임을 저장한다.

        Parameters
        ----------
        max_samples:
            최대 저장 샘플 수. None이면 제한 없음.
        max_duration_sec:
            최대 수집 시간(초). None이면 제한 없음.
        poll_interval_sec:
            provider.data polling 간격.

        Returns
        -------
        int
            이번 collect() 호출에서 저장된 샘플 수
        """
        self._stop_requested = False
        saved_count = 0
        start_time = time.monotonic()

        logging.info(
            "SegmentationDataCollector started: output_dir=%s, sample_period_sec=%.3f, save_depth=%s, common_metadata=%s",
            self.output_dir,
            self.sample_period_sec,
            self.save_depth,
            self.common_metadata,
        )

        while not self._stop_requested:
            if max_samples is not None and saved_count >= max_samples:
                break

            if max_duration_sec is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= max_duration_sec:
                    break

            frame = self.camera_provider.data
            if frame is None:
                time.sleep(poll_interval_sec)
                continue

            if not self._should_save(frame):
                time.sleep(poll_interval_sec)
                continue

            try:
                self.save_frame(frame)
                saved_count += 1
            except Exception as e:
                logging.exception("Failed to save frame: %s", e)
                time.sleep(poll_interval_sec)

        logging.info("SegmentationDataCollector stopped: saved_count=%d", saved_count)
        return saved_count