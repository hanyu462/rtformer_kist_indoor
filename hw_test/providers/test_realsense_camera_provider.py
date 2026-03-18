# test_realsense_camera_provider.py

"""
RealSenseCameraProvider — Hardware test script.

Tested APIs:
  start, stop, data (property)

Prerequisites:
  USB 연결을 통해 realsense camera를 연결

Usage: 
  python system_hw_test/providers/test_realsense_camera_provider.py

Controls (visualization window에서):
  q / ESC  — quit
"""

from __future__ import annotations

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import cv2
import numpy as np

from providers.realsense_camera_provider import RealSenseCameraProvider, CameraFrame


# ── 시각화 헬퍼 ───────────────────────────────────────────────────────────────

DEPTH_MAX_M = 5.0  # colormap 범위 상한 (미터)


def depth_to_colormap(depth_m: np.ndarray) -> np.ndarray:
    """float32 depth (meter) → BGR 컬러맵 이미지 (INFERNO)."""
    clipped = np.clip(depth_m, 0.0, DEPTH_MAX_M)
    normalized = (clipped / DEPTH_MAX_M * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def draw_overlay(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    frame: CameraFrame,
) -> np.ndarray:
    """컬러/depth 이미지에 정보 오버레이를 추가하고 가로로 이어붙인 이미지를 반환."""
    h, w = color_img.shape[:2]
    cx_px, cy_px = w // 2, h // 2

    center_dist = float(frame.depth[cy_px, cx_px])

    def put(img: np.ndarray, text: str, y: int, color=(0, 255, 0)) -> None:
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color,   1, cv2.LINE_AA)

    fps_text = f"FPS: {frame.camera_fps:.1f}" if frame.camera_fps > 0 else "FPS: N/A"

    color_disp = color_img.copy()
    put(color_disp, fps_text, 24)
    put(color_disp, f"fx={frame.intrinsics.fx:.1f}  fy={frame.intrinsics.fy:.1f}", 48)
    put(color_disp, f"cx={frame.intrinsics.cx:.1f}  cy={frame.intrinsics.cy:.1f}", 72)

    depth_disp = depth_img.copy()
    put(depth_disp, f"Center: {center_dist:.3f} m", 24, color=(255, 200, 0))
    put(depth_disp, f"Range: 0 ~ {DEPTH_MAX_M:.0f} m  (INFERNO)", 48, color=(200, 200, 200))
    cv2.drawMarker(depth_disp, (cx_px, cy_px), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)

    return np.hstack([color_disp, depth_disp])


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # -------------------------------------------------------------------------
    # Phase 0: Setup
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 0: Setup\n{'='*60}")
    RealSenseCameraProvider.reset()  # type: ignore[attr-defined]
    provider = RealSenseCameraProvider()
    print("  Provider created")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 1: Start
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 1: Start\n{'='*60}")
    try:
        provider.start()
    except RuntimeError as e:
        print(f"  FAIL: {e}")
        return 1
    print("  start()\n  OK")

    # -------------------------------------------------------------------------
    # Phase 2: Frame verification
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 2: Frame verification\n{'='*60}")
    frame = provider.data

    h, w = frame.bgr.shape[:2]
    print(f"  bgr  shape={frame.bgr.shape}  dtype={frame.bgr.dtype}")
    print(f"  depth shape={frame.depth.shape}  dtype={frame.depth.dtype}")
    print(f"  intrinsics: fx={frame.intrinsics.fx:.2f}  fy={frame.intrinsics.fy:.2f}  cx={frame.intrinsics.cx:.2f}  cy={frame.intrinsics.cy:.2f}")
    print(f"  center depth: {frame.depth[h//2, w//2]:.3f} m")
    print("  OK")

    # -------------------------------------------------------------------------
    # Phase 3: Live visualization
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 3: Live visualization\n{'='*60}")
    print("  Press 'q' or ESC in the window to quit.")

    window = "RealSenseCamera HW Test (q/ESC to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = provider.data
        if frame is None:
            time.sleep(0.01)
            continue

        depth_color = depth_to_colormap(frame.depth)
        combined = draw_overlay(frame.bgr, depth_color, frame)
        cv2.imshow(window, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break

    cv2.destroyAllWindows()
    print("  Visualization closed.")

    # -------------------------------------------------------------------------
    # Phase 4: Teardown
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}\n  Phase 4: Teardown\n{'='*60}")
    provider.stop()
    print("  stop()\n  OK")

    print("\n  All phases complete. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())