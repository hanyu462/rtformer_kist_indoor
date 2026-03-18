# scripts/collect_indoor_kist_l8.py

from __future__ import annotations

import argparse
import logging

from providers.realsense_camera_provider import RealSenseCameraProvider
from data_collector.segmentation_data_collector import SegmentationDataCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect indoor semantic segmentation dataset with RealSense D435"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/delight/79e8d782-53d6-4398-adeb-cee29f4bff2a/data/indoor_kist_l8",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sample-period-sec", type=float, default=0.5)

    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--building", type=str, default="kist_l8")
    parser.add_argument("--floor", type=int, default=8)

    parser.add_argument(
        "--save-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save depth as .npy (default: True). Use --no-save-depth to disable.",
    )

    return parser.parse_args()


def build_common_metadata(args: argparse.Namespace) -> dict:
    common_metadata = {
        "scene": args.scene,
        "building": args.building,
        "floor": args.floor,
    }
    return {key: value for key, value in common_metadata.items() if value is not None}


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    common_metadata = build_common_metadata(args)

    camera = RealSenseCameraProvider(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    collector = None

    try:
        logging.info("Starting RealSense camera...")
        camera.start()

        collector = SegmentationDataCollector(
            output_dir=args.output_dir,
            camera_provider=camera,
            sample_period_sec=args.sample_period_sec,
            save_depth=args.save_depth,
            image_ext=".png",
            common_metadata=common_metadata,
        )

        logging.info("Start infinite collection. Press Ctrl+C to stop.")
        logging.info("output_dir=%s", args.output_dir)
        logging.info("common_metadata=%s", common_metadata)
        logging.info("save_depth=%s", args.save_depth)

        collector.collect(
            max_samples=None,
            max_duration_sec=None,
        )

    except KeyboardInterrupt:
        logging.info("Ctrl+C received. Stopping collection...")

    finally:
        if collector is not None:
            collector.request_stop()

        try:
            camera.stop()
        except Exception as e:
            logging.warning("Camera stop failed: %s", e)

        logging.info("Collection finished.")


if __name__ == "__main__":
    main()