import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a confidence-map .npy file")
    parser.add_argument("npy_path", type=Path)
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Original map width when the npy was saved with np.packbits",
    )
    parser.add_argument("--output", type=Path, help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Open an OpenCV window")
    return parser.parse_args()


def load_confidence_map(path: Path, width: int) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")

    # save_confidence_map.py stores binary maps using np.packbits(axis=-1).
    if data.shape[1] != width:
        data = np.unpackbits(data.astype(np.uint8), axis=-1)[:, :width]

    return (data > 0).astype(np.uint8) * 255


def main():
    args = parse_args()
    image = load_confidence_map(args.npy_path, args.width)
    output = args.output or args.npy_path.with_suffix(".png")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to save image: {output}")
    print(f"Saved: {output} ({image.shape[1]}x{image.shape[0]})")

    if args.show:
        cv2.imshow("confidence map", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
