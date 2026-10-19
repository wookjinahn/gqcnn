"""Convert one SuctionNet/GraspNet frame for examples/policy.py.

Depth PNG values are divided by the per-frame factor_depth in meta/*.mat to
produce depth in metres. All labelled object pixels form the segmentation mask.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from autolab_core import CameraIntrinsics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path,
                        help="directory containing scene_XXXX folders")
    parser.add_argument("--scene", type=int, required=True)
    parser.add_argument("--camera", choices=("realsense", "kinect"),
                        default="realsense")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    camera_dir = (args.dataset_root / "scene_{:04d}".format(args.scene)
                  / args.camera)
    stem = "{:04d}".format(args.frame)
    depth_path = camera_dir / "depth" / (stem + ".png")
    label_path = camera_dir / "label" / (stem + ".png")
    meta_path = camera_dir / "meta" / (stem + ".mat")
    for path in (depth_path, label_path, meta_path):
        if not path.is_file():
            parser.error("Missing input: {}".format(path))

    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None or labels is None or depth_raw.ndim != 2 or labels.ndim != 2:
        parser.error("Expected single-channel depth and label PNGs")
    if depth_raw.shape != labels.shape:
        parser.error("Depth and label image shapes differ")

    meta = loadmat(str(meta_path))
    factor = float(np.asarray(meta["factor_depth"]).squeeze())
    intrinsics_matrix = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    if not np.isfinite(factor) or factor <= 0 or intrinsics_matrix.shape != (3, 3):
        parser.error("Invalid depth factor or camera intrinsics")

    depth_m = depth_raw.astype(np.float32) / factor
    mask = ((labels > 0) & (depth_raw > 0)).astype(np.uint8) * 255
    if not mask.any():
        parser.error("No labelled pixels have valid depth")
    height, width = depth_m.shape
    intr = CameraIntrinsics(
        "{}_scene_{:04d}".format(args.camera, args.scene),
        fx=intrinsics_matrix[0, 0], fy=intrinsics_matrix[1, 1],
        cx=intrinsics_matrix[0, 2], cy=intrinsics_matrix[1, 2],
        skew=intrinsics_matrix[0, 1], width=width, height=height)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (output_dir / "depth.npy", output_dir / "segmask.png",
               output_dir / "camera.intr", output_dir / "source.json")
    if any(path.exists() for path in outputs):
        parser.error("Output already exists in {}".format(output_dir))
    np.save(str(outputs[0]), depth_m)
    if not cv2.imwrite(str(outputs[1]), mask):
        raise IOError("Could not save segmentation mask")
    intr.save(str(outputs[2]))
    outputs[3].write_text(json.dumps({
        "dataset_root": str(args.dataset_root.resolve()),
        "scene": args.scene,
        "camera": args.camera,
        "frame": args.frame,
        "factor_depth": factor,
        "valid_object_pixels": int(np.count_nonzero(mask)),
    }, indent=2) + "\n")
    print("Prepared {}: {} valid object pixels".format(
        output_dir, np.count_nonzero(mask)))


if __name__ == "__main__":
    main()
