"""Export Dex-Net 3.0 CEM candidates for the SuctionNet benchmark.

Run with the isolated dexnet3 Python environment. Output follows the official
SuctionNet dump layout and contains [Q, outward direction (3), point in metres
(3)] for every candidate in the policy's final CEM set. The first row is the
policy's selected action (highest Q with the unfiltered replication config).
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from autolab_core import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                          RgbdImage, YamlConfig)
from gqcnn.grasping import (CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            SuctionPoint2D)


def scene_split(scene):
    if 100 <= scene < 130:
        return "test_seen"
    if 130 <= scene < 160:
        return "test_similar"
    if 160 <= scene < 190:
        return "test_novel"
    raise ValueError("Scene must be in 100..189: {}".format(scene))


def parse_indices(value, all_range):
    if value == "all":
        return list(all_range)
    indices = [int(part.strip()) for part in value.split(",")]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Expected nonempty unique comma-separated indices")
    return indices


def frame_state(dataset_root, scene, camera, frame, inpaint_rescale_factor):
    scene_name = "scene_{:04d}".format(scene)
    base = dataset_root / "scenes" / scene_name / camera
    if not base.is_dir():
        base = dataset_root / scene_name / camera
    stem = "{:04d}".format(frame)
    depth_path = base / "depth" / (stem + ".png")
    label_path = base / "label" / (stem + ".png")
    meta_path = base / "meta" / (stem + ".mat")
    for path in (depth_path, label_path, meta_path):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if (raw_depth is None or labels is None or raw_depth.ndim != 2
            or labels.ndim != 2 or raw_depth.shape != labels.shape):
        raise ValueError("Invalid depth/label PNGs for {}".format(stem))
    meta = loadmat(str(meta_path))
    factor = float(np.asarray(meta["factor_depth"]).squeeze())
    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=np.float64)
    if not np.isfinite(factor) or factor <= 0 or intrinsic.shape != (3, 3):
        raise ValueError("Invalid depth factor/intrinsics for {}".format(stem))
    height, width = raw_depth.shape
    camera_intr = CameraIntrinsics(
        "{}_{}".format(camera, scene_name),
        fx=intrinsic[0, 0], fy=intrinsic[1, 1],
        cx=intrinsic[0, 2], cy=intrinsic[1, 2],
        skew=intrinsic[0, 1], width=width, height=height)
    depth_data = raw_depth.astype(np.float32) / factor
    mask = ((labels > 0) & (raw_depth > 0)).astype(np.uint8) * 255
    if not np.any(mask):
        raise ValueError("No valid object pixels for {}".format(stem))

    depth_im = DepthImage(depth_data, frame=camera_intr.frame)
    color_im = ColorImage(np.zeros((height, width, 3), dtype=np.uint8),
                          frame=camera_intr.frame)
    segmask = BinaryImage(mask, frame=camera_intr.frame)
    segmask = segmask.mask_binary(depth_im.invalid_pixel_mask().inverse())
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    return RgbdImageState(rgbd_im, camera_intr, segmask=segmask)


def candidates_to_array(grasps, q_values):
    if len(grasps) != len(q_values):
        raise ValueError("Grasp/Q count mismatch")
    order = np.argsort(-np.asarray(q_values), kind="stable")
    rows = np.empty((len(order), 7), dtype=np.float32)
    for row_index, candidate_index in enumerate(order):
        grasp = grasps[candidate_index]
        if not isinstance(grasp, SuctionPoint2D):
            raise TypeError("Expected SuctionPoint2D")
        rows[row_index, 0] = q_values[candidate_index]
        rows[row_index, 1:4] = -grasp.axis
        rows[row_index, 4:7] = grasp.pose().translation
    if not np.all(np.isfinite(rows)):
        raise ValueError("Nonfinite suction prediction")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=Path(".ai/local/models/GQCNN-3.0"))
    parser.add_argument("--config", type=Path,
                        default=Path("cfg/examples/replication/dex-net_3.0.yaml"))
    parser.add_argument("--camera", choices=("realsense", "kinect"),
                        default="realsense")
    parser.add_argument("--scenes", default="100,101")
    parser.add_argument("--frames", default="all")
    args = parser.parse_args()

    scenes = parse_indices(args.scenes, range(100, 130))
    frames = parse_indices(args.frames, range(256))
    for scene in scenes:
        scene_split(scene)
    if any(frame < 0 or frame >= 256 for frame in frames):
        parser.error("Frames must be in 0..255")
    model_dir = args.model_dir.resolve()
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("gqcnn_config", {}).get("input_data_mode") not in (
            "suction", "tf_image_suction"):
        parser.error("Model is not a suction GQ-CNN")
    config = YamlConfig(str(args.config))
    policy_config = config["policy"]
    policy_config["metric"]["gqcnn_model"] = str(model_dir)
    policy = CrossEntropyRobustGraspingPolicy(policy_config)

    for scene in scenes:
        split = scene_split(scene)
        output_dir = (args.save_root / split / "scene_{:04d}".format(scene)
                      / args.camera / "suction")
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            output_path = output_dir / "{:04d}.npz".format(frame)
            if output_path.exists():
                print("skip scene {} frame {:04d}: exists".format(scene, frame),
                      flush=True)
                continue
            started = time.time()
            state = frame_state(args.dataset_root, scene, args.camera, frame,
                                config["inpaint_rescale_factor"])
            grasps, q_values = policy.action_set(state)
            rows = candidates_to_array(grasps, q_values)
            if len(rows) == 0:
                raise ValueError("Policy produced no candidates")
            temporary_path = output_path.with_name(output_path.name + ".tmp.npz")
            np.savez_compressed(str(temporary_path), rows)
            os.replace(str(temporary_path), str(output_path))
            print("scene {} frame {:04d}: {} candidates; top Q {:.4f}; {:.2f}s"
                  .format(scene, frame, len(rows), rows[0, 0],
                          time.time() - started), flush=True)


if __name__ == "__main__":
    main()
