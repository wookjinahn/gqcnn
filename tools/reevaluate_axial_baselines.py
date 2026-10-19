"""Re-score only baseline frames affected by the SuctionNet axial-axis fix.

The fixed evaluator delegates to the original API for every nonaxial suction.
Thus the saved original raw result is unchanged for a frame with no nonzero
axial direction. Re-evaluate every frame containing such a direction and
replace only those frame slices in a copy of the original raw result array.
"""

import argparse
import os
import runpy
from pathlib import Path

import numpy as np

from suctionnetAPI.utils import eval_utils

from eval_suctionnet_with_axial_fix import (smoothness_with_axial_fix,
                                            viewpoint_with_axial_fix)


METHODS = ("normal_std", "seg2grasp_committed", "seg2grasp_bestscore",
           "suctionnet_rgbd")


def affected_frames(dump_root, camera, scene):
    suction_dir = (dump_root / "test_seen" / "scene_{:04d}".format(scene)
                   / camera / "suction")
    frames = []
    axial_candidates = 0
    for frame in range(256):
        with np.load(str(suction_dir / "{:04d}.npz".format(frame))) as prediction:
            arr = prediction["arr_0"]
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError("Invalid prediction shape: {} frame {}"
                             .format(scene, frame))
        direction = arr[:, 1:4]
        axial = ((np.linalg.norm(direction, axis=1) > 1e-12)
                 & (np.linalg.norm(direction[:, :2], axis=1) <= 1e-12))
        count = int(np.count_nonzero(axial))
        if count:
            frames.append(frame)
            axial_candidates += count
    return frames, axial_candidates


def score(arr):
    return (float(np.mean(arr[:, :, :1, :])),
            float(np.mean(arr[:, :, :50, :])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preds-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    methods = args.methods.split(",")
    if not methods or any(method not in METHODS for method in methods):
        parser.error("Methods must be selected from {}".format(",".join(METHODS)))

    eval_utils.get_smoothness_score = smoothness_with_axial_fix
    eval_utils.viewpoint_to_matrix = viewpoint_with_axial_fix
    module = runpy.run_path(str(args.benchmark_dir / "eval_pilot.py"))
    evaluator = module["PilotEval"](root=str(args.root), camera=args.camera)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for method in methods:
        dump_root = args.preds_root / method
        original_path = args.preds_root / (method + "_res.npy")
        output_path = args.output_root / (method + "_axial_fixed_res.npy")
        if output_path.exists():
            parser.error("Output exists: {}".format(output_path))
        arr = np.load(str(original_path))
        if arr.shape != (30, 256, 50, 4):
            raise ValueError("Unexpected raw result shape: {}".format(arr.shape))
        old_score = score(arr)
        affected_count = 0
        candidate_count = 0
        for scene in range(100, 130):
            frames, candidates = affected_frames(dump_root, args.camera, scene)
            affected_count += len(frames)
            candidate_count += candidates
            if not frames:
                continue
            replacement = evaluator.eval_scene_frames(
                scene, "test_seen", str(dump_root), frames)
            if replacement.shape != (len(frames), 50, 4):
                raise ValueError("Unexpected re-evaluation shape")
            arr[scene - 100, frames] = replacement
            print("{} scene {}: {} affected frames".format(
                method, scene, len(frames)), flush=True)
        temporary = output_path.with_name(output_path.name + ".tmp.npy")
        np.save(str(temporary), arr)
        os.replace(str(temporary), str(output_path))
        new_score = score(arr)
        print("{}: {} axial candidates in {} frames; "
              "AP_top1 {:.4f} -> {:.4f}, AP_top50 {:.4f} -> {:.4f}"
              .format(method, candidate_count, affected_count,
                      old_score[0], new_score[0], old_score[1], new_score[1]),
              flush=True)


if __name__ == "__main__":
    main()
