"""Evaluate complete Dex-Net scenes as inference finishes, then aggregate AP.

Uses the existing benchmark's pilot evaluator, whose per-frame scoring body is
identical to the official full evaluator, with the axial-axis fix. Each scene
result is saved atomically and reused on restart.
"""

import argparse
import os
import runpy
import time
from pathlib import Path

import numpy as np

from suctionnetAPI.utils import eval_utils

from eval_suctionnet_with_axial_fix import (smoothness_with_axial_fix,
                                            viewpoint_with_axial_fix)


def all_frames_present(dump_root, camera, scene):
    folder = (dump_root / "test_seen" / "scene_{:04d}".format(scene)
              / camera / "suction")
    return all((folder / "{:04d}.npz".format(frame)).is_file()
               for frame in range(256))


def ap(res):
    return (float(np.mean(res[:, :, :1, :])),
            float(np.mean(res[:, :, :50, :])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--scenes", default="all",
                        help="all or a comma-separated scene list")
    parser.add_argument("--wait", action="store_true",
                        help="wait for each scene's 256 prediction files")
    args = parser.parse_args()
    scenes = (list(range(100, 130)) if args.scenes == "all"
              else [int(value) for value in args.scenes.split(",")])
    if not scenes or len(scenes) != len(set(scenes)) or any(
            scene < 100 or scene >= 130 for scene in scenes):
        parser.error("Scenes must be unique test_seen IDs in 100..129")

    eval_utils.get_smoothness_score = smoothness_with_axial_fix
    eval_utils.viewpoint_to_matrix = viewpoint_with_axial_fix
    module = runpy.run_path(str(args.benchmark_dir / "eval_pilot.py"))
    evaluator = module["PilotEval"](root=str(args.root), camera=args.camera)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pending = set(scenes)
    while pending:
        ready = [scene for scene in sorted(pending)
                 if (args.output_root / "scene_{:04d}_res.npy".format(scene)).is_file()
                 or all_frames_present(args.dump, args.camera, scene)]
        if not ready:
            if not args.wait:
                parser.error("Incomplete predictions for scenes {}".format(
                    ",".join(str(scene) for scene in sorted(pending))))
            time.sleep(10)
            continue
        for scene in ready:
            output_path = args.output_root / "scene_{:04d}_res.npy".format(scene)
            if not output_path.exists():
                result = evaluator.eval_scene_frames(
                    scene, "test_seen", str(args.dump), range(256))
                if result.shape != (256, 50, 4):
                    raise ValueError("Unexpected result shape for scene {}"
                                     .format(scene))
                temporary = output_path.with_name(output_path.name + ".tmp.npy")
                np.save(str(temporary), result)
                os.replace(str(temporary), str(output_path))
            result = np.load(str(output_path))
            print("scene {}: AP_top1 {:.4f}, AP_top50 {:.4f}".format(
                scene, float(result[:, :1, :].mean()),
                float(result[:, :50, :].mean())), flush=True)
            pending.remove(scene)

    if scenes != list(range(100, 130)):
        return
    result = np.stack([
        np.load(str(args.output_root / "scene_{:04d}_res.npy".format(scene)))
        for scene in scenes], axis=0)
    output_path = args.output_root / "dexnet3_candidates_res.npy"
    if output_path.exists():
        parser.error("Aggregate result already exists: {}".format(output_path))
    temporary = output_path.with_name(output_path.name + ".tmp.npy")
    np.save(str(temporary), result)
    os.replace(str(temporary), str(output_path))
    top1, top50 = ap(result)
    print("Dex-Net 3.0 test_seen: AP_top1 {:.4f}, AP_top50 {:.4f}".format(
        top1, top50), flush=True)
    print("AP_top1 per-threshold [0.2,0.4,0.6,0.8] = {}".format(
        [round(float(np.mean(result[:, :, :1, i])), 4) for i in range(4)]),
        flush=True)


if __name__ == "__main__":
    main()
