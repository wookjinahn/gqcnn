"""Run the existing Seg2Grasp SuctionNet evaluator with an axial-axis fix.

The vendored suctionnetAPI cannot form an XY basis for suction directions
exactly parallel to the camera optical axis. This wrapper leaves nonaxial
predictions untouched and supplies a deterministic basis for axial ones.
The same wrapper must score *all* methods in a comparison.

Example:
  conda run -n suctionnet_eval python tools/eval_suctionnet_with_axial_fix.py \
      --benchmark-dir ../Seg2Grasp/benchmarks/suctionnet_comparison \
      --mode pilot --root /path/to/graspnet --dump /path/to/preds \
      --scenes 100 --frames 0
"""

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np

from suctionnetAPI.utils import eval_utils


_original_smoothness = eval_utils.get_smoothness_score
_original_viewpoint = eval_utils.viewpoint_to_matrix
_rotate_x_90 = np.array([[1.0, 0.0, 0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0, 0.0]])


def is_axial(direction):
    direction = np.asarray(direction)
    return (np.linalg.norm(direction) > 1e-12
            and np.linalg.norm(direction[:2]) <= 1e-12)


def smoothness_with_axial_fix(suction_point, direction, model_points):
    if not is_axial(direction):
        return _original_smoothness(suction_point, direction, model_points)
    # Smoothness depends on relative geometry. Rotate point, direction, and
    # object together; the original scorer then receives a nonaxial direction.
    # No suction direction or object geometry is altered in their camera frame.
    return _original_smoothness(
        np.asarray(suction_point) @ _rotate_x_90.T,
        np.asarray(direction) @ _rotate_x_90.T,
        np.asarray(model_points) @ _rotate_x_90.T)


def viewpoint_with_axial_fix(direction):
    if not is_axial(direction):
        return _original_viewpoint(direction)
    axis_x = np.asarray(direction, dtype=np.float64)
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.array([0.0, 1.0, 0.0])
    axis_z = np.cross(axis_x, axis_y)
    return np.column_stack((axis_x, axis_y, axis_z))


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    args, evaluator_args = parser.parse_known_args()
    benchmark_dir = args.benchmark_dir.resolve()
    evaluator = benchmark_dir / ("eval_pilot.py" if args.mode == "pilot"
                                 else "full_eval.py")
    if not evaluator.is_file():
        parser.error("Missing benchmark evaluator: {}".format(evaluator))

    eval_utils.get_smoothness_score = smoothness_with_axial_fix
    eval_utils.viewpoint_to_matrix = viewpoint_with_axial_fix
    sys.argv = [str(evaluator)] + evaluator_args
    runpy.run_path(str(evaluator), run_name="__main__")


if __name__ == "__main__":
    main()
