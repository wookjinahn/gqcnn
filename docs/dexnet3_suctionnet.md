# Dex-Net 3.0 on example images and SuctionNet frames

Run from the repository root on Ubuntu 22.04. The setup creates a Python 3.7
Conda environment and downloads the original GQ-CNN 3.0 suction model into
ignored `.ai/local/`. It uses the model linked from the [official Dex-Net 3.0
page](https://berkeleyautomation.github.io/dex-net/); the Box URL in this
checkout's `download_models.sh` currently returns 404.

This path was verified on Ubuntu 22.04.5 with Conda 26.3.2. It keeps the
legacy environment, package caches, model, converted inputs, and predictions
under `.ai/local/`; the existing Conda base and other environments are not
modified. Setup requires `conda`, `unzip`, and network access to the Conda,
Python package, and Google Drive sources. The model ZIP is about 79 MB and is
checked against SHA256
`d048afb9974d4c3170ccb89289302ea871b795d421ee3910805675f35da7b1f6`.

```bash
bash .dev/setup.sh
conda activate "$PWD/.ai/local/dexnet3"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.ai/local/mpl"
```

The installed versions are in `envs/dexnet3.yml`: Python 3.7.16, pip 22.3.1,
TensorFlow 1.15.0, NumPy 1.19.5, autolab-core 1.1.1,
autolab-perception 1.0.0, and visualization 1.0.0. `pyglet` is pinned to
1.5.31 because 2.0.10 refused Python 3.7 at import time; `ruamel.yaml` is
pinned to 0.17.40 because 0.18 removed the API used by autolab-core. A CPU is
sufficient for these examples.

## Repository example image

```bash
python examples/policy.py GQCNN-3.0 \
  --model_dir .ai/local/models \
  --depth_image data/examples/single_object/primesense/depth_0.npy \
  --segmask data/examples/single_object/primesense/segmask_0.png \
  --config_filename cfg/examples/replication/dex-net_3.0.yaml \
  --result_dir .ai/local/results/example_0
```

The result folder contains `grasp.json`, `grasp.png`, and `suction.npz`.
Choose a new result folder when rerunning; existing result files are preserved.
For other supplied images, replace the `0` in both input names with an index
from 1 through 9.

Verified result for image 0: Q=0.5994, center `(365.77, 224.19)` pixels,
and depth 0.707 m. This confirms that the pre-trained model loaded and the
policy completed planning; it is not a measured grasp success rate.

## One SuctionNet frame

Point `dataset_root` to a directory containing `scene_XXXX` folders. This
example uses the local `test_seen` scene 0100, RealSense frame 0000:

```bash
python tools/prepare_suctionnet_frame.py \
  /home/wj/Desktop/Codes/graspnet_dataset \
  --scene 100 --camera realsense --frame 0 \
  --output_dir .ai/local/suctionnet/scene_0100/realsense/0000

python examples/policy.py GQCNN-3.0 \
  --model_dir .ai/local/models \
  --depth_image .ai/local/suctionnet/scene_0100/realsense/0000/depth.npy \
  --segmask .ai/local/suctionnet/scene_0100/realsense/0000/segmask.png \
  --camera_intr .ai/local/suctionnet/scene_0100/realsense/0000/camera.intr \
  --config_filename cfg/examples/replication/dex-net_3.0.yaml \
  --result_dir .ai/local/results/suctionnet_scene_0100_realsense_0000
```

The converter divides raw PNG depth by the frame's `factor_depth` in
`meta/0000.mat`, takes all nonzero object labels with valid depth as the mask,
and writes camera intrinsics from the same metadata. Each output folder also
contains `source.json` with the frame identity and conversion factor.
Use new output and result directories when rerunning a frame; existing files
are never overwritten by these commands.

The `suction.npz` array has one row in the SuctionNet evaluation convention:
`[score, outward_direction_x, outward_direction_y, outward_direction_z,
point_x_m, point_y_m, point_z_m]`. The GQ-CNN approach axis is negated for the
outward direction. This is one top-ranked prediction for inspection, not a
complete SuctionNet evaluation dump.

Verified result for scene 0100/frame 0000: 203,495 valid object pixels after
conversion; Q=0.2429, center `(418.39, 218.36)` pixels, depth 0.430 m. The
selected pixel is within the segmentation mask. The saved NPZ has shape
`(1, 7)` and outward direction `(0, 0, -1)` in camera coordinates.

GQ-CNN 3.0 was trained for a Primesense Carmine camera and its original
synthetic scenes. SuctionNet RealSense frames have different optics and scene
statistics, so the displayed Q value is a model output, not measured grasp
success or benchmark accuracy. The official SuctionNet evaluation requires
predictions for every frame in a test split; see the
[SuctionNet API](https://github.com/graspnet/suctionnetAPI).
