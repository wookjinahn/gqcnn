# Dex-Net 3.0 in the Seg2Grasp/SuctionNet comparison

This extends the existing `Seg2Grasp/benchmarks/suctionnet_comparison/`
benchmark with the original GQ-CNN 3.0 suction policy. Use the same GraspNet/SuctionNet
`test_seen` scenes 0100–0129, RealSense camera, and official suctionnetAPI AP
metric. The Dex-Net checkpoint was trained with a Primesense camera, so this
comparison measures transfer to RealSense images; it is not an in-domain test.

Run from the gqcnn repository root. First complete [Dex-Net setup](dexnet3_suctionnet.md).
The other benchmark's `suctionnet_eval` Conda environment and dataset root
(`scenes/`, `models/`, `dense_point_clouds/`) are also required. Set paths for
your machine:

```bash
export GRASPNET_ROOT=/path/to/graspnet_dataset
export S2G_BENCH=/path/to/Seg2Grasp/benchmarks/suctionnet_comparison
export BASELINE_PREDS=/path/to/existing/preds_full
```

## Export Dex-Net predictions

The exporter loads GQ-CNN once, then applies the replication CEM policy to
each frame. It reads the per-frame depth factor, camera intrinsics, and object
labels. The union of nonzero ground-truth object labels is the Dex-Net mask,
as in the earlier `normal_std` baseline; Seg2Grasp runs its own segmenter.
This input difference matters when interpreting the planner comparison.
The policy's selected grasp is rank 1; its final scored candidate set
provides the other ranks. Files are written in the same SuctionNet layout as
the existing benchmark, with rows `[Q, outward_direction_xyz,
camera_point_xyz_m]`. Existing files are skipped so interrupted runs resume.
Outputs, models, and datasets remain in ignored `.ai/local/` or outside Git.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.ai/local/mpl" PYTHONPATH="$PWD" \
  .ai/local/dexnet3/bin/python tools/dexnet3_suctionnet_infer.py \
  --dataset-root "$GRASPNET_ROOT" \
  --save-root .ai/local/benchmark_preds/dexnet3_candidates \
  --scenes "$(seq -s, 100 129)" --frames all
```

For a pilot, use `--scenes 100 --frames 0,1,2,3` and a separate `--save-root`.
The full prediction directory must contain 7,680 `.npz` files before the
aggregate result is complete. A frame without a valid grasp stops the exporter with an error;
it does not silently insert a synthetic suction point.

## Evaluate on the same metric

## Completed test_seen result

The full run completed on 30 RealSense scenes (0100–0129), 256 frames per
scene, using the same patched evaluator for every method:

| Method | AP_top1 | AP_top50 | AP_top1 per-thr [0.2/0.4/0.6/0.8] |
|---|---:|---:|---|
| SuctionNet `normal_std` | 0.1540 | 0.1010 | 0.315 / 0.188 / 0.095 / 0.018 |
| SuctionNet DeepLabV3+ RGB-D | 0.4625 | 0.2826 | 0.707 / 0.629 / 0.422 / 0.092 |
| Seg2Grasp committed | 0.2055 | 0.1204 | 0.342 / 0.269 / 0.169 / 0.042 |
| Seg2Grasp bestscore | 0.1503 | 0.1188 | 0.251 / 0.203 / 0.119 / 0.029 |
| **Dex-Net 3.0 GQ-CNN** | **0.0736** | **0.0271** | **0.175 / 0.087 / 0.029 / 0.002** |

Dex-Net ranks below the other four methods on this transfer setting. Its
checkpoint was trained for Primesense imagery, while the benchmark uses
RealSense frames. It also receives the union of ground-truth object labels as
its mask, like `normal_std`; Seg2Grasp uses its learned segmentation output.
The score is therefore a benchmark result for this exact protocol, not a
claim about in-domain Dex-Net performance. Raw Dex-Net results are in
`.ai/local/benchmark_preds/dexnet3_eval/dexnet3_candidates_res.npy` and are
ignored by Git.

The previous benchmark's API computes an undefined XY basis when a suction
direction is exactly parallel to the camera optical axis. Dex-Net produces
such directions. `tools/eval_suctionnet_with_axial_fix.py` runs the existing
pilot or full evaluator with a deterministic basis for these axial directions.
It leaves all other directions and the benchmark's 2 cm NMS unchanged. Apply
the wrapper to **every** method in the comparison. Zero-length directions from
other methods retain the API's original behavior.

For a full run, the incremental evaluator can start while inference is still
running. It waits for each scene's 256 files, saves a raw result per scene, and
then saves the 30-scene aggregate. Restarting it reuses completed scene results.
For parallel evaluation, run disjoint `--scenes` lists into the same output
root, then rerun once with `--scenes all` after every scene result exists to
write the aggregate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n suctionnet_eval python \
  tools/eval_dexnet3_suctionnet_incremental.py \
  --benchmark-dir "$S2G_BENCH" --root "$GRASPNET_ROOT" \
  --dump "$PWD/.ai/local/benchmark_preds/dexnet3_candidates" \
  --output-root "$PWD/.ai/local/benchmark_preds/dexnet3_eval" \
  --camera realsense --wait
```

The existing pilot and full evaluators can also be run directly through the
axial-fix wrapper:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n suctionnet_eval python tools/eval_suctionnet_with_axial_fix.py \
  --benchmark-dir "$S2G_BENCH" --mode pilot \
  --root "$GRASPNET_ROOT" \
  --dump "$PWD/.ai/local/benchmark_preds/dexnet3_candidates" \
  --scenes 100 --frames 0,1,2,3 --camera realsense

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n suctionnet_eval python tools/eval_suctionnet_with_axial_fix.py \
  --benchmark-dir "$S2G_BENCH" --mode full \
  --root "$GRASPNET_ROOT" \
  --dump "$PWD/.ai/local/benchmark_preds/dexnet3_candidates" \
  --camera realsense --proc 4 \
  --res-out "$PWD/.ai/local/benchmark_preds/dexnet3_candidates_res.npy"
```

For the four existing methods, `tools/reevaluate_axial_baselines.py` scans all
prediction files and re-scores only frames containing a nonzero axial
direction. The wrapper delegates every other direction to the original API,
so raw scores on unaffected frames remain valid. It copies the historical raw
result arrays and replaces each affected frame with the fixed pilot evaluator's
50×4 result. Original prediction files and result arrays are preserved.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run -n suctionnet_eval python tools/reevaluate_axial_baselines.py \
  --benchmark-dir "$S2G_BENCH" --root "$GRASPNET_ROOT" \
  --preds-root "$BASELINE_PREDS" \
  --output-root "$PWD/.ai/local/benchmark_preds" --camera realsense
```

The reference methods are `normal_std`, `seg2grasp_committed`,
`seg2grasp_bestscore`, and `suctionnet_rgbd`. Running the full wrapper on each
prediction directory is an equivalent, more expensive alternative.

`AP_top1` measures the grasp the policy would execute. `AP_top50` requires the
exported candidate set; the older single-grasp `suction.npz` output is not a
valid top-50 comparison. The benchmark scorer returns mean AP across four
seal × wrench thresholds (0.2, 0.4, 0.6, 0.8), with collision checking.
