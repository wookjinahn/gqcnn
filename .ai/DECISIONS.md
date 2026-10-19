# Architecture and design decisions

Append dated decisions with context, alternatives, rationale and consequences.
Do not overwrite earlier decisions; mark superseded decisions explicitly.

## 2026-10-19 — Isolated Dex-Net 3.0 inference environment and SuctionNet adapter

Context: The checked-out GQ-CNN code targets Python 3.5–3.7 and TensorFlow 1.15, while the Ubuntu 22.04 host has Python 3.13. The user requested Conda and plans to test the pre-trained suction policy on SuctionNet images.

Alternatives: Install legacy dependencies in Conda base; use the repository's Ubuntu 16.04 Docker image; retrain or port the model to a modern framework.

Decision and rationale: Keep Python 3.7 dependencies and downloaded model in ignored `.ai/local/`, define the direct package versions in `envs/dexnet3.yml`, and convert individual SuctionNet frames to the existing policy's depth, mask, and camera-intrinsics inputs. This preserves the user's other environments and allows the repository's original inference path to run on CPU.

Consequences: The legacy environment must be recreated for another machine. Predictions on RealSense SuctionNet images are out of the model's Primesense training domain; a single-image Q value is not a benchmark accuracy result. SuctionNet's evaluation row uses the outward direction, which is opposite the GQ-CNN approach axis.

## 2026-10-19 — Compare Dex-Net 3.0 with the existing SuctionNet AP benchmark

Context: The previous Seg2Grasp/SuctionNet comparison uses the official suctionnetAPI on RealSense test_seen scenes 0100–0129. The current Dex-Net example emits one suction per frame, and the API's smoothness and wrench basis calculations divide by zero for a nonzero direction exactly along the camera optical axis. Dex-Net's policy produces such directions.

Alternatives: Compare only the one-grasp example output; perturb axial directions in the prediction files; copy and modify the whole suctionnetAPI package; change the benchmark metric.

Decision and rationale: Export the original GQ-CNN 3.0 CEM policy's final scored candidate set after loading the model once. Rank it by the same Q score used to select the executed grasp, and convert each approach axis to the outward direction required by the API. Run the existing benchmark evaluator through a small wrapper that chooses a deterministic orthonormal basis for nonzero axial directions while leaving every other direction and the existing NMS behavior intact. Re-evaluate all methods with this wrapper before comparing AP. Keep the existing Seg2Grasp worktree and historical results untouched.

Consequences: AP_top1 describes the selected policy action and AP_top50 uses the scored candidate set. The axial basis convention must be recorded with the result because the API's ring-bin smoothness calculation depends slightly on in-plane basis choice. This remains a RealSense transfer comparison for a Primesense-trained model; the learned SuctionNet baseline is evaluated on its seen objects. Dex-Net uses the union of ground-truth object labels as its mask (like the earlier normal_std baseline), whereas Seg2Grasp uses its own segmenter; the methods therefore differ in mask input.

## 2026-10-19 — Full Dex-Net 3.0 test_seen result

Ran the original GQ-CNN 3.0 replication CEM policy on all 30 RealSense
`test_seen` scenes (0100–0129, 256 frames each), exporting the final scored
candidate set for every frame. Applied the same axial-direction-safe evaluator
to Dex-Net and selectively re-scored affected frames in each existing method.

| Method | AP_top1 | AP_top50 |
|---|---:|---:|
| SuctionNet normal_std | 0.1540 | 0.1010 |
| SuctionNet DeepLabV3+ RGB-D | 0.4625 | 0.2826 |
| Seg2Grasp committed | 0.2055 | 0.1204 |
| Seg2Grasp bestscore | 0.1503 | 0.1188 |
| Dex-Net 3.0 GQ-CNN | 0.0736 | 0.0271 |

The result is a RealSense transfer test for a Primesense-trained checkpoint,
not an in-domain Dex-Net score. Dex-Net and normal_std use the union of
ground-truth object labels as their input mask; Seg2Grasp uses its own
segmentation output. `test_similar` and `test_novel` were not evaluated.
