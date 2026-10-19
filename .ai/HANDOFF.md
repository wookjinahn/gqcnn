# Work handoff

## Current task
Complete the Dex-Net 3.0 comparison with the existing Seg2Grasp/SuctionNet benchmark. The earlier Dex-Net setup and documentation were committed and pushed.

## Goal
Use the same SuctionNet test_seen scenes, camera, prediction format, and official evaluator to compare Dex-Net 3.0 with the prior methods; keep models, datasets, environments, and outputs out of Git.

## Completed work
Completed the full Dex-Net 3.0 comparison. Generated exactly 7,680 prediction files (30 scenes × 256 frames) and 30 per-scene evaluator arrays, then aggregated them into `.ai/local/benchmark_preds/dexnet3_eval/dexnet3_candidates_res.npy`. Final patched-evaluator scores: Dex-Net AP_top1 0.0736/AP_top50 0.0271, per-threshold 0.1754/0.0871/0.0293/0.0025. The four existing methods' selectively fixed scores remain normal_std 0.1540/0.1010, learned SuctionNet 0.4625/0.2826, Seg2Grasp committed 0.2055/0.1204, and bestscore 0.1503/0.1188. Updated the benchmark guide and appended the final result to `DECISIONS.md`.

Selective baseline re-evaluation completed successfully for all four methods. It re-scored 472 `normal_std` frames (725 axial candidates), 547 Seg2Grasp committed frames (701), 545 bestscore frames (699), and 56 learned SuctionNet frames (85). The original and fixed aggregate AP values match to four decimals for every method; `normal_std` has 355 changed raw entries and a small AP_top1 increase of 0.00003255, while the other three raw arrays are bit-identical. Added `tools/eval_dexnet3_suctionnet_incremental.py` to score a scene as soon as its 256 prediction files exist and save each scene result atomically; the first scene matched the earlier pilot (0.0400/0.0168). Repartitioned inference into 15 disjoint two-scene workers with thread limits after confirming host memory and load headroom.

Added `tools/reevaluate_axial_baselines.py` to scan the four existing prediction dumps, re-score only frames with a nonzero axial direction, and preserve unaffected slices of the original raw result arrays. The full baseline evaluators were interrupted after CPU thread oversubscription was observed (host load average exceeded 300); no result arrays were created by those interrupted runs. Batch inference was likewise interrupted safely after atomic outputs, then resumed with BLAS/OMP/TensorFlow thread limits. Scene 0100 is complete and its fixed-evaluator AP_top1/AP_top50 is 0.0400/0.0168. Documented the selective re-evaluation in the benchmark guide and appended the benchmark design decision to `.ai/DECISIONS.md`.

Added `tools/dexnet3_suctionnet_infer.py` to load the trained GQ-CNN once and export each frame's final CEM candidates in the official benchmark format, and `tools/eval_suctionnet_with_axial_fix.py` to run the existing Seg2Grasp evaluators with a deterministic optical-axis basis for nonzero axial directions. Added `docs/dexnet3_suctionnet_benchmark.md` and linked it from the previous usage guide. The batch export for frames 0000–0001 passed; rank 1 of frame 0000 matches the previous single-frame `suction.npz` in score, position, and direction. Pilot evaluation of 16 frames with the fixed evaluator passed without the axial divide warnings for Dex-Net (AP_top1 0.0000, AP_top50 0.0098; preliminary only). The fixed evaluator also scored all four prior methods for a two-frame pilot; `normal_std` matched its original pilot output exactly (AP_top1 0.1250, AP_top50 0.0794). Outputs and logs are under ignored `.ai/local/`.

Checked the benchmark's prediction layout and pilot evaluator against Dex-Net's existing `suction.npz`. Copied the previously verified scene 0100/frame 0000 prediction into ignored `.ai/local/benchmark_preds/dexnet3/test_seen/scene_0100/realsense/suction/0000.npz` and ran the official pilot evaluator on that one frame. The evaluator consumed the file, but emitted divide-by-zero warnings for Dex-Net's exact optical-axis direction; its zero score is not yet interpretable. Inspected the GQ-CNN policy and confirmed `CrossEntropyRobustGraspingPolicy.action_set()` returns the final ranked candidate pool and Q values, which can provide AP_top50 predictions.

Located the previous Seg2Grasp/SuctionNet comparison in `/home/wj/Desktop/Codes/Seg2Grasp/benchmarks/suctionnet_comparison/` and the raw evaluation outputs in `/home/wj/Desktop/Codes/s2g_suctionnet_bench/preds_full/`. Read the benchmark README, full evaluator, decisions, and the saved evaluation summary. No benchmark files were changed; the Seg2Grasp repository has pre-existing edits that remain untouched.

Pushed commits `56651dc` and `e9c75c9` to `origin/master`; verified HEAD and the tracking branch match at `e9c75c9`, divergence is 0/0, and working/staged diffs are empty before this final checkpoint.

Read project instructions and previous handoff; inspected git status, both diffs, and recent log. Created the ignored `.ai/local/dexnet3` Conda environment with Python 3.7.16 and compatible packages. Downloaded the GQ-CNN 3.0 model from the official Dex-Net page's Google Drive link, verified ZIP integrity and SHA256, and extracted it to `.ai/local/models/GQCNN-3.0`. Ran the repository Primesense example and SuctionNet test_seen scene 0100 RealSense frame 0000. Added a frame converter, policy result JSON/PNG/NPZ output, an environment file, reproducible setup script, and usage documentation. Updated AGENTS.md with measured versions and appended a decision in DECISIONS.md. Expanded the usage guide with installation prerequisites, verified dependency versions, commands, and measured results; linked it from README.md. Reviewed explicit staged paths and diff, verified ignored local assets, fetched origin, and committed the code/docs as `56651dc`.

## Current state
The comparison is complete. Working tree changes are the new exporter/evaluator tools and documentation only; all large predictions, models, and raw arrays remain ignored under `.ai/local/`. The final ranking on this protocol is learned SuctionNet > Seg2Grasp committed > normal_std ≈ Seg2Grasp bestscore > Dex-Net 3.0.

Existing Seg2Grasp files remain untouched. The earlier single-grasp pilot's zero result remains invalid; the new candidate-pool pilot uses the fixed evaluator.

The new exporter writes scored candidate sets in the same `[score, direction(3), point_m(3)]` dump and directory layout. The older `examples/policy.py` still emits only one grasp per frame. A full metric result is not yet available.

Both initial inference paths work on CPU. Example frame 0 produced Q=0.5994 at pixel (365.77, 224.19), depth 0.707 m. SuctionNet scene 0100 RealSense frame 0000 produced Q=0.2429 at pixel (418.39, 218.36), depth 0.430 m; the selected pixel lies inside the object mask. Verified output is in `.ai/local/results/suctionnet_scene_0100_realsense_0000_verified/`. The user was asked whether to stop at one-frame inference or proceed to a full test-split evaluation; no answer yet. Code and docs are committed as `56651dc`; commits 56651dc and e9c75c9 were pushed to origin/master; matching tracking refs and clean diffs were verified before this final checkpoint. Existing worktree changes were preserved.

## Important findings
The SuctionNet API used by the previous benchmark divides by the XY norm of the suction direction in `get_smoothness_score()` and `viewpoint_to_matrix()` without handling a direction exactly parallel to the optical axis. Dex-Net's scene 0100/frame 0000 output has direction `(0,0,-1)`, causing both divide-by-zero warnings and a pilot AP_top1 of 0.0000. Repair the evaluator's basis construction and re-evaluate every compared method with the same patched evaluator before comparing scores. Do not perturb Dex-Net's physical direction just to bypass the issue. AP_top50 from a single-row dump is not comparable to the previous multi-candidate methods.

The earlier comparison covers GraspNet/SuctionNet `test_seen`, RealSense, scenes 0100–0129, 256 frames per scene. AP_top1/AP_top50: normal_std 0.1540/0.1010; learned SuctionNet DeepLabV3+ RGB-D 0.4625/0.2826; Seg2Grasp committed policy 0.2055/0.1204; Seg2Grasp bestscore 0.1503/0.1188. See the Seg2Grasp benchmark README and `.ai/DECISIONS.md`; the scratch `eval_results.txt` contains the three training-free runs, while the learned-model result is recorded in the README and decisions. `test_novel` was not run. These results are separate from the current Dex-Net 3.0 single-frame smoke test.

At startup master was aligned with origin/master at 499a609; after the first push both committed refs matched at e9c75c9 with 0/0 divergence. Host Ubuntu 22.04.5 and Conda 26.3.2; Python 3.13 is incompatible with this legacy code. The working environment has TensorFlow 1.15.0, NumPy 1.19.5, autolab-core 1.1.1, autolab-perception 1.0.0, visualization 1.0.0, pyglet 1.5.31, and ruamel.yaml 0.17.40; see `envs/dexnet3.yml`. GQ-CNN 3.0 was trained for Primesense Carmine, while the tested SuctionNet frame uses RealSense; Q is not benchmark accuracy. Local SuctionNet data includes test_seen scenes 0100–0129 under `/home/wj/Desktop/Codes/graspnet_dataset`. The dataset's depth factor is 1000 for the tested frame; metadata supplies intrinsics. SuctionNet's evaluation array uses `[score, outward_direction(3), camera_point_m(3)]`; the GQ-CNN approach axis is negated. Model ZIP SHA256: `d048afb9974d4c3170ccb89289302ea871b795d421ee3910805675f35da7b1f6`.

## Modified files
Commit `b7ce385` adds `tools/dexnet3_suctionnet_infer.py`, `tools/eval_suctionnet_with_axial_fix.py`, `tools/reevaluate_axial_baselines.py`, `tools/eval_dexnet3_suctionnet_incremental.py`, and `docs/dexnet3_suctionnet_benchmark.md`; it edits `docs/dexnet3_suctionnet.md`, `.ai/DECISIONS.md`, and this handoff. Large outputs remain ignored under `.ai/local/`.

Commit `56651dc` contains `.gitignore`, AGENTS.md, CLAUDE.md, README.md, `examples/policy.py`, `.dev/setup.sh`, `envs/dexnet3.yml`, `tools/prepare_suctionnet_frame.py`, `docs/dexnet3_suctionnet.md`, and `.ai/DECISIONS.md`. The first handoff commit e9c75c9 was pushed; this final publication checkpoint updates the same file. Ignored `.ai/local/` assets were not staged.

## Failed approaches
Running four inference processes and four four-worker baseline evaluations concurrently without per-process thread limits oversubscribed the host (load average >300) and slowed inference. Stopped only these new processes with Ctrl-C; atomic existing prediction files were preserved. Restarted inference with BLAS/OMP/TensorFlow thread limits. Replacing the full baseline re-evaluation with selective affected-frame re-evaluation because the fix is identical to the original API for every nonaxial direction.

Host Python invocation failed with `ModuleNotFoundError: numpy`. Initial sandboxed Conda and pip network calls failed DNS resolution; escalation allowed downloads. First Conda solve with `pip=23.0` conflicted with Python 3.7; removing the pin selected pip 22.3.1. Repository Box model URL returned HTTP 404; the official Dex-Net page linked an accessible Google Drive ZIP. Initial dependency import failed because pyglet 2.0.10 requires Python 3.8; pinned 1.5.31. First policy run failed because ruamel.yaml 0.18 removed `yaml.load`; pinned 0.17.40. Docker daemon was previously inaccessible in the sandbox and was not needed. The first `git add` failed because `.git/index.lock` could not be created in the read-only sandbox; the approved escalated retry succeeded.

## Test/build results
Final verification: `find .ai/local/benchmark_preds/dexnet3_candidates/test_seen -name '*.npz'` counted 7,680 files; 30 scene arrays and the aggregate `(30,256,50,4)` array exist. Aggregate scores were recomputed from the saved arrays. New scripts compile under their target environments and `git diff --check` passes. No errors were found in the inference/evaluation logs.

The selective baseline re-evaluation exited 0 and saved four `(30,256,50,4)` arrays; aggregate AP_top1/AP_top50 were normal_std 0.154004/0.100972, Seg2Grasp committed 0.205469/0.120446, bestscore 0.150326/0.118825, and learned SuctionNet 0.4625/0.2826 to four decimals. The scene 0100 incremental evaluator result matched the previous 256-frame pilot. The optional normal_std 256-frame pilot was interrupted to prioritize the incremental Dex-Net evaluation; its earlier 2-frame original/fixed equivalence check passed.

Fixed-evaluator full-scene pilot on Dex-Net scene 0100 (256 frames) passed without axial warnings: AP_top1 0.0400, AP_top50 0.0168, per-threshold top1 0.1250/0.0312/0.0039/0.0000. The selective baseline script passed Python compilation. This is not the final 30-scene comparison.

Both initial scripts passed Python compilation in their respective Conda environments. Batch inference on scene 0100 frames 0000–0001 produced scored candidate arrays (frame 0000 shape 50×7) and rank 1 reproduced the earlier single-frame result. Fixed evaluator pilot on scene 0100 frames 0000–0001 produced Dex-Net AP_top1 0.0000/AP_top50 0.0274 without axial warnings, and a 16-frame pilot produced 0.0000/0.0098. A two-frame `normal_std` pilot yielded 0.1250/0.0794 under both the original and fixed evaluator. Fixed 16-frame pilot scores were 0.0781/0.0666 for normal_std, 0.0000/0.0697 for Seg2Grasp committed, and 0.4688/0.2512 for learned SuctionNet; these are only pilot diagnostics. `normal_std` emitted its existing zero-direction warnings on this longer pilot; the axial fix intentionally does not change zero-length directions. `git diff --check` passed.

Ran `conda run -n suctionnet_eval python /home/wj/Desktop/Codes/Seg2Grasp/benchmarks/suctionnet_comparison/eval_pilot.py --root /home/wj/Desktop/Codes/graspnet_dataset --dump /home/wj/Desktop/Codes/gqcnn/.ai/local/benchmark_preds/dexnet3 --scenes 100 --frames 0 --camera realsense`: exited 0 and printed AP_top1=0.0000, but emitted NumPy invalid-divide warnings from `eval_utils.py:147` and `rotation.py:72`; this earlier run is an invalid diagnostic, superseded by the fixed evaluator pilot.

Read-only verification for the comparison lookup: inspected the Seg2Grasp benchmark README, `full_eval.py`, `.ai/DECISIONS.md`, and local scratch `preds_full/eval_results.txt`; no benchmark was rerun.

`bash .dev/setup.sh` passed with the populated environment/model. `MPLBACKEND=Agg MPLCONFIGDIR=$PWD/.ai/local/mpl PYTHONPATH=$PWD .ai/local/dexnet3/bin/python examples/policy.py GQCNN-3.0 --model_dir .ai/local/models --depth_image data/examples/single_object/primesense/depth_0.npy --segmask data/examples/single_object/primesense/segmask_0.png --config_filename cfg/examples/replication/dex-net_3.0.yaml --result_dir .ai/local/results/example_0` passed. `.ai/local/dexnet3/bin/python tools/prepare_suctionnet_frame.py /home/wj/Desktop/Codes/graspnet_dataset --scene 100 --camera realsense --frame 0 --output_dir .ai/local/suctionnet/scene_0100/realsense/0000` passed and found 203,495 valid object pixels. The same policy with this converted frame, its `camera.intr`, and result directory `.ai/local/results/suctionnet_scene_0100_realsense_0000_verified` passed. Saved NPZ shape (1, 7), outward direction (0, 0, -1), and mask at selected pixel 255 were checked. `git diff --check`, `bash -n .dev/setup.sh`, Python compilation of changed scripts, and `pip check` passed. No broad test suite or official SuctionNet AP evaluation was run.

## Remaining work
The benchmark itself is complete and published. `test_similar`/`test_novel` remain out of scope for this run. A fresh environment build from `envs/dexnet3.yml` has not been independently repeated.

## Next steps
If extending the study, evaluate `test_similar`/`test_novel` and consider a Dex-Net model trained for RealSense; otherwise no required work remains.

## Current branch
master (tracking origin/master; 0/0 divergence verified)

## Last relevant commit
2fd1378 Record completed Dex-Net benchmark handoff

## Last agent
codex

## Timestamp
2026-10-19 07:20:27 UTC
