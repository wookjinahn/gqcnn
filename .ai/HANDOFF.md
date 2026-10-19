# Work handoff

## Current task
Document, commit, and push the completed Dex-Net 3.0 Conda setup and example/SuctionNet inference work.

## Goal
Publish the reproducible setup, verified commands, results, and handoff without adding models, datasets, local environments, or credentials to Git.

## Completed work
Pushed commits `56651dc` and `e9c75c9` to `origin/master`; verified HEAD and the tracking branch match at `e9c75c9`, divergence is 0/0, and working/staged diffs are empty before this final checkpoint.

Read project instructions and previous handoff; inspected git status, both diffs, and recent log. Created the ignored `.ai/local/dexnet3` Conda environment with Python 3.7.16 and compatible packages. Downloaded the GQ-CNN 3.0 model from the official Dex-Net page's Google Drive link, verified ZIP integrity and SHA256, and extracted it to `.ai/local/models/GQCNN-3.0`. Ran the repository Primesense example and SuctionNet test_seen scene 0100 RealSense frame 0000. Added a frame converter, policy result JSON/PNG/NPZ output, an environment file, reproducible setup script, and usage documentation. Updated AGENTS.md with measured versions and appended a decision in DECISIONS.md. Expanded the usage guide with installation prerequisites, verified dependency versions, commands, and measured results; linked it from README.md. Reviewed explicit staged paths and diff, verified ignored local assets, fetched origin, and committed the code/docs as `56651dc`.

## Current state
Both initial inference paths work on CPU. Example frame 0 produced Q=0.5994 at pixel (365.77, 224.19), depth 0.707 m. SuctionNet scene 0100 RealSense frame 0000 produced Q=0.2429 at pixel (418.39, 218.36), depth 0.430 m; the selected pixel lies inside the object mask. Verified output is in `.ai/local/results/suctionnet_scene_0100_realsense_0000_verified/`. The user was asked whether to stop at one-frame inference or proceed to a full test-split evaluation; no answer yet. Code and docs are committed as `56651dc`; commits 56651dc and e9c75c9 were pushed to origin/master; matching tracking refs and clean diffs were verified before this final checkpoint. Existing worktree changes were preserved.

## Important findings
At startup master was aligned with origin/master at 499a609; after the first push both committed refs matched at e9c75c9 with 0/0 divergence. Host Ubuntu 22.04.5 and Conda 26.3.2; Python 3.13 is incompatible with this legacy code. The working environment has TensorFlow 1.15.0, NumPy 1.19.5, autolab-core 1.1.1, autolab-perception 1.0.0, visualization 1.0.0, pyglet 1.5.31, and ruamel.yaml 0.17.40; see `envs/dexnet3.yml`. GQ-CNN 3.0 was trained for Primesense Carmine, while the tested SuctionNet frame uses RealSense; Q is not benchmark accuracy. Local SuctionNet data includes test_seen scenes 0100–0129 under `/home/wj/Desktop/Codes/graspnet_dataset`. The dataset's depth factor is 1000 for the tested frame; metadata supplies intrinsics. SuctionNet's evaluation array uses `[score, outward_direction(3), camera_point_m(3)]`; the GQ-CNN approach axis is negated. Model ZIP SHA256: `d048afb9974d4c3170ccb89289302ea871b795d421ee3910805675f35da7b1f6`.

## Modified files
Commit `56651dc` contains `.gitignore`, AGENTS.md, CLAUDE.md, README.md, `examples/policy.py`, `.dev/setup.sh`, `envs/dexnet3.yml`, `tools/prepare_suctionnet_frame.py`, `docs/dexnet3_suctionnet.md`, and `.ai/DECISIONS.md`. The first handoff commit e9c75c9 was pushed; this final publication checkpoint updates the same file. Ignored `.ai/local/` assets were not staged.

## Failed approaches
Host Python invocation failed with `ModuleNotFoundError: numpy`. Initial sandboxed Conda and pip network calls failed DNS resolution; escalation allowed downloads. First Conda solve with `pip=23.0` conflicted with Python 3.7; removing the pin selected pip 22.3.1. Repository Box model URL returned HTTP 404; the official Dex-Net page linked an accessible Google Drive ZIP. Initial dependency import failed because pyglet 2.0.10 requires Python 3.8; pinned 1.5.31. First policy run failed because ruamel.yaml 0.18 removed `yaml.load`; pinned 0.17.40. Docker daemon was previously inaccessible in the sandbox and was not needed. The first `git add` failed because `.git/index.lock` could not be created in the read-only sandbox; the approved escalated retry succeeded.

## Test/build results
`bash .dev/setup.sh` passed with the populated environment/model. `MPLBACKEND=Agg MPLCONFIGDIR=$PWD/.ai/local/mpl PYTHONPATH=$PWD .ai/local/dexnet3/bin/python examples/policy.py GQCNN-3.0 --model_dir .ai/local/models --depth_image data/examples/single_object/primesense/depth_0.npy --segmask data/examples/single_object/primesense/segmask_0.png --config_filename cfg/examples/replication/dex-net_3.0.yaml --result_dir .ai/local/results/example_0` passed. `.ai/local/dexnet3/bin/python tools/prepare_suctionnet_frame.py /home/wj/Desktop/Codes/graspnet_dataset --scene 100 --camera realsense --frame 0 --output_dir .ai/local/suctionnet/scene_0100/realsense/0000` passed and found 203,495 valid object pixels. The same policy with this converted frame, its `camera.intr`, and result directory `.ai/local/results/suctionnet_scene_0100_realsense_0000_verified` passed. Saved NPZ shape (1, 7), outward direction (0, 0, -1), and mask at selected pixel 255 were checked. `git diff --check`, `bash -n .dev/setup.sh`, Python compilation of changed scripts, and `pip check` passed. No broad test suite or official SuctionNet AP evaluation was run.

## Remaining work
A full SuctionNet test-split evaluation remains outside the completed one-frame smoke test and awaits the user's requested scope. It requires inference for all frames and running the SuctionNet API; a fresh environment build from `envs/dexnet3.yml` has not been independently repeated.

## Next steps
If the user requests full evaluation, implement persistent batch inference and run the test_seen split using available dataset and labels, preserving benchmark work in other repositories.

## Current branch
master (tracking origin/master; 0/0 divergence verified after the first push, before this final checkpoint commit)

## Last relevant commit
e9c75c9 Record Dex-Net 3.0 setup handoff (substantive code/doc commit: 56651dc)

## Last agent
codex

## Timestamp
2026-10-19 02:42:47 UTC
