# Architecture and design decisions

Append dated decisions with context, alternatives, rationale and consequences.
Do not overwrite earlier decisions; mark superseded decisions explicitly.

## 2026-10-19 — Isolated Dex-Net 3.0 inference environment and SuctionNet adapter

Context: The checked-out GQ-CNN code targets Python 3.5–3.7 and TensorFlow 1.15, while the Ubuntu 22.04 host has Python 3.13. The user requested Conda and plans to test the pre-trained suction policy on SuctionNet images.

Alternatives: Install legacy dependencies in Conda base; use the repository's Ubuntu 16.04 Docker image; retrain or port the model to a modern framework.

Decision and rationale: Keep Python 3.7 dependencies and downloaded model in ignored `.ai/local/`, define the direct package versions in `envs/dexnet3.yml`, and convert individual SuctionNet frames to the existing policy's depth, mask, and camera-intrinsics inputs. This preserves the user's other environments and allows the repository's original inference path to run on CPU.

Consequences: The legacy environment must be recreated for another machine. Predictions on RealSense SuctionNet images are out of the model's Primesense training domain; a single-image Q value is not a benchmark accuracy result. SuctionNet's evaluation row uses the outward direction, which is opposite the GQ-CNN approach axis.
