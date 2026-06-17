# DROM: Multi-Skill Robotic Manipulation from Single Demonstration via Language-Guided Diffusion

![DROM Framework](DROM_framework.jpg)

## Abstract

Learning data-efficient and generalizable manipulation policies remains a central challenge in robotics, particularly for multi-skill and long-horizon tasks. We present DROM, a data-efficient, language-guided, multi-skill diffusion-based framework for robotic manipulation. DROM leverages Dynamic Movement Primitives (DMPs) to augment a single human demonstration per skill, generating a compact yet expressive multi-skill dataset used to train a unified diffusion policy. Extending the MPD formulation, we introduce a single diffusion model capable of handling multiple manipulation primitives, conditioned through a language encoder that maps diverse operator prompts to the corresponding skill. For long-horizon objectives, a high-level language model decomposes instructions into ordered primitives, which condition the diffusion model via cross-attention to generate skill-consistent trajectories while preserving real-time control. We validate DROM on two robotic platforms, a Franka Emika Panda and a FANUC CRX25ia, as well as in MuJoCo simulation, demonstrating robust multi-skill generalization and high task success with minimal demonstration requirements.

## Requirements

- Linux, Python >= 3.8
- An NVIDIA GPU with CUDA support (training and rollout sampling are GPU-accelerated)

Python dependencies (installed automatically, see [setup.py](setup.py)) include `robosuite`, `mimicgen`, `diffusers`, `torch`, and `egl_probe` for rendering.

## Installation

```bash
git clone https://github.com/vincenzopomponi/DROM_IROS.git
cd DROM_IROS
source install.sh
```

`install.sh` creates a virtual environment in `.env`, installs the `drom` package and its dependencies, and installs PyTorch with CUDA 12.8 support. Activate the environment in future sessions with `source .env/bin/activate`.

## Usage

### Train the Diffusion Model

```bash
python drom/scripts/train_drom.py --config drom/configs/low_dim/drom.json
```

Training configurations live under [drom/configs](drom/configs) (`low_dim/` and `image/` variants). Edit a config to point `train.data` at your own dataset(s) and language-conditioning prompts, or copy one as a starting point for a new experiment.

### Evaluate the trained policy

```bash
python drom/scripts/test_drom.py \
  --directory <path-to-trained-model-dir> \
  --validations 30 \
  --seed 30 \
  --render
```

Key arguments:

| Argument | Description | Default |
|---|---|---|
| `--directory` | Path to the trained model directory (containing `config.json` and checkpoints) | required |
| `--ckpt` | Checkpoint name, or `last` to use the most recent checkpoint | `last` |
| `--seed` | Random seed for evaluation | `30` |
| `--samples` | Trajectories sampled per validation case | `10` |
| `--validations` | Number of validation cases to run | `5` |
| `--camera` | Camera view used for rendering | `frontview` |
| `--render` | Enable rendering of rollouts | off |

## Repository structure

```
drom/
├── configs/         # Experiment configs (low_dim and image observation variants)
├── datasets/        # Dataset loading and normalization
├── demos/           # Folder to store acquired/generated demonstrations
├── models/          # Diffusion model, language-conditioning, and policy architectures
├── losses/          # Diffusion training losses
├── trainer/         # Training loop and data loaders
├── scripts/         # Entry points: train, test scripts
├── plotting/        # Trajectory and rollout visualization
└── utils/           # Shared helpers (HDF5 I/O, seeding, W&B logging, etc.)
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Contact

For questions or issues, please open a GitHub issue or contact Vincenzo Pomponi (vincenzo.pomponi@supsi.ch).
