# Expanding the Capabilities of Reinforcement Learning via Text Feedback

Yuda Song\*, Lili Chen\*, Fahim Tajwar, Rémi Munos, Deepak Pathak, J. Andrew Bagnell, Aarti Singh, Andrea Zanette

\*equal contribution

[[Paper]](https://arxiv.org/abs/2602.02482) [[Website]](https://rl-textfeedback.github.io/)

## Overview

Official codebase for Expanding the Capabilities of Reinforcement Learning via Text Feedback.

![Reinforcement Learning with Text Feedback](./teaser.png)

## Installation
Our codebase uses tinker, so please follow the setup from the official tinker-cookbook.

1. Sign up for Tinker through the [waitlist](https://thinkingmachines.ai/tinker).
2. Once you have access, create an API key from the [console](https://tinker-console.thinkingmachines.ai) and export it as environment variable `TINKER_API_KEY`.
3. Install tinker python client via `pip install tinker`
4. We recommend installing `tinker-cookbook` in a virtual env either with `conda` or `uv`. For running most examples, you can install via `pip install -e .`.

## Experiments

This experiment folder contains experiment scripts for RL training with text feedback. Each script defines preset configurations for reproducible experiments.

### Structure

```
experiments/
├── reasoning/          # Reasoning tasks (Knights & Knaves, Binary Matrix, Shortest Path)
│   ├── knights_knaves_grpo.py
│   ├── knights_knaves_sd.py
│   ├── knights_knaves_fm.py
│   ├── binary_matrix_*.py
│   └── shortest_path_*.py
└── math/               # Math tasks (DeepMath, DAPO)
    ├── deepmath_grpo.py
    ├── deepmath_sd.py
    ├── dapo_*.py
    └── ...
```

### Methods

| Method | Description | Key Config |
|--------|-------------|------------|
| `grpo` | GRPO baseline (no distillation) | `distillation_mode="none"` |
| `fm` | Feedback Modeling | `distillation_mode="feedback_modeling"` |
| `sd` | Self-Distillation | `distillation_mode="rl_reweight_mask"` |
| `sft` | SFT on correct y2 | `distillation_mode="sft"` |

### Usage

Run an experiment with default settings:

```bash
python -m experiments.reasoning.knights_knaves_sd
```

Override specific parameters:

```bash
python -m experiments.reasoning.knights_knaves_sd wandb_name=my_run learning_rate=1e-5
```


### Example Script

Each experiment script follows this pattern:

```python
import asyncio
import logging
import sys

import chz
from tinker_cookbook.recipes.rltf.train_with_critique import CLIConfig, cli_main


def build_config_blueprint() -> chz.Blueprint[CLIConfig]:
    """Build config blueprint with experiment defaults."""
    return chz.Blueprint(CLIConfig).apply(
        {
            # Model
            "model_name": "meta-llama/Llama-3.1-8B-Instruct",

            # Environment
            "env": "knights_knaves",
            "horizon": 2,
            "early_termination": True,

            # Judge
            "judge_type": "judge",

            # Training
            "group_size": 8,
            "groups_per_batch": 32,
            "loss_fn": "importance_sampling",

            # Distillation
            "distillation_mode": "rl_reweight_mask",
            "rl_coef": 0.1,
            "use_first_turn_baseline": True,
            "gamma": 0.1,

            # Logging
            "wandb_project": "knights",
            "wandb_name": "self-distillation",
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    cli_config = blueprint.make()

    asyncio.run(cli_main(cli_config))
```

### Available Environments

| Environment | Description | Script prefix |
|-------------|-------------|---------------|
| `knights_knaves` | Knights and Knaves logic puzzles | `knights_knaves_*` |
| `binary_matrix` | Binary matrix distance computation | `binary_matrix_*` |
| `shortest_path` | Grid pathfinding | `shortest_path_*` |
| `deepmath` | DeepMath math dataset | `deepmath_*` |
| `dapo` | DAPO math dataset | `dapo_*` |

## Citation

```bibtex
@article{song2026expanding,
  title={Expanding the Capabilities of Reinforcement Learning via Text Feedback},
  author={Song, Yuda and Chen, Lili and Tajwar, Fahim and Munos, Remi and Pathak, Deepak and Bagnell, J Andrew and Singh, Aarti and Zanette, Andrea},
  journal={arXiv preprint arXiv:2602.02482},
  year={2026}
}
```
