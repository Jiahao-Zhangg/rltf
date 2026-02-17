"""
Usage:
    python -m experiments.knights_knaves_critique

    # With overrides:
    python -m experiments.knights_knaves_critique wandb_name=my_run learning_rate=1e-5
"""

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

    # Build blueprint with defaults, then apply CLI overrides
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    cli_config = blueprint.make()

    asyncio.run(cli_main(cli_config))
