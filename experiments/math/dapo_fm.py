"""
Usage:
    python -m experiments.math.dapo_fm

    # With overrides:
    python -m experiments.math.dapo_fm wandb_name=my_run learning_rate=1e-5
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

            "gamma": 0.1,
            # Environment
            "env": "dapo",
            "horizon": 2,
            "early_termination": True,

            # Judge
            "judge_type": "judge",

            # Training
            "group_size": 8,
            "groups_per_batch": 32,
            "loss_fn": "importance_sampling",

            # Distillation
            "distillation_mode": "feedback_modeling",
            "sft_loss_weight": 0.1,

            # Logging
            "wandb_project": "dapo",
            "wandb_name": "feedback-modeling",
        }
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    cli_config = blueprint.make()

    asyncio.run(cli_main(cli_config))
