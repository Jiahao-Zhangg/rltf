"""
RL training for math problems with critique feedback.

This script extends the standard math RL training to include a critique feedback mechanism:
1. Generate initial solution y1
2. Get critique from judge model
3. Generate improved solution y2 based on critique
4. Train only on y2 (graded for correctness)
"""

import asyncio
import logging
from datetime import datetime

import chz
import tinker
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.rltf.envs import math as math_envs
from tinker_cookbook.recipes.rltf.envs import knights_knaves
from tinker_cookbook.recipes.rltf.envs import binary_matrix
from tinker_cookbook.recipes.rltf.envs import shortest_path
from tinker_cookbook.rl.train import Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from typing import Literal


logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for RL training with critique."""

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Judge model configuration
    judge_model_name: str = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    judge_renderer_name: str | None = None

    # Environment configuration
    env: str = "math"  # Options: math, polaris, deepmath, deepmath_hard, deepscaler, gsm8k, countdown, knights_knaves, hidden_op, count_primes, knight_swap, creative_writing
    test_envs: list[str] | None = None  # List of test environments (e.g., ["math500", "aime2024"]). If None, uses dataset default.
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    knights_knaves_num_problems: int = 20000

    # Training hyperparameters
    group_size: int = 8
    groups_per_batch: int = 64
    learning_rate: float = 2e-5
    max_tokens: int = 8096  # Need more tokens for two-stage generation
    kl_penalty_coef: float = 0.0

    # Number of optimizer steps per training iteration
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals
    eval_every: int = 20
    skip_first_eval: bool = True  # If True, skip evaluation at step 0
    eval_temperature: float | None = None  # Temperature for evaluation (None = use training temperature)
    eval_num_samples: int = 4  # Number of samples for multi-sample evaluation

    # Checkpointing
    save_every: int = 10

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Training configuration
    zero_out_first_transition: bool = False  # If True, don't train on y1, only on y2
    per_transition_advantages: bool = False  # If True, compute per-transition advantages for better credit assignment
    reward_y1: bool = True  # If True, give y1 actual reward based on correctness (instead of always 0)

    horizon: int = 2  # Number of stages
    
    # Tree structure hyperparameters
    group_from_stage: int | None = None  # Start tree branching from this stage (None = no tree mode)
    num_grouped_responses: int = 1  # How many responses to generate at each grouped stage (branching factor)
    early_termination: bool = False  # If True, end episode when generation is correct (for training)
    early_termination_test: bool = False  # If True, end episode when generation is correct (for test/eval)
    use_traj_return: bool = False  # If True, use trajectory-level returns for advantages (useful with early termination)

    judge_type: str = "judge"  # Options: "judge", "judge_with_correctness_only", "correctness", "judge_with_ground_truth"
    loss_fn: str = "importance_sampling"
    distillation_mode: str = "none"  # "rl" for RL distillation, "sft" for SFT distillation, "none" to disable

    # Dual critique mode: train model to generate its own critiques
    dual_critique_mode: bool = False  # If True, create dual paths (expert + model critique) for each problem
    model_judge_type: str = "judge_hint"  # Prompt type for model self-critique (can differ from expert judge_type)
    test_use_model_critique: bool = False  # If True, test uses model's own critique instead of expert judge

    # SFT distillation hyperparameters
    sft_loss_weight: float = 0.1  # Weight for combining SFT loss with RL loss (when using combined training)
    distillation_start_iteration: int = 0  # Start distillation only after this iteration (0 = start immediately)
    sft_buffer_size: int = 0  # Max SFT samples to keep in buffer (0 = disabled)

    seed: int = 0  # Random seed for data shuffling
    start_batch_index: int = 0  # For resuming training from a specific batch index

    # Discount factor for computing returns (used when use_traj_return=True)
    gamma: float = 0.1  

    # PPO/CISPO clipping thresholds 
    clip_low: float = 0.8  # Lower threshold for importance ratio clipping
    clip_high: float = 1.2  # Upper threshold for importance ratio clipping

    use_first_turn_baseline : bool = False  # If True, use first turn baseline for advantage estimation

    rl_coef: float | list[float] = 0.1  # Multiplier for advantages in RL distillation modes (can be [start, end] for linear schedule)

async def setup_judge_model(
    service_client: tinker.ServiceClient,
    judge_model_name: str,
    judge_renderer_name: str | None,
) -> tuple[TinkerTokenCompleter, renderers.Renderer]:
    """Set up the judge model for generating critiques."""
    logger.info(f"Setting up judge model: {judge_model_name}")

    # Create sampling client directly (no LoRA needed for inference-only judge)
    judge_sampling_client = service_client.create_sampling_client(base_model=judge_model_name)

    # Set up tokenizer and renderer
    judge_tokenizer = get_tokenizer(judge_model_name)
    if judge_renderer_name is None:
        judge_renderer_name = model_info.get_recommended_renderer_name(judge_model_name)
    judge_renderer = renderers.get_renderer(judge_renderer_name, tokenizer=judge_tokenizer)

    # Create completer
    judge_completer = TinkerTokenCompleter(
        sampling_client=judge_sampling_client,
        max_tokens=8096,  # For critique generation
    )

    return judge_completer, judge_renderer


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training with critique."""

    # Get renderer names
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )
    judge_renderer_name = cli_config.judge_renderer_name or model_info.get_recommended_renderer_name(
        cli_config.judge_model_name
    )

    # Create base dataset builder

    if cli_config.env == "knights_knaves":
        # Use Knights and Knaves dataset
        base_dataset_builder = knights_knaves.KnightsAndKnavesDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            num_problems=cli_config.knights_knaves_num_problems,
        )
    elif cli_config.env == "binary_matrix":
        # Use Binary Matrix dataset
        base_dataset_builder = binary_matrix.BinaryMatrixDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
        )
    elif cli_config.env == "shortest_path":
        # Use Shortest Path dataset
        base_dataset_builder = shortest_path.ShortestPathDatasetBuilder(
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
        )
    else:
        # Use regular math dataset with critique
        base_dataset_builder = math_envs.get_math_dataset_builder(
            dataset_name=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name_for_tokenizer=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            test_group_size=cli_config.test_group_size,
            test_envs=cli_config.test_envs,
        )

    # Create critique dataset builder (serializable, defers judge setup to __call__)
    @chz.chz
    class CritiqueDatasetBuilder:
        base_builder: RLDatasetBuilder
        service_client_base_url: str | None
        judge_model_name: str
        judge_renderer_name: str
        reward_y1: bool
        horizon: int
        group_from_stage: int | None
        num_grouped_responses: int
        early_termination: bool
        early_termination_test: bool
        judge_type: str
        # Dual critique parameters
        dual_critique_mode: bool = False
        model_judge_type: str = "judge_hint"
        test_use_model_critique: bool = False

        async def __call__(self):
            # Set up judge model at call time (not at config time)
            service_client = tinker.ServiceClient(base_url=self.service_client_base_url)

            if not (self.judge_type == "correctness"):
                judge_completer, judge_renderer = await setup_judge_model(
                    service_client=service_client,
                    judge_model_name=self.judge_model_name,
                    judge_renderer_name=self.judge_renderer_name,
                )
            else:
                judge_completer = None
                judge_renderer = None

            # Set up model completer for dual critique mode (model self-critique)
            # NOTE: model_completer will be None initially and will be dynamically updated
            # by the training loop via dataset.update_sampling_client(sampling_client)
            # This ensures the model uses its latest weights for self-critique
            model_completer = None
            if self.dual_critique_mode:
                logger.info("Dual critique mode enabled - model_completer will be updated dynamically from training loop")

            train_dataset, test_dataset = await self.base_builder()

            # Use appropriate wrapper based on dataset type
            if isinstance(self.base_builder, knights_knaves.KnightsAndKnavesDatasetBuilder):
                CritiqueWrapper = knights_knaves.KnightsAndKnavesCritiqueDataset
            elif isinstance(self.base_builder, binary_matrix.BinaryMatrixDatasetBuilder):
                CritiqueWrapper = binary_matrix.BinaryMatrixCritiqueDataset
            elif isinstance(self.base_builder, shortest_path.ShortestPathDatasetBuilder):
                CritiqueWrapper = shortest_path.ShortestPathCritiqueDataset
            else:
                CritiqueWrapper = math_envs.MathCritiqueDataset

            # Build kwargs that are common to all critique wrappers
            train_kwargs = {
                "reward_y1": self.reward_y1,
                "horizon": self.horizon,
                "group_responses_from_stage": self.group_from_stage,
                "num_grouped_responses": self.num_grouped_responses,
                "early_termination": self.early_termination,
                "judge_type": self.judge_type,
            }

            # Add dual critique parameters for supported critique wrappers
            if CritiqueWrapper in (math_envs.MathCritiqueDataset, knights_knaves.KnightsAndKnavesCritiqueDataset):
                train_kwargs.update({
                    "dual_critique_mode": self.dual_critique_mode,
                    "model_completer": model_completer,
                    "model_judge_type": self.model_judge_type,
                })
            elif self.dual_critique_mode:
                # Dual critique mode requested but not supported by this environment
                logger.warning(
                    f"dual_critique_mode=True but environment {self.base_builder.__class__.__name__} "
                    f"uses {CritiqueWrapper.__name__} which doesn't support dual critique mode yet. "
                    f"Dual critique mode is currently supported with math_envs.MathCritiqueDataset and knights_knaves.KnightsAndKnavesCritiqueDataset. "
                    f"Training will proceed without dual critique."
                )

            train_critique = CritiqueWrapper(
                train_dataset,
                judge_completer,
                judge_renderer,
                **train_kwargs,
            )

            # Handle test dataset(s): can be None, single RLDataset, or list of (name, RLDataset)
            if test_dataset is None:
                test_critique = None
            elif isinstance(test_dataset, list):
                # Multiple test datasets with names
                # Build kwargs for test datasets
                test_kwargs = {
                    "reward_y1": self.reward_y1,
                    "horizon": self.horizon,
                    "group_responses_from_stage": None,  # No tree rollouts for evaluation
                    "num_grouped_responses": 1,  # No tree rollouts for evaluation
                    "early_termination": self.early_termination_test,  # Use separate test setting
                    "judge_type": self.judge_type,
                }

                # Add dual critique parameters for supported critique wrappers
                if CritiqueWrapper in (math_envs.MathCritiqueDataset, knights_knaves.KnightsAndKnavesCritiqueDataset):
                    test_kwargs.update({
                        "dual_critique_mode": False,  # Test doesn't use dual paths
                        "model_completer": model_completer,
                        "model_judge_type": self.model_judge_type,
                        "use_model_critique": self.test_use_model_critique,  # Use model's own critique if requested
                    })

                test_critique = [
                    (name, CritiqueWrapper(
                        ds,
                        judge_completer,
                        judge_renderer,
                        **test_kwargs,
                    ))
                    for name, ds in test_dataset
                ]
            else:
                # Single test dataset
                test_kwargs = {
                    "reward_y1": self.reward_y1,
                    "horizon": self.horizon,
                    "group_responses_from_stage": None,  # No tree rollouts for evaluation
                    "num_grouped_responses": 1,  # No tree rollouts for evaluation
                    "early_termination": self.early_termination_test,  # Use separate test setting
                    "judge_type": self.judge_type,
                }

                # Add dual critique parameters for supported critique wrappers
                if CritiqueWrapper in (math_envs.MathCritiqueDataset, knights_knaves.KnightsAndKnavesCritiqueDataset):
                    test_kwargs.update({
                        "dual_critique_mode": False,  # Test doesn't use dual paths
                        "model_completer": model_completer,
                        "model_judge_type": self.model_judge_type,
                        "use_model_critique": self.test_use_model_critique,  # Use model's own critique if requested
                    })

                test_critique = CritiqueWrapper(
                    test_dataset,
                    judge_completer,
                    judge_renderer,
                    **test_kwargs,
                )

            return train_critique, test_critique
        
    critique_dataset_builder = CritiqueDatasetBuilder(
        base_builder=base_dataset_builder,
        service_client_base_url=cli_config.base_url,
        judge_model_name=cli_config.judge_model_name,
        judge_renderer_name=judge_renderer_name,
        reward_y1=cli_config.reward_y1,
        horizon=cli_config.horizon,
        group_from_stage=cli_config.group_from_stage,
        num_grouped_responses=cli_config.num_grouped_responses,
        early_termination=cli_config.early_termination,
        early_termination_test=cli_config.early_termination_test,
        judge_type=cli_config.judge_type,
        # Dual critique parameters
        dual_critique_mode=cli_config.dual_critique_mode,
        model_judge_type=cli_config.model_judge_type,
        test_use_model_critique=cli_config.test_use_model_critique,
    )
    dataset_builder = critique_dataset_builder


    # Set up logging
    model_name = cli_config.model_name.replace("/", "-")
    run_name = f"{cli_config.env}-critique-{model_name}-{cli_config.judge_type}-{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-{cli_config.loss_fn}-distill_{cli_config.distillation_mode}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"./tinker-examples/math_rl_critique/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    # Create full config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        skip_first_eval=cli_config.skip_first_eval,
        eval_temperature=cli_config.eval_temperature,
        eval_num_samples=cli_config.eval_num_samples,
        save_every=cli_config.save_every,
        loss_fn=cli_config.loss_fn,
        distillation_mode=cli_config.distillation_mode,
        sft_loss_weight=cli_config.sft_loss_weight,
        distillation_start_iteration=cli_config.distillation_start_iteration,
        early_termination=cli_config.early_termination,
        start_batch_index=cli_config.start_batch_index,
        gamma=cli_config.gamma,
        clip_low=cli_config.clip_low,
        clip_high=cli_config.clip_high,
        use_first_turn_baseline=cli_config.use_first_turn_baseline,
        rl_coef=cli_config.rl_coef,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Base Model: {cli_config.model_name}")
    logger.info(f"  Judge Model: {cli_config.judge_model_name}")
    logger.info(f"  Environment: {cli_config.env}")
    logger.info(f"  Group Size: {cli_config.group_size}")
    logger.info(f"  Groups per Batch: {cli_config.groups_per_batch}")
    logger.info(f"  Learning Rate: {cli_config.learning_rate}")
    logger.info(f"  Reward Y1: {cli_config.reward_y1}")
    logger.info(f"  Zero out first transition: {cli_config.zero_out_first_transition}")
    logger.info(f"  Per-transition advantages: {cli_config.per_transition_advantages}")
    logger.info(f"  Tree mode: {cli_config.group_from_stage is not None}")
    if cli_config.group_from_stage is not None:
        logger.info(f"    Group from stage: {cli_config.group_from_stage}")
        logger.info(f"    Num grouped responses: {cli_config.num_grouped_responses}")
    logger.info(f"  Early termination (train): {cli_config.early_termination}")
    logger.info(f"  Early termination (test): {cli_config.early_termination_test}")
    logger.info(f"  Use trajectory-level returns: {cli_config.use_traj_return}")
    logger.info(f"  Log Path: {log_path}")
    logger.info("=" * 80)

    # Run training
    await main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
