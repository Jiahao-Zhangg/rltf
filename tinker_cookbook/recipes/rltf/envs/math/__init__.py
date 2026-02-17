"""Math environments for RL training."""

from tinker_cookbook.recipes.rltf.envs.math.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.recipes.rltf.envs.math.math_env import (
    MathEnv,
    MathDataset,
    MathDatasetBuilder,
    Gsm8kDataset,
    Gsm8kDatasetBuilder,
    PolarisDataset,
    PolarisDatasetBuilder,
    DeepMathDataset,
    DeepMathDatasetBuilder,
    DeepMathHardDataset,
    DeepMathHardDatasetBuilder,
    DeepScaleRDataset,
    DeepScaleRDatasetBuilder,
    Math500Dataset,
    BeyondAIMEDataset,
    AIME2024Dataset,
    AIME2025Dataset,
    DAPODataset,
    DAPODatasetBuilder,
    safe_grade,
    extract_gsm8k_final_answer,
    create_test_datasets,
    get_math_dataset_builder,
)
from tinker_cookbook.recipes.rltf.envs.math.math_env_critique import (
    MathEnvWithCritique,
    MathCritiqueGroupBuilder,
    MathCritiqueDataset,
)
from tinker_cookbook.recipes.rltf.envs.math.math_env_dual_critique import (
    MathEnvWithDualCritique,
)

__all__ = [
    # Grading
    "extract_boxed",
    "grade_answer",
    "grade_answer_math_verify",
    "run_with_timeout_signal",
    # Base env
    "MathEnv",
    "MathDataset",
    "MathDatasetBuilder",
    "safe_grade",
    "extract_gsm8k_final_answer",
    "create_test_datasets",
    "get_math_dataset_builder",
    # Dataset variants
    "Gsm8kDataset",
    "Gsm8kDatasetBuilder",
    "PolarisDataset",
    "PolarisDatasetBuilder",
    "DeepMathDataset",
    "DeepMathDatasetBuilder",
    "DeepMathHardDataset",
    "DeepMathHardDatasetBuilder",
    "DeepScaleRDataset",
    "DeepScaleRDatasetBuilder",
    "Math500Dataset",
    "BeyondAIMEDataset",
    "AIME2024Dataset",
    "AIME2025Dataset",
    "DAPODataset",
    "DAPODatasetBuilder",
    # Critique envs
    "MathEnvWithCritique",
    "MathCritiqueGroupBuilder",
    "MathCritiqueDataset",
    "MathEnvWithDualCritique",
]
