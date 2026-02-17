import math
import re
from functools import partial
from typing import Literal, Sequence, cast

import chz
from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.recipes.rltf.envs.math.math_grading import (
    extract_boxed,
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


class MathEnv(ProblemEnv):
    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "math_verify",
        timeout: float = 1.0,
    ):
        super().__init__(renderer, convo_prefix)
        self.problem = problem
        self.answer = answer
        self.grader = grader
        self.timeout = timeout

    @classmethod
    def question_suffix(cls) -> str:
        # return " Write your answer in \\boxed{} format."
        return " Let's think step by step and output the final answer within \\boxed{}."

    def get_question(self) -> str:
        return self.problem + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        try:
            answer = extract_boxed(sample_str)
        except ValueError:
            return False
        return safe_grade(answer, self.answer, self.grader, self.timeout)

    def get_reference_answer(self) -> str:
        return self.answer

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": "How many r's are in strawberry?" + MathEnv.question_suffix(),
            },
            {
                "role": "assistant",
                "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
            },
        ]


def safe_grade(given_answer: str, ground_truth: str, grader: str = "math_verify", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(
        grader_func, args=(given_answer, ground_truth), timeout_seconds=int(math.ceil(timeout))
    )
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out


def extract_gsm8k_final_answer(text: str) -> str:
    """Extract the final numeric/string answer from a GSM8K solution field.

    GSM8K format typically places the final answer on a line starting with
    '####'. We take the substring following '####' on the last such line.
    """
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            content = content.replace(",", "").strip()
            return content
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip()
    raise ValueError("No GSM8K final answer found")


def _get_hendrycks_math_test() -> Dataset:
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return cast(Dataset, test_dataset)


def _get_beyond_aime_test() -> Dataset:
    test_dataset = load_dataset("ByteDance-Seed/BeyondAIME", "default", split="test")
    return cast(Dataset, test_dataset)


def _get_aime_2024_test() -> Dataset:
    # Load AIME 2024 dataset (contains 30 problems repeated 32 times for best-of-32 metrics)
    # We'll use only the first 30 unique problems for evaluation
    full_dataset = load_dataset("BytedTsinghua-SIA/AIME-2024", "default", split="train")
    # Select only first 30 rows (the unique problems)
    test_dataset = full_dataset.select(range(30))
    return cast(Dataset, test_dataset)


def _get_aime_2025_test() -> Dataset:
    # Load AIME 2025 dataset (30 problems from AIME 2025)
    test_dataset = load_dataset("math-ai/aime25", split="test")
    return cast(Dataset, test_dataset)


def _get_hendrycks_math_train() -> Dataset:
    # For Hendrycks MATH, the standard is to use both the "train" and "test" splits for
    # training. The "test" split here is NOT the same as the MATH-500 test split above,
    # which is a commonly-held-out subset of 500 of the below 12.5k problems. To construct
    # a clean training set, we filter out problems that exist in the MATH-500 test set,
    # resulting in 12000 train and 500 test problems.

    test_problems: set[str] = {
        problem["problem"]  # pyright: ignore[reportArgumentType, reportCallIssue]
        for problem in _get_hendrycks_math_test()
    }

    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            ds = ds.filter(lambda example: example["problem"] not in test_problems)
            pieces.append(ds)
    full_dataset = concatenate_datasets(pieces)

    return full_dataset


class MathDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split == "train":
            self.ds = _get_hendrycks_math_train().shuffle(seed=seed)
        elif split == "test":
            self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            answer = extract_boxed(x["solution"])
        except ValueError:  # not sure if this happens
            logger.warning(f"No answer found for {x['solution']}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, x["problem"], answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class MathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = None
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["math500", "aime2024"])  # List of test environments

    async def __call__(self) -> tuple[MathDataset, list[tuple[str, MathDataset]]]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset
        train_dataset = MathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
            seed=self.seed,
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
            convo_prefix=convo_prefix,
        )

        return (train_dataset, test_datasets)


class PolarisDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        if split == "train":
            self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(seed=seed)
        else:
            # Use BeyondAIME test set
            self.ds = load_dataset("ByteDance-Seed/BeyondAIME", "default", split="test")
            logger.info(f"Polaris test: Using {len(self.ds)} problems from BeyondAIME")

        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        # if answer is not str, convert to str
        if not isinstance(answer, str):
            answer = str(answer)
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="polaris",
        )


@chz.chz
class PolarisDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["math500", "aime2024"])  # List of test environments

    async def __call__(self) -> tuple[PolarisDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_dataset = PolarisDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                split="train",
                seed=self.seed,
            )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
        )

        return (train_dataset, test_datasets)   


class DeepMathDataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        # Don't call super().__init__ since we're overriding the dataset loading
        full_ds = load_dataset("zwhe99/DeepMath-103K", split="train")

        if split == "train":
            # Use all but last 500 for training
            self.ds = full_ds.select(range(len(full_ds) - 500)).shuffle(seed=seed)
        else:
            # Use last 500 for testing
            self.ds = full_ds.select(range(len(full_ds) - 500, len(full_ds)))

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", "")
        answer = x.get("final_answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepmath",
        )


@chz.chz
class DeepMathDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["math500", "aime2024"])  # List of test environments

    async def __call__(self) -> tuple[DeepMathDataset, list[tuple[str, MathDataset]]]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset: DeepMath
        train_dataset = DeepMathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            split="train",
            seed=self.seed,
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
        )

        return (train_dataset, test_datasets)


class DeepMathHardDataset(MathDataset):
    """DeepMath dataset filtered to only include problems with difficulty > 6."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        test_group_size: int = 1,
    ):
        if split == "train":
            # Load DeepMath and filter by difficulty > 6
            full_ds = load_dataset("zwhe99/DeepMath-103K", split="train")

            if "difficulty" in full_ds.column_names:
                self.ds = full_ds.filter(lambda x: x["difficulty"] > 6).shuffle(seed=0)
                logger.info(f"DeepMathHard: Filtered to {len(self.ds)} problems with difficulty > 6")
            else:
                logger.warning(f"DeepMath dataset has no 'difficulty' field. Available fields: {full_ds.column_names}")
                logger.warning("Using all DeepMath data as fallback")
                self.ds = full_ds.shuffle(seed=0)
        else:
            # Use BeyondAIME test set
            self.ds = load_dataset("ByteDance-Seed/BeyondAIME", "default", split="test")
            logger.info(f"DeepMathHard test: Using {len(self.ds)} problems from BeyondAIME")

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else test_group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        problem = x.get("question", x.get("problem", ""))
        answer = x.get("final_answer", x.get("answer", ""))
        if not (problem and answer):
            return None
        # Convert answer to string (BeyondAIME dataset has integer answers)
        answer = str(answer)
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepmath_hard",
        )


@chz.chz
class DeepMathHardDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["beyondaime"])  # List of test environments (default: BeyondAIME for hard problems)

    async def __call__(self) -> tuple[DeepMathHardDataset, list[tuple[str, MathDataset]]]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset
        train_dataset = DeepMathHardDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            split="train",
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
        )

        return (train_dataset, test_datasets)


class DeepScaleRDataset(MathDataset):
    """DeepScaleR dataset with BeyondAIME test set."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
        test_group_size: int = 1,
    ):
        if split == "train":
            # Load DeepScaleR dataset
            self.ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train").shuffle(seed=seed)
            logger.info(f"DeepScaleR: Loaded {len(self.ds)} problems from DeepScaleR-Preview-Dataset")
        else:
            # Use BeyondAIME test set
            self.ds = load_dataset("ByteDance-Seed/BeyondAIME", "default", split="test")
            logger.info(f"DeepScaleR test: Using {len(self.ds)} problems from BeyondAIME")

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else test_group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the dataset
        # DeepScaleR has "problem" and "answer" fields
        # BeyondAIME also has "problem" and "answer" fields
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        # Convert answer to string (BeyondAIME dataset has integer answers)
        answer = str(answer)
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="deepscaler",
        )


@chz.chz
class DeepScaleRDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["beyondaime"])  # List of test environments (default: BeyondAIME for hard problems)

    async def __call__(self) -> tuple[DeepScaleRDataset, list[tuple[str, MathDataset]]]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset
        train_dataset = DeepScaleRDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            split="train",
            seed=self.seed,
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
        )

        return (train_dataset, test_datasets)


class Gsm8kDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    @classmethod
    def question_suffix(cls) -> str:
        return " Provide a numerical answer without units, written inside \\boxed{}."

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse GSM8K row: {e}")
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
        )


@chz.chz
class Gsm8kDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    convo_prefix: list[renderers.Message] | None | Literal["standard"] = "standard"
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["math500", "aime2024"])  # List of test environments

    async def __call__(self) -> tuple[Gsm8kDataset, list[tuple[str, MathDataset]]]:
        if self.convo_prefix == "standard":
            convo_prefix = MathEnv.standard_fewshot_prefix()
        else:
            convo_prefix = self.convo_prefix
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset
        train_dataset = Gsm8kDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=convo_prefix,
            split="train",
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
            convo_prefix=convo_prefix,
        )

        return (train_dataset, test_datasets)


class Math500Dataset(MathDataset):
    """Simple test dataset for MATH-500 problems."""
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        group_size: int = 2,  # Number of generations per problem
    ):
        self.ds = _get_hendrycks_math_test()
        self.batch_size = batch_size
        self.group_size = 2
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        problem = x.get("problem", "")
        try:
            answer = extract_boxed(x["solution"])
        except (ValueError, KeyError):
            return None
        if not problem:
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="math500",
        )


class BeyondAIMEDataset(MathDataset):
    """Simple test dataset for BeyondAIME problems."""
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        group_size: int = 4,  # Number of generations per problem
    ):
        self.ds = _get_beyond_aime_test()
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        # Convert answer to string (BeyondAIME dataset has integer answers)
        answer = str(answer)
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="beyondaime",
        )


class AIME2024Dataset(MathDataset):
    """Simple test dataset for AIME 2024 problems."""
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        group_size: int = 4,  # Number of generations per problem
    ):
        self.ds = _get_aime_2024_test()
        self.batch_size = batch_size
        self.group_size = 4
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict, group_size: int
    ) -> ProblemGroupBuilder | None:
        # AIME-2024 schema: prompt is a list of messages, answer is in reward_model.ground_truth
        prompt_messages = x.get("prompt", [])
        if not prompt_messages:
            return None
        problem = prompt_messages[0].get("content", "")

        reward_model = x.get("reward_model", {})
        answer = reward_model.get("ground_truth", "")

        if not (problem and answer):
            return None

        # Answer is already a string
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="aime2024",
        )


class AIME2025Dataset(MathDataset):
    """Simple test dataset for AIME 2025 problems."""
    def __init__(
        self,
        batch_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        group_size: int = 4,  # Number of generations per problem
    ):
        self.ds = _get_aime_2025_test()
        self.batch_size = batch_size
        self.group_size = 4
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict, group_size: int
    ) -> ProblemGroupBuilder | None:
        # AIME-2025 schema: simple problem/answer format
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        # Convert answer to string (in case it's numeric)
        answer = str(answer)
        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="aime2025",
        )


class DAPODataset(MathDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        self.split = split
        if split == "train":
            # Load DAPO dataset for training
            full_ds = load_dataset("ftajwar/deduplicated_dapo_dataset", split="train")
            self.ds = full_ds.shuffle(seed=seed)
        else:
            # Use MATH-500 for testing
            self.ds = _get_hendrycks_math_test()

        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        if self.split == "train":
            # DAPO schema
            problem = x.get("prompt", "")
            answer = x.get("answer", "")
            if not (problem and answer):
                return None
            # Wrap answer in \boxed{} format to match MATH dataset convention
            answer = f"\\boxed{{{answer}}}"
        else:
            # MATH-500 schema
            problem = x.get("problem", "")
            try:
                answer = extract_boxed(x["solution"])
            except (ValueError, KeyError):
                return None
            if not problem:
                return None

        return ProblemGroupBuilder(
            env_thunk=partial(
                MathEnv, problem, answer, self.renderer, convo_prefix=self.convo_prefix
            ),
            num_envs=group_size,
            dataset_name="dapo",
        )


@chz.chz
class DAPODatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    test_group_size: int = 1  # Number of generations per test problem for more robust evaluation
    test_envs: list[str] = chz.field(default_factory=lambda: ["math500", "aime2024"])  # List of test environments

    async def __call__(self) -> tuple[DAPODataset, list[tuple[str, MathDataset]]]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        # Training dataset: DAPO
        train_dataset = DAPODataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            split="train",
            seed=self.seed,
        )

        # Test datasets: configurable via test_envs
        test_datasets = create_test_datasets(
            test_env_names=self.test_envs,
            batch_size=self.batch_size,
            renderer=renderer,
            test_group_size=self.test_group_size,
        )

        return (train_dataset, test_datasets)


# Populate the dataset builder map after all classes are defined
DATASET_BUILDER_MAP = {
    "math": MathDatasetBuilder,
    "polaris": PolarisDatasetBuilder,
    "deepmath": DeepMathDatasetBuilder,
    "deepmath_hard": DeepMathHardDatasetBuilder,
    "deepscaler": DeepScaleRDatasetBuilder,
    "gsm8k": Gsm8kDatasetBuilder,
    "dapo": DAPODatasetBuilder,
}

# Test dataset mapping
TEST_DATASET_MAP = {
    "math500": Math500Dataset,
    "aime2024": AIME2024Dataset,
    "aime2025": AIME2025Dataset,
    "beyondaime": BeyondAIMEDataset,
}


def create_test_datasets(
    test_env_names: list[str],
    batch_size: int,
    renderer: renderers.Renderer,
    test_group_size: int = 1,
    convo_prefix: list[renderers.Message] | None = None,
) -> list[tuple[str, MathDataset]]:
    """
    Create test datasets from a list of test environment names.

    Args:
        test_env_names: List of test env names (e.g., ["math500", "aime2024"])
        batch_size: Batch size for test datasets
        renderer: Renderer to use
        test_group_size: Number of generations per test problem
        convo_prefix: Optional conversation prefix

    Returns:
        List of (name, dataset) tuples
    """
    test_datasets = []
    for name in test_env_names:
        if name not in TEST_DATASET_MAP:
            available = ", ".join(TEST_DATASET_MAP.keys())
            raise ValueError(f"Unknown test env '{name}'. Available: {available}")

        dataset_class = TEST_DATASET_MAP[name]
        dataset = dataset_class(
            batch_size=batch_size,
            renderer=renderer,
            group_size=test_group_size,
            convo_prefix=convo_prefix,
        )
        test_datasets.append((name, dataset))

    return test_datasets


def get_math_dataset_builder(
    dataset_name: str,
    batch_size: int,
    model_name_for_tokenizer: str,
    renderer_name: str,
    group_size: int,
    seed: int = 0,
    test_group_size: int = 1,
    test_envs: list[str] | None = None,
) -> RLDatasetBuilder:
    """
    Unified function to get any math dataset builder.
    Args:
        dataset_name: One of "math", "polaris", "deepmath", or "gsm8k"
        batch_size: Number of groups per batch
        model_name_for_tokenizer: Model name for tokenizer
        renderer_name: Name of the renderer to use
        group_size: Number of environments per group
        seed: Random seed for data shuffling (default: 0)
        test_group_size: Number of generations per test problem (default: 1)
        test_envs: List of test environment names (default: ["math500", "aime2024"])
    Returns:
        The appropriate dataset builder instance
    """
    builder_class = DATASET_BUILDER_MAP[dataset_name]

    # Build kwargs - only include test_envs if provided
    kwargs = {
        "batch_size": batch_size,
        "model_name_for_tokenizer": model_name_for_tokenizer,
        "renderer_name": renderer_name,
        "group_size": group_size,
        "seed": seed,
        "test_group_size": test_group_size,
    }

    if test_envs is not None:
        kwargs["test_envs"] = test_envs

    return builder_class(**kwargs)
