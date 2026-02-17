"""
Binary Matrix environment for RL training.

Based on the implementation from reasoning-gym:
https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/algorithmic/binary_matrix.py

A problem where given a binary matrix, you must find the Manhattan distance
to the nearest 0 for each cell.

Reference: https://leetcode.com/problems/01-matrix/description/
"""

import random
from collections import deque
from functools import partial
from typing import Sequence

import chz
import numpy as np
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


QUESTION_TEMPLATE = """Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

The output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.

Find the distance to the nearest 0 for each cell in the matrix below:
{matrix}"""


def generate_binary_matrix(rng: random.Random, n: int, p_zero: float = 0.25) -> list[list[int]]:
    """Generate a random binary matrix with at least one 0."""
    # Ensure at least one 0 in the matrix, so that a solution exists
    numbers = [0] + [0 if rng.random() < p_zero else 1 for _ in range(n**2 - 1)]
    rng.shuffle(numbers)
    matrix = [numbers[i * n : (i + 1) * n] for i in range(n)]
    return matrix


def compute_distances(matrix: list[list[int]]) -> list[list[int]]:
    """Get the distance to the nearest 0 for each cell in the matrix using BFS."""
    n = len(matrix)
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    visited = set()
    queue = deque()

    output = [[float("inf")] * n for _ in range(n)]

    # Start BFS from all 0 cells
    for r in range(n):
        for c in range(n):
            if matrix[r][c] == 0:
                output[r][c] = 0
                visited.add((r, c))
                queue.append((r, c))

    # BFS to compute distances
    clock = 1
    while True:
        temp = deque()
        while queue:
            r, c = queue.popleft()
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if (
                    0 <= new_r < n
                    and 0 <= new_c < n
                    and (new_r, new_c) not in visited
                    and matrix[new_r][new_c] == 1
                ):
                    output[new_r][new_c] = clock
                    visited.add((new_r, new_c))
                    temp.append((new_r, new_c))
        if temp:
            queue = temp
        else:
            break
        clock += 1

    return output


def matrix_to_str(matrix: list[list[int]]) -> str:
    """Convert matrix to string representation."""
    return "\n".join(" ".join(str(x) for x in row) for row in matrix)


class BinaryMatrixEnv(ProblemEnv):
    """Binary Matrix environment for RL training."""

    def __init__(
        self,
        matrix: list[list[int]],
        solution: list[list[int]],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.matrix = matrix
        self.solution = solution
        self.matrix_str = matrix_to_str(matrix)
        self.solution_str = matrix_to_str(solution)

    @property
    def problem(self) -> str:
        """Compatibility property for critique system."""
        return self.get_question()

    @property
    def answer(self) -> str:
        """Compatibility property for critique system."""
        return self.solution_str

    @classmethod
    def question_suffix(cls) -> str:
        return " Put your final answer in \\boxed{} format."

    def get_question(self) -> str:
        """Return the problem text."""
        question = QUESTION_TEMPLATE.format(matrix=self.matrix_str)
        return question + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if response has boxed answer."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer matches the correct solution matrix."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            answer_str = extract_boxed(sample_str)

            # Normalize whitespace for comparison
            answer_normalized = " ".join(answer_str.split())
            solution_normalized = " ".join(self.solution_str.split())

            # Try direct string comparison first
            if answer_normalized == solution_normalized:
                return True

            # Try parsing as Python list of lists
            try:
                parsed_answer = eval(answer_str)
                if isinstance(parsed_answer, list):
                    parsed_str = matrix_to_str(parsed_answer)
                    parsed_normalized = " ".join(parsed_str.split())
                    if parsed_normalized == solution_normalized:
                        return True
            except:
                pass

            return False

        except (ValueError, TypeError):
            return False

    def get_reference_answer(self) -> str:
        """Get the reference answer."""
        return self.solution_str

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": """Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell.

The output should be a matrix of the same size as the input matrix, where each cell contains the distance to the nearest 0.

Find the distance to the nearest 0 for each cell in the matrix below:
0 1 1
1 1 0
1 0 1
 Put your final answer in \\boxed{} format.""",
            },
            {
                "role": "assistant",
                "content": """Let me solve this step by step.

I need to find the Manhattan distance from each cell to the nearest 0.

Starting from the 0s in the matrix:
- Position (0,0) has value 0
- Position (1,2) has value 0
- Position (2,1) has value 0

For each cell, I'll compute the minimum Manhattan distance to any 0:

Row 0:
- (0,0): distance 0 (already a 0)
- (0,1): distance 1 (from (0,0) or (1,2))
- (0,2): distance 1 (from (1,2))

Row 1:
- (1,0): distance 1 (from (0,0) or (2,1))
- (1,1): distance 1 (from any of the 0s)
- (1,2): distance 0 (already a 0)

Row 2:
- (2,0): distance 1 (from (2,1))
- (2,1): distance 0 (already a 0)
- (2,2): distance 1 (from (1,2) or (2,1))

Therefore: \\boxed{0 1 1
1 1 0
1 0 1}""",
            },
        ]


class BinaryMatrixDataset(RLDataset):
    """Dataset of Binary Matrix problems for RL training."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        problems: list[dict],
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.problems = problems

    def __len__(self) -> int:
        """Return number of batches in the dataset."""
        import math
        return math.ceil(len(self.problems) / self.batch_size)

    def get_batch(self, batch_idx: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        batch_start = batch_idx * self.batch_size
        batch_end = min((batch_idx + 1) * self.batch_size, len(self.problems))

        builders = []
        for idx in range(batch_start, batch_end):
            problem = self.problems[idx]

            builder = ProblemGroupBuilder(
                env_thunk=partial(
                    BinaryMatrixEnv,
                    matrix=problem["matrix"],
                    solution=problem["solution"],
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name="binary_matrix",
            )
            builders.append(builder)

        return builders


@chz.chz
class BinaryMatrixDatasetBuilder(RLDatasetBuilder):
    """Builder for BinaryMatrixDataset following standard RLDatasetBuilder interface."""

    batch_size: int = 8
    group_size: int = 8
    model_name_for_tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"
    renderer_name: str | None = None
    seed: int = 0
    num_problems: int = 20000
    min_n: int = 3  # Minimum matrix size
    max_n: int = 5  # Maximum matrix size
    p_zero: float = 0.25  # Probability of a cell being 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        import pickle
        from pathlib import Path
        from tinker_cookbook import model_info

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        if self.renderer_name is None:
            renderer_name = model_info.get_recommended_renderer_name(self.model_name_for_tokenizer)
        else:
            renderer_name = self.renderer_name
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        # Use cache in home directory
        cache_dir = Path.home() / ".cache" / "tinker_cookbook" / "binary_matrix"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"problems_seed{self.seed}_n{self.num_problems}_size{self.min_n}-{self.max_n}_p{self.p_zero}.pkl"

        # Try to load from cache if available
        if cache_file.exists():
            logger.info(f"Loading Binary Matrix problems from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                train_problems = cached_data['train_problems']
                test_problems = cached_data['test_problems']
            logger.info(f"Loaded {len(train_problems)} train and {len(test_problems)} test problems from cache")
        else:
            # Generate problems from scratch
            logger.info(f"Generating {self.num_problems} Binary Matrix problems...")
            logger.info(f"Parameters: matrix_size={self.min_n}-{self.max_n}, p_zero={self.p_zero}")

            rng = random.Random(self.seed)
            all_problems = []

            for idx in range(self.num_problems):
                # Generate random matrix size
                n = rng.randint(self.min_n, self.max_n)

                # Generate matrix and compute solution
                matrix = generate_binary_matrix(rng, n, self.p_zero)
                solution = compute_distances(matrix)

                all_problems.append({
                    "matrix": matrix,
                    "solution": solution,
                    "n": n,
                })

                if (idx + 1) % 1000 == 0:
                    logger.info(f"  Generated {idx + 1}/{self.num_problems} problems")

            logger.info(f"Generated {len(all_problems)} problems")

            # Split into train and test (last 200 for test, rest for train)
            num_test = min(200, len(all_problems))
            test_problems = all_problems[-num_test:] if num_test > 0 else []
            train_problems = all_problems[:-num_test] if num_test > 0 else all_problems

            logger.info(f"Split into {len(train_problems)} train and {len(test_problems)} test problems")

            # Save to cache
            logger.info(f"Saving Binary Matrix problems to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'train_problems': train_problems,
                    'test_problems': test_problems,
                }, f)

        convo_prefix = None

        train_dataset = BinaryMatrixDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=train_problems,
            convo_prefix=convo_prefix,
        )

        test_dataset = BinaryMatrixDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=test_problems,
            convo_prefix=convo_prefix,
        )

        logger.info(f"Created Binary Matrix datasets: {len(train_problems)} train, {len(test_problems)} test")

        return train_dataset, test_dataset
