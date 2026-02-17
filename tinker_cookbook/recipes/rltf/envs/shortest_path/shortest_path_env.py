"""
Shortest path finding environment.

Based on the implementation from reasoning-gym:
https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/graphs/shortest_path.py

A pathfinding task where:
- Given a grid with obstacles (X), open cells (O), start (*), and destination (#)
- Find the shortest path from start to destination
- Use directional commands: right, down, left, up
- Return "infeasible" if no path exists
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


class ShortestPathProblemGenerator:
    """Generates shortest path problems on grids with configurable difficulty."""

    def __init__(
        self,
        rand_seed: int,
        min_rows: int = 5,
        max_rows: int = 8,
        min_cols: int = 5,
        max_cols: int = 8,
        p_blocked: float = 0.4,
    ):
        self.rng = np.random.default_rng(rand_seed)
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.min_cols = min_cols
        self.max_cols = max_cols
        self.p_blocked = p_blocked

    def _get_grid(self, rows: int, cols: int) -> tuple[list[list[str]], tuple[int, int], tuple[int, int]]:
        """Generate a random grid with obstacles, start, and destination."""
        # Create grid with obstacles
        grid = []
        for _ in range(rows):
            row = []
            for _ in range(cols):
                if self.rng.random() < self.p_blocked:
                    row.append('X')
                else:
                    row.append('O')
            grid.append(row)

        # Place start and destination
        start_r = int(self.rng.integers(0, rows))
        start_c = int(self.rng.integers(0, cols))
        grid[start_r][start_c] = '*'

        dest_r = int(self.rng.integers(0, rows))
        dest_c = int(self.rng.integers(0, cols))
        grid[dest_r][dest_c] = '#'

        return grid, (start_r, start_c), (dest_r, dest_c)

    def _get_answer(self, grid: list[list[str]], start: tuple[int, int], dest: tuple[int, int]) -> str | None:
        """Use BFS to find the shortest path."""
        rows = len(grid)
        cols = len(grid[0])

        # BFS
        queue = deque([(start, [])])
        visited = {start}

        directions = {
            'right': (0, 1),
            'down': (1, 0),
            'left': (0, -1),
            'up': (-1, 0),
        }

        while queue:
            (r, c), path = queue.popleft()

            if (r, c) == dest:
                return path

            for direction_name, (dr, dc) in directions.items():
                nr, nc = r + dr, c + dc

                if (0 <= nr < rows and 0 <= nc < cols and
                    (nr, nc) not in visited and
                    grid[nr][nc] in ['O', '#']):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [direction_name]))

        return None  # No path found

    def _is_valid_path(
        self,
        grid: list[list[str]],
        start: tuple[int, int],
        dest: tuple[int, int],
        path: list[str]
    ) -> bool:
        """Check if a path is valid (moves through open cells from start to dest)."""
        rows = len(grid)
        cols = len(grid[0])

        r, c = start

        directions = {
            'right': (0, 1),
            'down': (1, 0),
            'left': (0, -1),
            'up': (-1, 0),
        }

        for direction in path:
            if direction not in directions:
                return False

            dr, dc = directions[direction]
            r, c = r + dr, c + dc

            if not (0 <= r < rows and 0 <= c < cols):
                return False

            if grid[r][c] not in ['O', '#']:
                return False

        return (r, c) == dest

    def generate_problem(self) -> dict:
        """Generate a single shortest path problem."""
        rows = self.rng.integers(self.min_rows, self.max_rows + 1)
        cols = self.rng.integers(self.min_cols, self.max_cols + 1)

        grid, start, dest = self._get_grid(rows, cols)
        answer_path = self._get_answer(grid, start, dest)

        # Format grid as string
        grid_str = '\n'.join([' '.join(row) for row in grid])

        question = f"""Find the shortest path from the start (*) to the destination (#) in this grid.
You can only move through open cells (O) and cannot pass through obstacles (X).
Valid moves are: right, down, left, up.

Grid:
{grid_str}

If a path exists, return it as a sequence of moves (e.g., "right, down, right").
If no path exists, return "infeasible"."""

        if answer_path is None:
            answer = "infeasible"
        else:
            answer = ", ".join(answer_path)

        return {
            "question": question,
            "answer": answer,
            "grid": grid,
            "start": start,
            "dest": dest,
            "rows": rows,
            "cols": cols,
        }


def normalize_path_answer(answer_str: str) -> str | list[str]:
    """Normalize a path answer string."""
    answer_str = answer_str.lower().strip()

    # Check for infeasible
    if "infeasible" in answer_str or "impossible" in answer_str or "no path" in answer_str:
        return "infeasible"

    # Parse path
    # Remove common punctuation and split
    answer_str = answer_str.replace('.', ',').replace(';', ',').replace('and', ',')
    parts = [p.strip() for p in answer_str.split(',') if p.strip()]

    # Filter to valid directions
    valid_directions = {'right', 'down', 'left', 'up', 'r', 'd', 'l', 'u'}
    path = []
    for part in parts:
        # Try to extract direction
        words = part.split()
        for word in words:
            word_clean = word.strip('.,;()[]{}"\' ')
            if word_clean in valid_directions:
                # Normalize to full names
                direction_map = {
                    'right': 'right', 'r': 'right',
                    'down': 'down', 'd': 'down',
                    'left': 'left', 'l': 'left',
                    'up': 'up', 'u': 'up',
                }
                path.append(direction_map[word_clean])

    return path if path else "infeasible"


def check_shortest_path_answer(
    predicted_str: str,
    expected_str: str,
    grid: list[list[str]],
    start: tuple[int, int],
    dest: tuple[int, int],
) -> bool:
    """Check if the predicted path is correct."""
    try:
        predicted = normalize_path_answer(predicted_str)
        expected = normalize_path_answer(expected_str)

        # Both infeasible
        if predicted == "infeasible" and expected == "infeasible":
            return True

        # One is infeasible but not the other
        if predicted == "infeasible" or expected == "infeasible":
            return False

        # Check if predicted path is valid and reaches destination
        generator = ShortestPathProblemGenerator(rand_seed=0)
        if not generator._is_valid_path(grid, start, dest, predicted):
            return False

        # Accept if same length as expected (may be different shortest path)
        return len(predicted) == len(expected)

    except (ValueError, TypeError, AttributeError, KeyError):
        return False


class ShortestPathEnv(ProblemEnv):
    """Shortest path environment for RL training."""

    def __init__(
        self,
        question: str,
        answer: str,
        grid: list[list[str]],
        start: tuple[int, int],
        dest: tuple[int, int],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.question_text = question
        self.answer_text = answer
        self.grid = grid
        self.start = start
        self.dest = dest

    @property
    def problem(self) -> str:
        """Compatibility property for critique system."""
        return self.question_text

    @property
    def answer(self) -> str:
        """Compatibility property for critique system."""
        return self.answer_text

    @classmethod
    def question_suffix(cls) -> str:
        return " Put your final answer in \\boxed{} format."

    def get_question(self) -> str:
        """Return the problem text."""
        return self.question_text + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if response has boxed answer."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer correctly finds the shortest path."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            answer_str = extract_boxed(sample_str)

            return check_shortest_path_answer(
                answer_str,
                self.answer_text,
                self.grid,
                self.start,
                self.dest,
            )

        except (ValueError, TypeError):
            return False

    def get_reference_answer(self) -> str:
        """Get the reference answer."""
        return self.answer_text

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": """Find the shortest path from the start (*) to the destination (#) in this grid.
You can only move through open cells (O) and cannot pass through obstacles (X).
Valid moves are: right, down, left, up.

Grid:
O O X O
* O O O
X X O #

If a path exists, return it as a sequence of moves (e.g., "right, down, right").
If no path exists, return "infeasible". Put your final answer in \\boxed{} format.""",
            },
            {
                "role": "assistant",
                "content": """Let me find the shortest path step by step.

Starting position: (1, 0) - marked with *
Destination: (2, 3) - marked with #

I need to navigate from (1,0) to (2,3) avoiding X cells.

From (1,0):
- Can't go left (out of bounds)
- Can't go up (would go to row 0)
- Can't go down (row 2, col 0 is X)
- Can go right to (1,1)

From (1,1):
- Can go right to (1,2)

From (1,2):
- Can go right to (1,3)

From (1,3):
- Can go down to (2,3) - this is the destination!

Path: right, right, right, down

\\boxed{right, right, right, down}""",
            },
        ]


class ShortestPathDataset(RLDataset):
    """Dataset of shortest path problems for RL training."""

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
                    ShortestPathEnv,
                    question=problem["question"],
                    answer=problem["answer"],
                    grid=problem["grid"],
                    start=problem["start"],
                    dest=problem["dest"],
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name="shortest_path",
            )
            builders.append(builder)

        return builders


@chz.chz
class ShortestPathDatasetBuilder(RLDatasetBuilder):
    """Builder for ShortestPathDataset following standard RLDatasetBuilder interface."""

    batch_size: int = 8
    group_size: int = 8
    model_name_for_tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"
    renderer_name: str | None = None
    seed: int = 0
    num_problems: int = 20000
    min_rows: int = 5  # Minimum grid rows
    max_rows: int = 6  # Maximum grid rows
    min_cols: int = 5  # Minimum grid columns
    max_cols: int = 6  # Maximum grid columns
    p_blocked: float = 0.4  # Probability of a cell being blocked

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
        cache_dir = Path.home() / ".cache" / "tinker_cookbook" / "shortest_path"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"problems_seed{self.seed}_n{self.num_problems}_rows{self.min_rows}-{self.max_rows}_cols{self.min_cols}-{self.max_cols}_p{self.p_blocked}.pkl"

        # Try to load from cache if available
        if cache_file.exists():
            logger.info(f"Loading shortest path problems from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                train_problems = cached_data['train_problems']
                test_problems = cached_data['test_problems']
            logger.info(f"Loaded {len(train_problems)} train and {len(test_problems)} test problems from cache")
        else:
            # Generate problems from scratch
            logger.info(f"Generating {self.num_problems} shortest path problems...")
            logger.info(
                f"Parameters: rows={self.min_rows}-{self.max_rows}, "
                f"cols={self.min_cols}-{self.max_cols}, "
                f"p_blocked={self.p_blocked}"
            )

            rng = random.Random(self.seed)
            all_problems = []

            for i in range(self.num_problems):
                generator = ShortestPathProblemGenerator(
                    rand_seed=rng.randint(0, 2**31),
                    min_rows=self.min_rows,
                    max_rows=self.max_rows,
                    min_cols=self.min_cols,
                    max_cols=self.max_cols,
                    p_blocked=self.p_blocked,
                )
                problem = generator.generate_problem()
                all_problems.append(problem)

                if (i + 1) % 1000 == 0:
                    logger.info(f"  Generated {i + 1}/{self.num_problems} problems")

            logger.info(f"Generated {len(all_problems)} problems")

            # Split into train and test (last 200 for test, rest for train)
            num_test = min(200, len(all_problems))
            test_problems = all_problems[-num_test:] if num_test > 0 else []
            train_problems = all_problems[:-num_test] if num_test > 0 else all_problems

            logger.info(f"Split into {len(train_problems)} train and {len(test_problems)} test problems")

            # Save to cache
            logger.info(f"Saving shortest path problems to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'train_problems': train_problems,
                    'test_problems': test_problems,
                }, f)

        # convo_prefix = ShortestPathEnv.standard_fewshot_prefix()
        convo_prefix = None

        train_dataset = ShortestPathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=train_problems,
            convo_prefix=convo_prefix,
        )

        test_dataset = ShortestPathDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=test_problems,
            convo_prefix=convo_prefix,
        )

        logger.info(f"Created shortest path datasets: {len(train_problems)} train, {len(test_problems)} test")

        return train_dataset, test_dataset
