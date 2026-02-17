"""
Knights and Knaves logic puzzle environment.

Based on the implementation from reasoning-gym:
https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/logic/knights_knaves.py

A logic puzzle where:
- Knights always tell the truth
- Knaves always lie
- Given statements from each person, determine who is a knight and who is a knave
- Each problem has exactly one valid solution
"""

import copy
import itertools
import random
from functools import partial
from typing import Sequence

import chz
import numpy as np
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

COMMON_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason",
    "Isabella", "William", "Mia", "James", "Charlotte", "Benjamin", "Amelia",
    "Lucas", "Harper", "Henry", "Evelyn", "Alexander", "Abigail", "Michael",
    "Emily", "Daniel", "Elizabeth", "Jacob", "Sofia", "Logan", "Avery",
    "Jackson", "Ella", "Sebastian", "Scarlett", "Jack", "Grace", "Aiden",
    "Chloe", "Owen", "Victoria", "Samuel", "Riley", "Matthew", "Aria",
    "Joseph", "Lily", "Luke", "Aurora", "David", "Zoey", "Oliver", "Penelope",
]

KNIGHT_KNAVE_PAIRS = [
    ["a knight", "a knave"],
    ["a pioneer", "a laggard"],
    ["a saint", "a sinner"],
    ["a hero", "a villain"],
    ["an angel", "a devil"],
    ["an altruist", "an egoist"],
    ["a sage", "a fool"],
]

VALID_ROLES = {pair[i].split()[1] for pair in KNIGHT_KNAVE_PAIRS for i in range(2)}

PREFIX = (
    "A very special island is inhabited only by {knight}s and {knave}s. "
    + "{Knight}s always tell the truth, and {knave}s always lie. "
)

POSTFIX = "So who is {a_knight} and who is {a_knave}?"

TEMPLATES = [
    "{name} said that {content}.",
    "{name} told you that {content}.",
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    'In {name}\'s words: "{content}".',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    "{name} expressed that {content}.",
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]


class KKProblemSampler:
    """Samples logical statements recursively with depth/width constraints."""

    def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
        self.rng = np.random.default_rng(rand_seed)
        self.n_people = n_people
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def sample(self):
        """Sample one statement per person."""
        statements = tuple(
            self._sample_statement(person_id, self.depth_constraint) for person_id in range(self.n_people)
        )
        return self._immutable_statements(statements)

    def sample_valid_problems(
        self,
        n_problems: int,
        max_retry: int = 1000,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True,
    ):
        """Sample valid problems with unique solutions."""
        problems = []
        unique_statements = set()
        for _ in range(n_problems):
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                sol = solutions[0] if len(solutions) > 0 else None
                problems.append({"statements": statements, "solution": sol, "all_solutions": solutions})
                unique_statements.add(statements)
                break
        return problems

    def _sample_statement(self, person_id: int, depth_constraint: int):
        """Recursively sample a logical statement."""
        dice = self.rng.integers(0, 6)
        if depth_constraint == 1 or dice == 0:
            # Primitive statement
            while True:
                knight_or_knave = self.rng.choice(["telling-truth", "lying"])
                person = self.rng.integers(0, self.n_people)
                # Prevent contradiction "I am lying"
                if not (knight_or_knave == "lying" and person == person_id):
                    return (knight_or_knave, person)
        if dice == 1:
            # Negation
            return ("not", self._sample_statement(person_id, depth_constraint - 1))
        if dice in [2, 3]:
            # AND/OR
            operator = ["and", "or"][dice - 2]
            n_substatements = self.rng.integers(2, self.width_constraint + 1)
            return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
        if dice in [4, 5]:
            # Implication/Biconditional
            operator = ["->", "<=>"][dice - 4]
            return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

    def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
        """Sample multiple substatements with optional deduplication."""
        sub_statements = []
        dedup_set = set()
        while True:
            stmt = self._sample_statement(person_id, depth_constraint - 1)
            if dedup:
                if stmt in dedup_set:
                    continue
                dedup_set.add(stmt)
            sub_statements.append(stmt)
            if len(sub_statements) == count:
                break
        return tuple(sub_statements)

    def _immutable_statements(self, mutable_statements):
        """Convert numpy types to Python types for immutability."""
        def _make_immutable(x):
            if isinstance(x, (list, tuple)):
                return tuple(_make_immutable(child) for child in x)
            if isinstance(x, np.str_):
                return str(x)
            if isinstance(x, np.int64):
                return int(x)
            return x

        return tuple(_make_immutable(s) for s in mutable_statements)


class KKProblemFormatter:
    """Formats logical statements into natural language puzzles."""

    def __init__(self, rand_seed, problem):
        self.rng = np.random.default_rng(rand_seed)
        self.problem = problem

    def format_problem(self):
        """Convert abstract statements to natural language puzzle."""
        statements = copy.deepcopy(self.problem["statements"])
        n_people = len(statements)
        names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
        knight_knave = self.rng.choice(KNIGHT_KNAVE_PAIRS)
        knight_knave = {
            "knight": knight_knave[0].split()[1],
            "knave": knight_knave[1].split()[1],
            "a_knight": knight_knave[0],
            "a_knave": knight_knave[1],
        }
        knight_knave["Knight"] = knight_knave["knight"].capitalize()
        knight_knave["Knave"] = knight_knave["knave"].capitalize()

        text = PREFIX.format(**knight_knave)
        text += f"You meet {n_people} inhabitants: "
        text += ", ".join(names[:-1]) + ", and " + names[-1] + "."

        text_statements = []
        for i, stmt in enumerate(statements):
            tmpl = self.rng.choice(TEMPLATES)
            content = self._format_statement(names, knight_knave, stmt)
            text_statements.append(" " + tmpl.format(name=names[i], content=content))

        text += "".join(text_statements)
        text += " " + POSTFIX.format(**knight_knave)

        # Format instruction
        format_ex = ", ".join(f"{name} is a {knight_knave['knight']}/{knight_knave['knave']}" for name in names[:-1])
        if len(names) > 1:
            format_ex += f", and {names[-1]} is a {knight_knave['knight']}/{knight_knave['knave']}"
        else:
            format_ex = f"{names[0]} is a {knight_knave['knight']}/{knight_knave['knave']}"

        text += f' (Format your answer like: "{format_ex}")'

        if self.problem["solution"] is None:
            solution_text = "No valid solution exists."
        else:
            solution_stmts = []
            for name, indicator in zip(names, self.problem["solution"]):
                if indicator:
                    solution_stmts.append(name + " is " + knight_knave["a_knight"])
                else:
                    solution_stmts.append(name + " is " + knight_knave["a_knave"])
            solution_text = ", ".join(solution_stmts[:-1]) + ", and " + solution_stmts[-1] + "."

        return {
            "quiz": text,
            "names": names,
            "knight_knave": knight_knave,
            "solution": self.problem["solution"],
            "solution_text": solution_text,
        }

    def _format_statement(self, names, knight_knave, statement, depth=0):
        """Recursively format a logical statement with appropriate parentheses."""
        # Base case: primitive statement
        if statement[0] in ("telling-truth", "lying"):
            return self._format_knight_knave(names, knight_knave, statement)

        # Negation
        if statement[0] == "not":
            # Special case: negate primitive by using complementary term
            if statement[1][0] in ("telling-truth", "lying"):
                complementary = (
                    "lying" if statement[1][0] == "telling-truth" else "telling-truth",
                    statement[1][1],
                )
                return self._format_knight_knave(names, knight_knave, complementary)
            else:
                # Complex negation
                inner_content = self._format_statement(names, knight_knave, statement[1], depth + 1)
                if statement[1][0] not in ("telling-truth", "lying"):
                    inner_content = f"({inner_content})"
                return f"it is not the case that {inner_content}"

        # AND/OR
        if statement[0] in ["and", "or"]:
            formatted_substmts = []
            for sub_stmt in statement[1:]:
                sub_content = self._format_statement(names, knight_knave, sub_stmt, depth + 1)
                # Add parentheses for complex subexpressions
                if sub_stmt[0] not in ("telling-truth", "lying"):
                    sub_content = f"({sub_content})"
                formatted_substmts.append(sub_content)
            connector = f" {statement[0]} "
            return connector.join(formatted_substmts)

        # Implication
        if statement[0] == "->":
            antecedent = self._format_statement(names, knight_knave, statement[1], depth + 1)
            consequent = self._format_statement(names, knight_knave, statement[2], depth + 1)

            if statement[1][0] not in ("telling-truth", "lying"):
                antecedent = f"({antecedent})"
            if statement[2][0] not in ("telling-truth", "lying"):
                consequent = f"({consequent})"

            return f"if {antecedent} then {consequent}"

        # Biconditional
        if statement[0] == "<=>":
            left = self._format_statement(names, knight_knave, statement[1], depth + 1)
            right = self._format_statement(names, knight_knave, statement[2], depth + 1)

            if statement[1][0] not in ("telling-truth", "lying"):
                left = f"({left})"
            if statement[2][0] not in ("telling-truth", "lying"):
                right = f"({right})"

            return f"{left} if and only if {right}"

        raise ValueError(f"Unknown statement type: {statement[0]}")

    def _format_knight_knave(self, names, knight_knave, statement):
        """Format primitive knight/knave statement."""
        assert statement[0] in ("telling-truth", "lying")
        text = names[statement[1]] + " is "
        text += {"telling-truth": knight_knave["a_knight"], "lying": knight_knave["a_knave"]}[statement[0]]
        return text


def find_solution(statements):
    """Find all valid solutions given statements."""
    n_people = len(statements)
    single_statement = ("and",) + tuple(
        ("<=>", ("telling-truth", i), statements[i]) for i in range(len(statements))
    )
    # Brute force all 2^n assignments
    solutions = []
    for assignments in itertools.product([True, False], repeat=n_people):
        if test_satisfiability(single_statement, assignments):
            solutions.append(assignments)
    return solutions


def test_satisfiability(statement, assignments):
    """Recursively test if a statement is satisfied under given assignments."""
    if statement[0] == "telling-truth":
        return assignments[statement[1]]
    if statement[0] == "lying":
        return not assignments[statement[1]]
    if statement[0] == "not":
        return not test_satisfiability(statement[1], assignments)
    if statement[0] == "and":
        return all(test_satisfiability(statement[i], assignments) for i in range(1, len(statement)))
    if statement[0] == "or":
        return any(test_satisfiability(statement[i], assignments) for i in range(1, len(statement)))
    if statement[0] == "->":
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (not val1) or val2
    if statement[0] == "<=>":
        val1 = test_satisfiability(statement[1], assignments)
        val2 = test_satisfiability(statement[2], assignments)
        return (val1 and val2) or ((not val1) and (not val2))
    raise KeyError(f"Unknown statement: {statement}")


def normalize_answer(answer: str) -> set[tuple[str, str]]:
    """Convert answer string into normalized set of (name, role) tuples."""
    answer = answer.lower().strip().replace(".", " ").replace(",", " ").replace(")", " ").replace("(", " ")
    parts = [p.strip() for p in answer.replace(" and ", " ").split()]

    assignments = set()
    current_name = None

    for part in parts:
        if part in ["is", "a", "an"]:
            continue
        if part in VALID_ROLES:
            if current_name:
                assignments.add((current_name, part))
                current_name = None
        else:
            current_name = part

    return assignments


class KnightsAndKnavesEnv(ProblemEnv):
    """Knights and Knaves logic puzzle environment for RL training."""

    def __init__(
        self,
        quiz: str,
        solution_text: str,
        names: list[str],
        knight_knave: dict[str, str],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        super().__init__(renderer, convo_prefix, format_coef)
        self.quiz = quiz
        self.solution_text = solution_text
        self.names = names
        self.knight_knave = knight_knave

    @property
    def problem(self) -> str:
        """Compatibility property for critique system."""
        return self.quiz

    @property
    def answer(self) -> str:
        """Compatibility property for critique system."""
        return self.solution_text

    @classmethod
    def question_suffix(cls) -> str:
        return " Put your final answer in \\boxed{} format."

    def get_question(self) -> str:
        """Return the puzzle text."""
        return self.quiz + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if response has boxed answer."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer correctly identifies all knights and knaves."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            answer_str = extract_boxed(sample_str)

            oracle_assignments = normalize_answer(self.solution_text)
            answer_assignments = normalize_answer(answer_str)

            # Full credit for exact match
            return oracle_assignments == answer_assignments

        except (ValueError, TypeError):
            return False

    def get_reference_answer(self) -> str:
        """Get the reference answer."""
        return self.solution_text

    @staticmethod
    def standard_fewshot_prefix() -> list[renderers.Message]:
        return [
            {
                "role": "user",
                "content": """A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Alice and Bob. Alice says, "Bob is a knave." Bob says, "Alice and I are both knaves." So who is a knight and who is a knave? Put your final answer in \\boxed{} format.""",
            },
            {
                "role": "assistant",
                "content": """Let me analyze this step by step.

Alice says: "Bob is a knave."
Bob says: "Alice and I are both knaves."

Case 1: Alice is a knight
- If Alice is a knight, her statement must be true
- So Bob is a knave
- Bob's statement "Alice and I are both knaves" must be false (since he's a liar)
- Indeed, this statement is false because Alice is a knight
- This is consistent! ✓

Case 2: Alice is a knave
- If Alice is a knave, her statement must be false
- So Bob is a knight
- Bob's statement "Alice and I are both knaves" must be true (since he tells truth)
- But this means Bob is a knave, contradicting our assumption
- This is inconsistent! ✗

Therefore: \\boxed{Alice is a knight, and Bob is a knave}""",
            },
        ]


class KnightsAndKnavesDataset(RLDataset):
    """Dataset of Knights and Knaves problems for RL training."""

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
            formatted = self.problems[idx]

            builder = ProblemGroupBuilder(
                env_thunk=partial(
                    KnightsAndKnavesEnv,
                    quiz=formatted["quiz"],
                    solution_text=formatted["solution_text"],
                    names=formatted["names"],
                    knight_knave=formatted["knight_knave"],
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=self.group_size,
                dataset_name="knights_knaves",
            )
            builders.append(builder)

        return builders


@chz.chz
class KnightsAndKnavesDatasetBuilder(RLDatasetBuilder):
    """Builder for KnightsAndKnavesDataset following standard RLDatasetBuilder interface."""

    batch_size: int = 8
    group_size: int = 8
    model_name_for_tokenizer: str = "meta-llama/Llama-3.3-70B-Instruct"
    renderer_name: str | None = None
    seed: int = 0
    num_problems: int = 20000
    n_people: int = 5  # Number of people (2-5)
    depth_constraint: int = 3  # Logical depth (1-3)
    width_constraint: int = 3  # Logical width (2-3)

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
        cache_dir = Path.home() / ".cache" / "tinker_cookbook" / "knights_knaves"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"problems_seed{self.seed}_n{self.num_problems}_people{self.n_people}_depth{self.depth_constraint}_width{self.width_constraint}.pkl"

        # Try to load from cache if available
        if cache_file.exists():
            logger.info(f"Loading Knights and Knaves problems from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                train_problems = cached_data['train_problems']
                test_problems = cached_data['test_problems']
            logger.info(f"Loaded {len(train_problems)} train and {len(test_problems)} test problems from cache")
        else:
            # Generate problems from scratch
            logger.info(f"Generating {self.num_problems} Knights and Knaves problems...")
            logger.info(f"Parameters: n_people={self.n_people}, depth={self.depth_constraint}, width={self.width_constraint}")

            rng = random.Random(self.seed)
            all_formatted_problems = []
            unique_statements = set()  # Track unique logical structures

            attempts = 0
            max_attempts = self.num_problems * 100  # Give plenty of attempts
            consecutive_failures = 0
            max_consecutive_failures = 1000  # Stop if we fail to find new problems 1000 times in a row

            while len(all_formatted_problems) < self.num_problems and attempts < max_attempts:
                attempts += 1

                # Sample a valid problem with unique logical structure
                sampler = KKProblemSampler(
                    rand_seed=rng.randint(0, 2**31),
                    n_people=self.n_people,
                    depth_constraint=self.depth_constraint,
                    width_constraint=self.width_constraint,
                )
                problems = sampler.sample_valid_problems(1, skip_no_solution=True, skip_multiple_solutions=True)

                if len(problems) == 0:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Stopping early: failed to generate valid problems {max_consecutive_failures} times in a row")
                        break
                    continue

                problem = problems[0]
                statements = problem["statements"]

                # Skip if we've seen this logical structure before
                if statements in unique_statements:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Stopping early: likely exhausted unique problem space (found {len(all_formatted_problems)} unique problems)")
                        break
                    continue

                # Successfully found a new unique problem
                consecutive_failures = 0
                unique_statements.add(statements)

                # Format the problem with random names/roles
                formatter = KKProblemFormatter(rand_seed=rng.randint(0, 2**31), problem=problem)
                formatted = formatter.format_problem()

                all_formatted_problems.append(formatted)

                if len(all_formatted_problems) % 100 == 0:
                    logger.info(f"  Generated {len(all_formatted_problems)}/{self.num_problems} problems (attempts: {attempts})")

            if len(all_formatted_problems) < self.num_problems:
                shortage = self.num_problems - len(all_formatted_problems)
                logger.warning(
                    f"Could only generate {len(all_formatted_problems)}/{self.num_problems} unique problems "
                    f"(short by {shortage}). Consider reducing num_problems or increasing depth/width/n_people constraints."
                )

                # Ensure we have at least some problems
                if len(all_formatted_problems) < 10:
                    raise ValueError(
                        f"Failed to generate sufficient unique problems (only got {len(all_formatted_problems)}). "
                        f"Try increasing depth_constraint (currently {self.depth_constraint}) or "
                        f"width_constraint (currently {self.width_constraint}) or "
                        f"n_people (currently {self.n_people}), or decrease num_problems."
                    )

            logger.info(f"Generated {len(all_formatted_problems)} problems with unique logical structures")

            # Split into train and test (last 100 for test, rest for train)
            num_test = min(200, len(all_formatted_problems))
            test_problems = all_formatted_problems[-num_test:] if num_test > 0 else []
            train_problems = all_formatted_problems[:-num_test] if num_test > 0 else all_formatted_problems

            logger.info(f"Split into {len(train_problems)} train and {len(test_problems)} test problems")
            logger.info("Test problems have logical structures unseen in training")

            # Save to cache
            logger.info(f"Saving Knights and Knaves problems to cache: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'train_problems': train_problems,
                    'test_problems': test_problems,
                }, f)

        # convo_prefix = KnightsAndKnavesEnv.standard_fewshot_prefix()
        convo_prefix = None

        train_dataset = KnightsAndKnavesDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=train_problems,
            convo_prefix=convo_prefix,
        )

        test_dataset = KnightsAndKnavesDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            problems=test_problems,
            convo_prefix=convo_prefix,
        )

        logger.info(f"Created Knights and Knaves datasets: {len(train_problems)} train, {len(test_problems)} test (all with unique logical structures)")

        return train_dataset, test_dataset
