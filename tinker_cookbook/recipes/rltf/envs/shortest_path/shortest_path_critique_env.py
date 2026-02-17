import logging
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TokenCompleter
from tinker_cookbook.recipes.rltf.envs.shortest_path.shortest_path_env import (
    ShortestPathEnv,
    check_shortest_path_answer,
)
from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)

logger = logging.getLogger(__name__)


class ShortestPathEnvWithCritique(Env):
    """
    Shortest path environment that uses critique feedback for refinement with optional early termination.

    The episode proceeds in multiple stages based on horizon:
    1. Generate y1 from prompt x
    2. If early_termination=True and y1 is correct: END (no critique, full reward)
    3. If y1 is incorrect: Generate critique and continue to y2
    4. Generate y2 from prompt (x, y1, critique)
    5. If early_termination=True and y2 is correct: END (no critique, full reward)
    6. Continue for additional turns if horizon > 2
    """

    def __init__(
        self,
        question: str,
        answer: str,
        grid: list[list[str]],
        start: tuple[int, int],
        dest: tuple[int, int],
        renderer: renderers.Renderer,
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        reward_y1: bool = False,
        horizon: int = 2,
        early_termination: bool = False,
        judge_type: str = "judge",
    ):
        self.question_text = question
        self.answer_text = answer
        self.solution_text = answer  # Alias for compatibility with judge prompts
        self.grid = grid
        self.start = start
        self.dest = dest
        self.renderer = renderer
        self.judge_completer = judge_completer
        self.judge_renderer = judge_renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef
        self.reward_y1 = reward_y1
        self.horizon = horizon
        self.early_termination = early_termination
        self.judge_type = judge_type

        # State tracking
        self.stage = 1
        self.previous_response = None
        self.critique = None

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying unpicklable async objects."""
        import copy
        # Create new instance with same immutable parameters but fresh state
        new_env = ShortestPathEnvWithCritique(
            question=self.question_text,
            answer=self.answer_text,
            grid=copy.deepcopy(self.grid, memo),
            start=self.start,
            dest=self.dest,
            renderer=self.renderer,
            judge_completer=self.judge_completer,
            judge_renderer=self.judge_renderer,
            convo_prefix=copy.deepcopy(self.convo_prefix, memo),
            format_coef=self.format_coef,
            reward_y1=self.reward_y1,
            horizon=self.horizon,
            early_termination=self.early_termination,
            judge_type=self.judge_type,
        )

        # Copy mutable state
        new_env.stage = self.stage
        new_env.previous_response = self.previous_response
        new_env.critique = self.critique

        # Copy any additional attributes that might have been set
        for attr_name in ['is_tree_rollout', 'group_from_stage', 'num_grouped_responses', 'tree_group_id', '_current_ob']:
            if hasattr(self, attr_name):
                setattr(new_env, attr_name, getattr(self, attr_name))

        return new_env

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @classmethod
    def question_suffix(cls) -> str:
        return " Let's think step by step and output the final answer within \\boxed{}."

    def get_question(self) -> str:
        """Return the problem text."""
        return self.question_text + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if the response has a boxed answer."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer correctly finds the shortest path."""
        try:
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

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Return the initial question."""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()}
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def _get_critique(self, response_text: str, is_correct: bool) -> str:
        """Generate critique for the current response."""
        if self.judge_type == "correctness":
            # Simple correctness feedback without judge model
            if is_correct:
                critique = "Your previous attempt was correct."
            else:
                critique = "Your previous attempt was incorrect."
            return critique

        elif self.judge_type == "judge":
            critique_prompt = f"""You are an expert grader for math/logic problems.

Problem:
{self.get_question()}

Correct Final Answer (for evaluation only — do NOT reveal this to the student):
{self.solution_text}

Student Solution:
{response_text}

Your task:
- Analyze the student solution step by step.
- Focus on correctness and logical consistency.
- Identify all the mistake(s), if any.
- Give one or several concrete, actionable hints based on their work so they can easily arrive at the correct answer based on your hints.
- Do NOT reveal or directly state the correct solution or final answer in any part of your response.
- Keep the Critique section under 200 words.

Format your response exactly as:

Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 200 words, ending with either "Your previous attempt was correct." or "Your previous attempt was incorrect."]"""


        else:
            raise ValueError(f"Invalid judge_type: {self.judge_type}")

        messages = [{"role": "user", "content": critique_prompt}]
        prompt_model_input = self.judge_renderer.build_generation_prompt(messages)

        # Use judge completer to generate critique
        ob = prompt_model_input
        critique_result = await self.judge_completer(
            ob, self.judge_renderer.get_stop_sequences()
        )

        # Decode and extract critique
        full_response = self.judge_renderer.tokenizer.decode(critique_result.tokens)

        # Extract critique after "Critique:"
        if "Critique:" in full_response:
            critique = full_response.split("Critique:")[-1].strip()
        else:
            lines = full_response.strip().split('\n')
            critique = lines[-1] if lines else full_response

        # Remove common chat template end tokens
        critique = critique.replace("<|im_end|>", "").replace("</s>", "").replace("<|endoftext|>", "").strip()

        return critique

    async def step(self, action: Action) -> StepResult:
        """Process action based on current stage."""
        message, parse_success = self.renderer.parse_response(action)

        if self.stage != self.horizon:
            # Store current response
            self.previous_response = message["content"]

            # Check if current response is correct
            current_correct_format = float(parse_success) and float(self.check_format(self.previous_response))
            current_correct_answer = float(self.check_answer(self.previous_response))

            # If early termination is enabled and the generation is correct, end the episode immediately
            if self.early_termination and current_correct_answer:
                # Compute final reward
                total_reward = self.format_coef * (current_correct_format - 1) + current_correct_answer

                return StepResult(
                    reward=total_reward,
                    episode_done=True,
                    next_observation=tinker.ModelInput.empty(),
                    next_stop_condition=self.stop_condition,
                    metrics={
                        "stage": self.stage,
                        "format": current_correct_format,
                        "correct": current_correct_answer,
                        "early_termination": 1.0,
                        f"y{self.stage}_format": current_correct_format,
                        f"y{self.stage}_correct": current_correct_answer,
                    },
                )

            # If not correct, generate critique and continue
            self.critique = await self._get_critique(self.previous_response, current_correct_answer)

            feedback_prompt = f"""Question: {self.get_question()}

You are given an expert reasoning guide to help you solve the problem. Use it to produce a correct, complete solution.

Expert Guide:
{self.critique}

Instructions:
- Write your answer as a fresh solution to the original problem, using the guide as support.
- Do not mention the guide, feedback, or any revision process.
Let's think step by step and output the final answer within \\boxed{{}}."""

            convo = self.convo_prefix + [
                {"role": "user", "content": feedback_prompt},
            ]
            next_ob = self.renderer.build_generation_prompt(convo)

            self.stage += 1

            # Compute reward for intermediate generation if enabled
            if self.reward_y1:
                prev_reward = self.format_coef * (current_correct_format - 1) + current_correct_answer
            else:
                prev_reward = 0.0

            return StepResult(
                reward=prev_reward,
                episode_done=False,
                next_observation=next_ob,
                next_stop_condition=self.stop_condition,
                metrics={
                    "stage": self.stage - 1,
                    "has_critique": 1 if self.critique else 0,
                    f"y{self.stage - 1}_format": current_correct_format,
                    f"y{self.stage - 1}_correct": current_correct_answer,
                },
            )
        else:
            # Final stage: compute reward based on final answer
            final_response = message["content"]
            final_correct_format = float(parse_success) and float(self.check_format(final_response))
            final_correct_answer = float(self.check_answer(final_response))

            total_reward = self.format_coef * (final_correct_format - 1) + final_correct_answer

            # Collect metrics from all stages
            metrics = {
                "stage": self.stage,
                "format": final_correct_format,
                "correct": final_correct_answer,
                "early_termination": 0.0,
                f"y{self.stage}_format": final_correct_format,
                f"y{self.stage}_correct": final_correct_answer,
            }

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=metrics,
            )


@dataclass(frozen=True)
class ShortestPathCritiqueGroupBuilder(EnvGroupBuilder):
    """Builder for groups of ShortestPathEnvWithCritique for GRPO."""

    question: str
    answer: str
    grid: list[list[str]]
    start: tuple[int, int]
    dest: tuple[int, int]
    renderer: renderers.Renderer
    judge_completer: TokenCompleter
    judge_renderer: renderers.Renderer
    num_envs: int
    convo_prefix: list[renderers.Message] | None = None
    format_coef: float = 0.1
    reward_y1: bool = False
    horizon: int = 2
    early_termination: bool = False
    judge_type: str = "judge"

    async def make_envs(self) -> Sequence[Env]:
        """Create group of environments with same problem."""
        return [
            ShortestPathEnvWithCritique(
                question=self.question,
                answer=self.answer,
                grid=self.grid,
                start=self.start,
                dest=self.dest,
                renderer=self.renderer,
                judge_completer=self.judge_completer,
                judge_renderer=self.judge_renderer,
                convo_prefix=self.convo_prefix,
                format_coef=self.format_coef,
                reward_y1=self.reward_y1,
                horizon=self.horizon,
                early_termination=self.early_termination,
                judge_type=self.judge_type,
            )
            for _ in range(self.num_envs)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        """No additional group-level rewards needed."""
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return ["shortest_path_critique", "grpo"]


# Dataset wrapper for ShortestPathEnvWithCritique
class ShortestPathCritiqueDataset:
    """Wrapper dataset for shortest path RL with critique feedback."""

    def __init__(
        self,
        base_dataset,  # ShortestPathDataset
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        reward_y1: bool = False,
        horizon: int = 2,
        group_responses_from_stage: int | None = None,
        num_grouped_responses: int = 1,
        early_termination: bool = False,
        judge_type: str = "judge",
    ):
        self.base_dataset = base_dataset
        self.judge_completer = judge_completer
        self.judge_renderer = judge_renderer
        self.reward_y1 = reward_y1
        self.horizon = horizon
        self.group_responses_from_stage = group_responses_from_stage
        self.num_grouped_responses = num_grouped_responses
        self.early_termination = early_termination
        self.judge_type = judge_type

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders with critique support."""
        # Get base environment builders
        base_builders = self.base_dataset.get_batch(index)

        # Convert to critique builders
        critique_builders = []
        for builder in base_builders:
            if hasattr(builder, 'env_thunk'):
                # Wrap the env_thunk to create ShortestPathEnvWithCritique
                original_thunk = builder.env_thunk

                def make_critique_env(orig_thunk=original_thunk):
                    # Call original thunk to get base env
                    base_env = orig_thunk()

                    # Create critique env with same parameters
                    return ShortestPathEnvWithCritique(
                        question=base_env.question_text,
                        answer=base_env.answer_text,
                        grid=base_env.grid,
                        start=base_env.start,
                        dest=base_env.dest,
                        renderer=base_env.renderer,
                        judge_completer=self.judge_completer,
                        judge_renderer=self.judge_renderer,
                        convo_prefix=getattr(base_env, 'convo_prefix', []),
                        format_coef=getattr(base_env, 'format_coef', 0.1),
                        reward_y1=self.reward_y1,
                        horizon=self.horizon,
                        early_termination=self.early_termination,
                        judge_type=self.judge_type,
                    )

                # Create a simple builder wrapper
                from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
                critique_builders.append(
                    ProblemGroupBuilder(
                        env_thunk=make_critique_env,
                        num_envs=builder.num_envs,
                        dataset_name="shortest_path_critique",
                    )
                )

        return critique_builders

    def __len__(self) -> int:
        """Return number of batches in the dataset."""
        return len(self.base_dataset)
