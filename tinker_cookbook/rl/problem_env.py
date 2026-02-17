import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


class ProblemEnv(Env):
    def __init__(
        self,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @abstractmethod
    def get_question(self) -> str:
        pass

    @abstractmethod
    def check_answer(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def check_format(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        pass

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_format = float(parse_success) and float(self.check_format(message["content"]))
        correct_answer = float(self.check_answer(message["content"]))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, Correct: {'✓' if correct_answer else '✗'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "problems"
    # Tree structure parameters (optional)
    group_from_stage: int | None = None  # Start grouping from this stage onwards (None = no tree mode)
    num_grouped_responses: int = 1  # How many responses to generate at each grouped stage

    async def make_envs(self) -> Sequence[Env]:
        envs = []
        for i in range(self.num_envs):
            env = self.env_thunk()
            
            # Set tree mode metadata if tree parameters are provided
            if self.group_from_stage is not None:
                env.is_tree_rollout = True
                env.group_from_stage = self.group_from_stage
                env.num_grouped_responses = self.num_grouped_responses
                env.tree_group_id = i
            
            envs.append(env)
        
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        tags = [self.dataset_name]
        if self.group_from_stage is not None:
            tags.extend(["tree", f"from_stage_{self.group_from_stage}"])
        return tags
