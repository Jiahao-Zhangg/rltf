"""
Math environment with critique feedback mechanism.

This implements a multi-stage generation process:
1. Generate initial solution y1
2. Get critique c from judge model (unless y1 is correct and early_termination=True)
3. Generate improved solution y2
4. Repeat steps 2-3 for additional turns if horizon > 2
5. Grade only the final solution for reward

Key features:
- Early termination: If early_termination=True (default), episodes end immediately when a correct solution is generated
- Tree structure support: Compatible with TreeStructureBuilder for dynamic branching of incorrect responses
- Backward compatibility: Setting early_termination=False restores original behavior
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TokenCompleter
from tinker_cookbook.recipes.rltf.envs.math.math_env import MathEnv, safe_grade
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


class MathEnvWithCritique(Env):
    """
    Math environment that uses critique feedback for refinement with optional early termination.

    The episode proceeds in multiple stages based on horizon:
    1. Generate y1 from prompt x
    2. If early_termination=True and y1 is correct: END (no critique, full reward)
    3. If y1 is incorrect: Generate critique and continue to y2
    4. Generate y2 from prompt (x, y1, critique)  
    5. If early_termination=True and y2 is correct: END (no critique, full reward)
    6. Continue for additional turns if horizon > 2

    With early_termination=False, behaves like the original implementation where
    only the final solution at horizon is graded for reward.
    
    Tree structure branching is handled by the rollout function, not the environment.
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        format_coef: float = 0.1,
        reward_y1: bool = False,  # If True, give intermediate stages actual reward
        horizon: int = 2,
        early_termination: bool = False,  # If True, end episode when generation is correct
        judge_type: str = "judge",
        # Dual critique mode parameters
        dual_critique_mode: bool = False,  # If True, train model to generate its own critiques (creates dual paths)
        model_completer: TokenCompleter | None = None,  # For model self-critique
        model_judge_type: str = "judge",  # Prompt type for model self-critique
        # Test mode parameter
        use_model_critique: bool = False,  # If True, use model's own critique instead of expert judge (for testing)
    ):
        if dual_critique_mode:
            assert model_completer is not None, "model_completer required when dual_critique_mode=True"

        if use_model_critique:
            assert model_completer is not None, "model_completer required when use_model_critique=True"

        self.problem = problem
        self.answer = answer
        self.renderer = renderer
        self.judge_completer = judge_completer
        self.judge_renderer = judge_renderer
        self.convo_prefix = convo_prefix or []
        self.grader = grader
        self.timeout = timeout
        self.format_coef = format_coef
        self.reward_y1 = reward_y1
        self.horizon = horizon
        self.early_termination = early_termination

        self.judge_type = judge_type
        self.dual_critique_mode = dual_critique_mode
        self.model_completer = model_completer
        self.model_judge_type = model_judge_type
        self.use_model_critique = use_model_critique

        # State tracking
        self.stage = 1
        self.previous_response = None
        self.critique = None  # Expert critique
        self.model_critique = None  # Model self-critique (for dual mode)

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying unpicklable async objects."""
        import copy
        # Create new instance with same immutable parameters but fresh state
        new_env = MathEnvWithCritique(
            problem=self.problem,
            answer=self.answer,
            renderer=self.renderer,  # Reuse (don't copy) - shared read-only object
            judge_completer=self.judge_completer,  # Reuse (don't copy) - shared async object
            judge_renderer=self.judge_renderer,  # Reuse (don't copy) - shared read-only object
            convo_prefix=copy.deepcopy(self.convo_prefix, memo),
            grader=self.grader,
            timeout=self.timeout,
            format_coef=self.format_coef,
            reward_y1=self.reward_y1,
            horizon=self.horizon,
            early_termination=self.early_termination,
            judge_type=self.judge_type,
            dual_critique_mode=self.dual_critique_mode,
            model_completer=self.model_completer,  # Reuse (don't copy) - shared async object
            model_judge_type=self.model_judge_type,
            use_model_critique=self.use_model_critique,
        )

        # Copy mutable state
        new_env.stage = self.stage
        new_env.previous_response = self.previous_response
        new_env.critique = self.critique
        new_env.model_critique = self.model_critique

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

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Initial prompt with the original question."""
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def _get_critique(self, response_text: str, current_correct_answer: bool) -> str:
        if self.judge_type == "correctness":
            # If correctness judge is enabled, use it to generate a critique
            if current_correct_answer:
                critique = "Your previous attempt was correct."
            else:
                critique = "Your previous attempt was incorrect."
            return critique
        
        else:
            critique_prompt = f"""You are an expert grader for math/logic problems.

Problem:
{self.get_question()}

Correct Final Answer (for evaluation only — do NOT reveal this to the student):
{self.answer}

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
        

        messages = [{"role": "user", "content": critique_prompt}]
        prompt_model_input = self.judge_renderer.build_generation_prompt(messages)

        # Use judge completer to generate critique - prompt is already a ModelInput
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

    async def _get_model_critique(self, response_text: str, current_correct_answer: float) -> tuple[str, Action, tinker.ModelInput]:
        """
        Generate model self-critique using model completer.

        Returns:
            tuple of (critique_text, critique_action, critique_ob) - Action and observation needed for training
        """
        if self.model_completer is None:
            raise RuntimeError(
                "model_completer is None but _get_model_critique() was called. "
                "This happens when dual_critique_mode=True but the training loop has not yet "
                "called dataset.update_sampling_client(sampling_client) to initialize the model completer. "
                "The training loop should call update_sampling_client() before the first batch."
            )

        critique_prompt = f"""You are an expert grader for math/logic problems.

Problem:
{self.get_question()}

Student Solution:
{response_text}

Your task:
- Analyze the student solution step by step.
- Focus on correctness and logical consistency.
- Identify potential mistake(s), if any.
- Provide concrete, actionable hints to improve the solution.
- Keep the Critique section under 200 words.

Format your response exactly as:

Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 200 words, ending with either "Your previous attempt was correct." or "Your previous attempt was incorrect."]"""

        messages = [{"role": "user", "content": critique_prompt}]
        prompt_model_input = self.renderer.build_generation_prompt(messages)

        # Use model completer to generate critique
        critique_result = await self.model_completer(
            prompt_model_input, self.renderer.get_stop_sequences()
        )

        # Decode and extract critique
        full_response = self.renderer.tokenizer.decode(critique_result.tokens)

        # Extract critique after "Critique:"
        if "Critique:" in full_response:
            critique = full_response.split("Critique:")[-1].strip()
        else:
            lines = full_response.strip().split('\n')
            critique = lines[-1] if lines else full_response

        # Remove common chat template end tokens
        critique = critique.replace("<|im_end|>", "").replace("</s>", "").replace("<|endoftext|>", "").strip()

        # Return critique text, Action, and observation (all needed for training)
        return critique, critique_result, prompt_model_input

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
            if self.use_model_critique:
                # Use model's own critique (for testing self-critique capability)
                critique_text, _, _ = await self._get_model_critique(self.previous_response, current_correct_answer)
                self.critique = critique_text
            else:
                # Use expert judge critique (default)
                self.critique = await self._get_critique(self.previous_response, current_correct_answer)

            feedback_prompt = f"""Question: {self.get_question()}

You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.

Your Previous Solution:
{self.previous_response}

Expert Critique:
{self.critique}

Instructions:
- Write a complete solution as if you are answering the Question for the first time. Do not refer to your previous attempt.
- Do NOT mention or refer to the critique or the revision process.
- Use the critique only to improve correctness, clarity, and reasoning.
- Do NOT use words like: critique, feedback, guide, previous, attempt, revise, correction, “correctly/incorrectly”, “as suggested”.

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
                    "early_termination": 0.0,
                },
            )

        elif self.stage == self.horizon:
            # Grade final solution
            final_text = message["content"]
            correct_format = float(parse_success) and float(self.check_format(final_text))
            correct_answer = float(self.check_answer(final_text))
            total_reward = self.format_coef * (correct_format - 1) + correct_answer

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "stage": self.horizon,
                    "format": correct_format,
                    "correct": correct_answer,
                    "early_termination": 0.0,
                    f"y{self.stage}_format": correct_format,
                    f"y{self.stage}_correct": correct_answer,
                },
            )

        else:
            raise ValueError(f"Invalid stage: {self.stage}")


@dataclass(frozen=True)
class MathCritiqueGroupBuilder(EnvGroupBuilder):
    """Builder for groups of MathEnvWithCritique environments."""

    env_thunk: Callable[[], MathEnvWithCritique]
    num_envs: int
    dataset_name: str = "math_critique"
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
        tags = [self.dataset_name, "critique"]
        if self.group_from_stage is not None:
            tags.extend(["tree", f"from_stage_{self.group_from_stage}"])
        return tags




class MathCritiqueDataset:
    """Dataset for math RL with critique feedback."""

    def __init__(
        self,
        base_dataset,  # MathDataset or similar
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        reward_y1: bool = False,  # If True, give intermediate stages actual reward
        horizon: int = 2,  # Number of generation stages
        # Grouped response parameters
        group_responses_from_stage: int | None = None,  # Start grouping from this stage onwards (None = no grouping)
        num_grouped_responses: int = 1,  # How many responses per group at each grouped stage
        early_termination: bool = True,  # If True, end episode when generation is correct
        judge_type: str = "judge",
        # Dual critique parameters
        dual_critique_mode: bool = False,  # If True, train model to generate its own critiques (creates dual paths for training)
        model_completer: TokenCompleter | None = None,  # For model self-critique (will be set via update_sampling_client)
        model_judge_type: str = "judge_hint",  # Prompt type for model self-critique
        # Test mode parameter
        use_model_critique: bool = False,  # If True, use model's own critique instead of expert judge (for testing)
    ):
        # Note: model_completer can be None initially - it will be set via update_sampling_client()
        # from the training loop before the first batch is sampled

        self.base_dataset = base_dataset
        self.judge_completer = judge_completer
        self.judge_renderer = judge_renderer
        self.reward_y1 = reward_y1
        self.horizon = horizon
        self.group_responses_from_stage = group_responses_from_stage
        self.num_grouped_responses = num_grouped_responses
        self.early_termination = early_termination
        self.judge_type = judge_type
        self.dual_critique_mode = dual_critique_mode
        self.model_completer = model_completer
        self.model_judge_type = model_judge_type
        self.use_model_critique = use_model_critique

        # For dynamic model_completer updates from training loop
        self._current_sampling_client = None
        self._renderer_for_model = None  # Will be set when we know the model's renderer

    def update_sampling_client(self, sampling_client):
        """Update the sampling client used for model self-critique (called by training loop)."""
        self._current_sampling_client = sampling_client

        # Create/update model_completer if in dual critique mode OR use_model_critique mode
        if (self.dual_critique_mode or self.use_model_critique) and sampling_client is not None:
            from tinker_cookbook.completers import TinkerTokenCompleter
            self.model_completer = TinkerTokenCompleter(
                sampling_client=sampling_client,
                max_tokens=8096,
            )
            mode_name = "dual critique" if self.dual_critique_mode else "model self-critique test"
            logger.info(f"Updated model_completer with latest sampling client for {mode_name}")

    def get_batch(self, index: int) -> list[EnvGroupBuilder]:
        """Get a batch of environment group builders with critique support."""
        # Get base environment builders
        base_builders = self.base_dataset.get_batch(index)

        # Convert to critique builders
        critique_builders = []
        for builder in base_builders:
            if hasattr(builder, 'env_thunk'):
                # Wrap the env_thunk to create MathEnvWithCritique
                original_thunk = builder.env_thunk

                def make_critique_env(orig_thunk=original_thunk):
                    # Call original thunk to get base env
                    base_env = orig_thunk()

                    # Create critique env with same parameters
                    # IMPORTANT: Use self.model_completer to dynamically look up the latest value
                    # (updated via update_sampling_client from training loop)
                    return MathEnvWithCritique(
                        problem=base_env.problem,
                        answer=base_env.answer,
                        renderer=base_env.renderer,
                        judge_completer=self.judge_completer,
                        judge_renderer=self.judge_renderer,
                        convo_prefix=getattr(base_env, 'convo_prefix', []),
                        grader=getattr(base_env, 'grader', 'sympy'),
                        timeout=getattr(base_env, 'timeout', 1.0),
                        reward_y1=self.reward_y1,
                        horizon=self.horizon,
                        early_termination=self.early_termination,
                        judge_type=self.judge_type,
                        # Dual critique parameters - use self.model_completer to get latest value
                        dual_critique_mode=self.dual_critique_mode,
                        model_completer=self.model_completer,  # This is looked up when make_critique_env is called
                        model_judge_type=self.model_judge_type,
                        # Test mode parameter
                        use_model_critique=self.use_model_critique,
                    )

                # For dual critique mode, create group_size * 2 environments (expert + model for each instance)
                # Otherwise use the builder's num_envs
                num_envs = builder.num_envs * 2 if self.dual_critique_mode else builder.num_envs

                # Use MathCritiqueGroupBuilder with optional tree parameters
                critique_builders.append(
                    MathCritiqueGroupBuilder(
                        env_thunk=make_critique_env,
                        num_envs=num_envs,
                        dataset_name=getattr(builder, 'dataset_name', 'math_critique'),
                        group_from_stage=self.group_responses_from_stage,
                        num_grouped_responses=self.num_grouped_responses,
                    )
                )

        return critique_builders

    def __len__(self) -> int:
        return len(self.base_dataset)
