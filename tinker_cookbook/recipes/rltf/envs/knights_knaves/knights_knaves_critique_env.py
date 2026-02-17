"""
Knights and Knaves environment with critique feedback mechanism.

This implements a multi-stage generation process for Knights and Knaves logic puzzles:
1. Generate initial solution y1
2. Get critique c from judge model (unless y1 is correct and early_termination=True)
3. Generate improved solution y2
4. Repeat steps 2-3 for additional turns if horizon > 2
5. Grade only the final solution for reward
"""

import logging
from dataclasses import dataclass
from typing import Sequence

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.recipes.rltf.envs.knights_knaves.knights_knaves_env import KnightsAndKnavesEnv
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


class KnightsAndKnavesEnvWithCritique(Env):
    """
    Knights and Knaves environment that uses critique feedback for refinement with optional early termination.

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
        quiz: str,
        solution_text: str,
        names: list[str],
        knight_knave: dict[str, str],
        renderer: renderers.Renderer,
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        reward_y1: bool = False,
        horizon: int = 2,
        early_termination: bool = False,
        judge_type: str = "judge",
        # Dual critique mode parameters
        dual_critique_mode: bool = False,
        model_completer: TokenCompleter | None = None,
        model_judge_type: str = "judge_hint",
        # Test mode parameter
        use_model_critique: bool = False,
    ):
        if dual_critique_mode:
            assert model_completer is not None, "model_completer required when dual_critique_mode=True"

        if use_model_critique:
            assert model_completer is not None, "model_completer required when use_model_critique=True"

        self.quiz = quiz
        self.solution_text = solution_text
        self.names = names
        self.knight_knave = knight_knave
        self.renderer = renderer
        self.judge_completer = judge_completer
        self.judge_renderer = judge_renderer
        self.convo_prefix = convo_prefix or []
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
        self.critique = None
        self.model_critique = None

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying unpicklable async objects."""
        import copy
        # Create new instance with same immutable parameters but fresh state
        new_env = KnightsAndKnavesEnvWithCritique(
            quiz=self.quiz,
            solution_text=self.solution_text,
            names=self.names,
            knight_knave=self.knight_knave,
            renderer=self.renderer,
            judge_completer=self.judge_completer,
            judge_renderer=self.judge_renderer,
            convo_prefix=copy.deepcopy(self.convo_prefix, memo),
            format_coef=self.format_coef,
            reward_y1=self.reward_y1,
            horizon=self.horizon,
            early_termination=self.early_termination,
            judge_type=self.judge_type,
            dual_critique_mode=self.dual_critique_mode,
            model_completer=self.model_completer,
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
        """Return the puzzle text."""
        return self.quiz + self.question_suffix()

    def check_format(self, sample_str: str) -> bool:
        """Check if the response has a boxed answer."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Check if answer correctly identifies all knights and knaves."""
        try:
            from tinker_cookbook.recipes.rltf.envs.knights_knaves.knights_knaves_env import normalize_answer
            answer_str = extract_boxed(sample_str)

            oracle_assignments = normalize_answer(self.solution_text)
            answer_assignments = normalize_answer(answer_str)

            return oracle_assignments == answer_assignments
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

        elif self.judge_type == "judge_with_correctness_only":
            """Generate critique using judge model - correctness only."""
            critique_prompt = f"""Problem: {self.get_question()}

Solution: {response_text}

Please analyze this solution step by step, then provide a one-sentence judgement on whether the solution is correct or incorrect. Only focus on the correctness of the final answer.
If the solution is correct, say "Your previous attempt was correct." If the solution is incorrect, say "Your previous attempt was incorrect." and nothing else. **Do not perform new calculations or reveal the correct solution.**
Format your response as:
Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 50 words]"""

        elif self.judge_type == "judge":
            critique_prompt = f"""Problem: {self.get_question()}
Solution: {response_text}

Please analyze this solution step by step, then provide a very short but detailed and actionable critique. Focus on the logical consistency of the reasoning and whether the assignments satisfy all the constraints. Keep the final critique under 50 words. **Do not reveal the correct solution.**

Format your response as:
Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 50 words]"""

        elif self.judge_type == "judge_full":
            critique_prompt = f"""Problem: {self.get_question()}
Solution: {response_text}

Please analyze this solution step by step, then provide a very short but detailed and actionable critique. Focus on:
1. Whether the logical reasoning is sound
2. Whether all statements are checked for consistency
3. Whether the proposed assignment makes knights tell truth and knaves lie
4. Any logical errors or contradictions

If the solution is correct, start with "Your previous attempt was correct." If the solution is incorrect, start with "Your previous attempt was incorrect." Keep the final critique under 80 words. **Do not reveal the correct solution.**

Format your response as:
Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 80 words]"""

        elif self.judge_type == "judge_with_ground_truth":
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
- Identify the main mistake(s), if any.
- Give a concrete, actionable hint based on their work so they can significantly improve.
- Do NOT reveal or directly state the correct solution or final answer in any part of your response.
- Keep the Critique section under 80 words.

Format your response exactly as:

Thinking: [Your step-by-step analysis]
Critique: [Your final critique in under 80 words, ending with either "Your previous attempt was correct." or "Your previous attempt was incorrect."]"""
        elif self.judge_type == "judge_hint":
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


        elif self.judge_type == "judge_hint_x":
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
- Give concrete, actionable hints based on their work so they can easily arrive at the correct answer based on your hints.
- Do NOT directly state the correct solution, but you can hint towards it if that helps the student solving the question.
- Keep the Critique section under 300 words.

Format your response exactly as:

Thinking: [Your step-by-step analysis]
Critique: [Your final critique, ending with either "Your previous attempt was correct." or "Your previous attempt was incorrect."]"""
            
        elif self.judge_type == "judge_guide":
            critique_prompt = f"""You are an expert reasoning coach. Your task is to write a standalone reasoning guide that helps a student solve the problem on a fresh attempt.

Problem:
{self.get_question()}

Reference Final Answer (for evaluation only — DO NOT reveal):
{self.solution_text}

Student Attempt:
{response_text}

Goal:
Write a guide that, when provided alongside ONLY the Problem (and without the Student Attempt), significantly improves the student’s chance of solving the problem correctly.

Your task:
- Do NOT provide any shortcut for the student in your guide, such as revealing the final answer.
- Do NOT refer to or quote the student’s attempt.
- The guide must be self-contained.
- Incorporate insights from the student’s mistakes into the guide, without explicitly mentioning them.
- Give a guide tailored to this problem.

Length:
- Keep the guide under 100 words.

Output format exactly:
Thinking: [Your step-by-step analysis on how to create the guide]
Guide: [Your guide for the student to solve the problem independently]
"""


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
        if self.judge_type == "judge_guide":
            # Extract guide after "Guide:"
            if "Guide:" in full_response:
                critique = full_response.split("Guide:")[-1].strip()
        else:
            lines = full_response.strip().split('\n')
            critique = lines[-1] if lines else full_response

        # Remove common chat template end tokens
        critique = critique.replace("<|im_end|>", "").replace("</s>", "").replace("<|endoftext|>", "").strip()

        return critique

    async def _get_model_critique(self, response_text: str, is_correct: bool) -> tuple[str, TokensWithLogprobs, tinker.ModelInput]:
        """
        Generate model self-critique using model completer.

        Returns:
            tuple of (critique_text, critique_result, critique_ob) - TokensWithLogprobs and observation needed for training
        """
        if self.model_completer is None:
            raise RuntimeError(
                "model_completer is None but _get_model_critique() was called. "
                "This happens when dual_critique_mode=True but the training loop has not yet "
                "called dataset.update_sampling_client(sampling_client) to initialize the model completer. "
                "The training loop should call update_sampling_client() before the first batch."
            )

        # Build prompt for model to critique itself (similar format to expert judge)
        if self.model_judge_type == "judge_hint":
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
        else:
            raise ValueError(f"Unsupported model_judge_type: {self.model_judge_type}")

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

        # Return critique text, TokensWithLogprobs, and observation (all needed for training)
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

            if self.judge_type != "judge_guide":
                # Build prompt for next generation
                feedback_prompt = f"""Question: {self.get_question()}

You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.

Your Previous Solution:
{self.previous_response}

Expert Critique:
{self.critique}

Instructions:
- Write your answer as a fresh solution to the original problem. Do not refer to your previous attempt.
- Do not mention or refer to the critique or the revision process.
- Use the critique only to improve correctness, clarity, and reasoning.
- Avoid using phrases like "Correctly applying the critique..." or "Reexamining my earlier solution...", etc., as the final answer should stand alone.

Let's think step by step and output the final answer within \\boxed{{}}."""
            else:
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
class KnightsAndKnavesCritiqueGroupBuilder(EnvGroupBuilder):
    """Builder for groups of KnightsAndKnavesEnvWithCritique for GRPO."""

    quiz: str
    solution_text: str
    names: list[str]
    knight_knave: dict[str, str]
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
        """Create group of environments with same puzzle."""
        return [
            KnightsAndKnavesEnvWithCritique(
                quiz=self.quiz,
                solution_text=self.solution_text,
                names=self.names,
                knight_knave=self.knight_knave,
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
        return ["knights_knaves_critique", "grpo"]


# Dataset wrapper for KnightsAndKnavesEnvWithCritique
class KnightsAndKnavesCritiqueDataset:
    """Wrapper dataset for Knights and Knaves RL with critique feedback."""

    def __init__(
        self,
        base_dataset,  # KnightsAndKnavesDataset
        judge_completer: TokenCompleter,
        judge_renderer: renderers.Renderer,
        reward_y1: bool = False,
        horizon: int = 2,
        group_responses_from_stage: int | None = None,
        num_grouped_responses: int = 1,
        early_termination: bool = False,
        judge_type: str = "judge",
        # Dual critique parameters
        dual_critique_mode: bool = False,
        model_completer: TokenCompleter | None = None,
        model_judge_type: str = "judge_hint",
        # Test mode parameter
        use_model_critique: bool = False,
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
        self.dual_critique_mode = dual_critique_mode
        self.model_completer = model_completer
        self.model_judge_type = model_judge_type
        self.use_model_critique = use_model_critique

        # For dynamic model_completer updates from training loop
        self._current_sampling_client = None

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

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders with critique support."""
        # Get base environment builders
        base_builders = self.base_dataset.get_batch(index)

        # Convert to critique builders
        critique_builders = []
        for builder in base_builders:
            if hasattr(builder, 'env_thunk'):
                # Wrap the env_thunk to create KnightsAndKnavesEnvWithCritique
                original_thunk = builder.env_thunk

                def make_critique_env(orig_thunk=original_thunk):
                    # Call original thunk to get base env
                    base_env = orig_thunk()

                    # Create critique env with same parameters
                    # IMPORTANT: Use self.model_completer to dynamically look up the latest value
                    # (updated via update_sampling_client from training loop)
                    return KnightsAndKnavesEnvWithCritique(
                        quiz=base_env.quiz,
                        solution_text=base_env.solution_text,
                        names=base_env.names,
                        knight_knave=base_env.knight_knave,
                        renderer=base_env.renderer,
                        judge_completer=self.judge_completer,
                        judge_renderer=self.judge_renderer,
                        convo_prefix=getattr(base_env, 'convo_prefix', []),
                        format_coef=getattr(base_env, 'format_coef', 0.1),
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

                # Create a simple builder wrapper
                from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
                critique_builders.append(
                    ProblemGroupBuilder(
                        env_thunk=make_critique_env,
                        num_envs=num_envs,
                        dataset_name="knights_knaves_critique",
                    )
                )

        return critique_builders

    def __len__(self) -> int:
        """Return number of batches in the dataset."""
        return len(self.base_dataset)
