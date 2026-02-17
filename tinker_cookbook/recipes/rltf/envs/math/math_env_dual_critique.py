"""
Math environment with dual critique mechanism (expert + model self-critique).

This implements a dual-path training process:
1. Generate initial solution y1
2. Branch into two parallel paths:
   - Expert path: Get critique c_expert from judge → generate y2_expert
   - Model path: Get critique c_model from model → generate y2_model
3. Train:
   - y1: Train on r_y1 (or 0)
   - c_expert: DON'T train (but store for future SFT distillation)
   - c_model: Train on r_y2_model (critique quality measured by y2 outcome)
   - y2_expert: Train on r_y2_expert
   - y2_model: Train on r_y2_model

Key features:
- Dual paths after y1 for training both critique generation and refinement
- Expert critique stored for future SFT distillation
- Separate prompts for expert vs model critique generation
- Flag to disable expert path for test/eval (use_expert=False)
"""

import logging
from typing import Literal

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.recipes.rltf.envs.math.math_env import MathEnv
from tinker_cookbook.rl.types import (
    Action,
    Env,
    Observation,
    StepResult,
)

logger = logging.getLogger(__name__)


class MathEnvWithDualCritique(Env):
    """
    Math environment that trains model to generate its own critiques.

    Training mode (use_expert=True):
    - After y1, creates two paths: expert critique and model critique
    - Both paths share the same y1 tokens
    - Trains model critique based on y2 outcome

    Eval mode (use_expert=False):
    - Only uses model's self-critique path
    - Standard horizon=2 environment
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        judge_completer: TokenCompleter | None,  # For expert critique (can be None if use_expert=False)
        model_completer: TokenCompleter,  # For model self-critique
        judge_renderer: renderers.Renderer | None = None,  # For expert critique
        convo_prefix: list[renderers.Message] | None = None,
        grader: Literal["sympy", "math_verify"] = "sympy",
        timeout: float = 1.0,
        format_coef: float = 0.1,
        reward_y1: bool = False,
        horizon: int = 2,
        early_termination: bool = False,
        judge_type: str = "judge_hint",  # For expert critique
        model_judge_type: str = "judge_hint",  # For model self-critique (can differ from expert)
        use_expert: bool = True,  # If False, only use model critique (for eval)
    ):
        assert horizon == 2, "DualCritiqueEnv currently only supports horizon=2"

        if use_expert:
            assert judge_completer is not None, "judge_completer required when use_expert=True"
            assert judge_renderer is not None, "judge_renderer required when use_expert=True"

        self.problem = problem
        self.answer = answer
        self.renderer = renderer
        self.judge_completer = judge_completer
        self.model_completer = model_completer
        self.judge_renderer = judge_renderer
        self.convo_prefix = convo_prefix or []
        self.grader = grader
        self.timeout = timeout
        self.format_coef = format_coef
        self.reward_y1 = reward_y1
        self.horizon = horizon
        self.early_termination = early_termination
        self.judge_type = judge_type
        self.model_judge_type = model_judge_type
        self.use_expert = use_expert

        # State tracking
        self.stage = 1
        self.previous_response = None
        self.expert_critique = None
        self.model_critique = None

        # For branching: track which path we're on
        self.path = None  # Will be "expert" or "model" or None (before branching)

    def __deepcopy__(self, memo):
        """Custom deepcopy that avoids copying unpicklable async objects."""
        import copy
        # Create new instance with same immutable parameters but fresh state
        new_env = MathEnvWithDualCritique(
            problem=self.problem,
            answer=self.answer,
            renderer=self.renderer,
            judge_completer=self.judge_completer,
            model_completer=self.model_completer,
            judge_renderer=self.judge_renderer,
            convo_prefix=copy.deepcopy(self.convo_prefix),
            grader=self.grader,
            timeout=self.timeout,
            format_coef=self.format_coef,
            reward_y1=self.reward_y1,
            horizon=self.horizon,
            early_termination=self.early_termination,
            judge_type=self.judge_type,
            model_judge_type=self.model_judge_type,
            use_expert=self.use_expert,
        )
        return new_env

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    def get_question(self) -> str:
        """Get the initial question/problem text."""
        return self.problem

    def check_format(self, response_text: str) -> bool:
        """Check if response is in correct format (has \\boxed{})."""
        try:
            from tinker_cookbook.recipes.rltf.envs.math.math_grading import extract_boxed
            extract_boxed(response_text)
            return True
        except (ValueError, KeyError):
            return False

    def check_answer(self, response_text: str) -> bool:
        """Check if the answer is correct."""
        from tinker_cookbook.recipes.rltf.envs.math.math_env import safe_grade
        return safe_grade(
            response_text=response_text,
            ground_truth=self.answer,
            grader=self.grader,
            timeout=self.timeout,
        )

    def initial_observation(self) -> Observation:
        """Get the initial observation (the problem)."""
        initial_convo = self.convo_prefix + [
            {"role": "user", "content": self.problem}
        ]
        return self.renderer.build_generation_prompt(initial_convo)

    def _get_expert_critique_prompt(self, response_text: str) -> str:
        """Get prompt for expert judge to generate critique."""
        # Reuse logic from MathEnvWithCritique based on judge_type
        if self.judge_type == "judge_hint":
            return f"""You are an expert grader for math/logic problems.

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
        else:
            # Add other judge types as needed
            raise ValueError(f"Unsupported expert judge_type: {self.judge_type}")

    def _get_model_critique_prompt(self, response_text: str) -> str:
        """Get prompt for model to generate self-critique."""
        # Similar to expert but can be customized differently
        if self.model_judge_type == "judge_hint":
            return f"""You are an expert grader for math/logic problems.

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
            raise ValueError(f"Unsupported model judge_type: {self.model_judge_type}")

    async def _get_expert_critique(self, response_text: str) -> str:
        """Generate expert critique using judge completer."""
        critique_prompt = self._get_expert_critique_prompt(response_text)
        messages = [{"role": "user", "content": critique_prompt}]
        prompt_model_input = self.judge_renderer.build_generation_prompt(messages)

        critique_result = await self.judge_completer(
            prompt_model_input, self.judge_renderer.get_stop_sequences()
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

    async def _get_model_critique(self, response_text: str) -> tuple[str, Action]:
        """Generate model self-critique using model completer."""
        critique_prompt = self._get_model_critique_prompt(response_text)
        messages = [{"role": "user", "content": critique_prompt}]
        prompt_model_input = self.renderer.build_generation_prompt(messages)

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

        # Return both critique text and Action (for training)
        return critique, critique_result

    async def step(self, action: Action) -> StepResult:
        """Process action based on current stage and path."""
        message, parse_success = self.renderer.parse_response(action)

        if self.stage == 1:
            # First stage: store y1 and prepare for critique
            self.previous_response = message["content"]

            # Check correctness
            current_correct_format = float(parse_success) and float(self.check_format(self.previous_response))
            current_correct_answer = float(self.check_answer(self.previous_response))

            # Early termination check
            if self.early_termination and current_correct_answer:
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

            # Not correct or early termination disabled - continue to critique stage
            # This is where we need to branch for dual critique
            # For now, return a placeholder - we'll handle branching in the rollout logic
            raise NotImplementedError("Branching logic not yet implemented in step()")

        elif self.stage == 2:
            # Second stage: final answer y2
            final_response = message["content"]
            final_correct_format = float(parse_success) and float(self.check_format(final_response))
            final_correct_answer = float(self.check_answer(final_response))

            total_reward = self.format_coef * (final_correct_format - 1) + final_correct_answer

            return StepResult(
                reward=total_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "stage": self.stage,
                    "format": final_correct_format,
                    "correct": final_correct_answer,
                    f"y{self.stage}_format": final_correct_format,
                    f"y{self.stage}_correct": final_correct_answer,
                },
            )

        else:
            raise ValueError(f"Invalid stage: {self.stage}")
