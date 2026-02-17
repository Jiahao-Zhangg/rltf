import asyncio
import copy
import logging
from typing import Sequence

from tinker_cookbook.completers import TokenCompleter
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


@logtree.scope_header_decorator
async def do_single_rollout(policy: TokenCompleter, env: Env) -> Trajectory:
    transitions = []
    ob, stop_condition = await env.initial_observation()
    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        transition = Transition(
            ob=ob,
            ac=ac_with_logprobs,
            reward=step_result.reward,
            episode_done=step_result.episode_done,
            metrics=step_result.metrics,
        )
        transitions.append(transition)
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition
        if step_result.episode_done:
            break
    return Trajectory(transitions=transitions, final_ob=ob)


@logtree.scope_header_decorator
async def do_dual_critique_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter, envs_G: Sequence[Env]
) -> TrajectoryGroup:
    """
    Special rollout for dual critique mode.

    Processes group_size problem instances, each with two parallel paths:
    - Expert path: judge generates critique → model generates y2_expert
    - Model path: model generates critique → model generates y2_model

    Training targets (per instance):
    - y1: Train on r_y1
    - c_expert: DON'T train (marked with is_expert_critique=True)
    - c_model: Train on r_y2_model (marked with is_model_critique=True)
    - y2_expert: Train on r_y2_expert
    - y2_model: Train on r_y2_model

    Args:
        envs_G: List of environments. Length must be even (group_size * 2).
                Environments are paired: [expert0, model0, expert1, model1, ...]
    """
    assert len(envs_G) % 2 == 0, f"Dual critique mode requires even number of environments, got {len(envs_G)}"

    num_instances = len(envs_G) // 2
    logger.info(f"Dual critique rollout: {num_instances} problem instances, {len(envs_G)} total environments")

    # Process all instances in parallel
    all_trajectories = []

    async def process_instance(instance_idx: int):
        """Process one problem instance (expert + model paths)."""
        expert_env = envs_G[instance_idx * 2]
        model_env = envs_G[instance_idx * 2 + 1]

        # Both envs should have dual_critique_mode=True
        assert hasattr(expert_env, 'dual_critique_mode') and expert_env.dual_critique_mode
        assert hasattr(model_env, 'dual_critique_mode') and model_env.dual_critique_mode

        return await _do_single_dual_critique_rollout(expert_env, model_env, policy)

    # Run all instances in parallel
    instance_results = await asyncio.gather(*[process_instance(i) for i in range(num_instances)])

    # Flatten trajectories and collect rewards/metrics
    for traj_expert, traj_model in instance_results:
        all_trajectories.extend([traj_expert, traj_model])

    # Compute group rewards for all trajectories
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(all_trajectories, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Log group structure for debugging
    logger.info(f"Dual critique group structure: {num_instances} instances × 2 paths = {len(all_trajectories)} trajectories")

    # Compute actual trajectory rewards from transitions (group rewards might be placeholder 0s)
    actual_rewards = [sum(t.reward for t in traj.transitions) for traj in all_trajectories]
    logger.info(f"Trajectory rewards (from transitions): {[f'{r:.3f}' for r in actual_rewards]}")

    # Log path types and transition details
    for i, traj in enumerate(all_trajectories):
        path_type = "expert" if i % 2 == 0 else "model"
        instance_idx = i // 2
        transition_rewards = [t.reward for t in traj.transitions]
        total_reward = sum(transition_rewards)
        logger.info(f"  Traj {i} (instance {instance_idx}, {path_type}): {len(traj.transitions)} transitions, rewards={transition_rewards}, total={total_reward:.3f}")

    return TrajectoryGroup(all_trajectories, list(rewards_G), list(metrics_G))


async def _do_single_dual_critique_rollout(
    expert_env: Env, model_env: Env, policy: TokenCompleter
) -> tuple[Trajectory, Trajectory]:
    """
    Process a single problem instance with dual critique paths.
    Returns (expert_trajectory, model_trajectory).

    Supports any environment with dual_critique_mode (MathEnvWithCritique, KnightsAndKnavesEnvWithCritique, etc.)
    """
    # Check that both environments have dual critique mode enabled
    assert hasattr(expert_env, 'dual_critique_mode') and expert_env.dual_critique_mode
    assert hasattr(model_env, 'dual_critique_mode') and model_env.dual_critique_mode

    # Stage 1: Generate y1 (shared between both paths)
    ob_y1, stop_condition_y1 = await expert_env.initial_observation()
    y1_action = await policy(ob_y1, stop_condition_y1)

    # Parse y1 to get response text
    message, parse_success = expert_env.renderer.parse_response(y1_action.tokens)
    y1_response_text = message["content"]

    # Check y1 correctness
    y1_correct_format = float(parse_success) and float(expert_env.check_format(y1_response_text))
    y1_correct_answer = float(expert_env.check_answer(y1_response_text))

    # Early termination check
    if expert_env.early_termination and y1_correct_answer:
        # Both paths terminate early with same result
        total_reward = expert_env.format_coef * (y1_correct_format - 1) + y1_correct_answer

        # Create identical trajectories for both paths
        traj_expert = Trajectory(
            transitions=[Transition(
                ob=ob_y1,
                ac=y1_action,
                reward=total_reward,
                episode_done=True,
                metrics={
                    "stage": 1,
                    "format": y1_correct_format,
                    "correct": y1_correct_answer,
                    "early_termination": 1.0,
                    "y1_format": y1_correct_format,
                    "y1_correct": y1_correct_answer,
                    "path": "expert",
                },
            )],
            final_ob=ob_y1,  # No next ob since we terminated
        )

        traj_model = Trajectory(
            transitions=[Transition(
                ob=ob_y1,
                ac=y1_action,  # Same y1 action
                reward=total_reward,
                episode_done=True,
                metrics={
                    "stage": 1,
                    "format": y1_correct_format,
                    "correct": y1_correct_answer,
                    "early_termination": 1.0,
                    "y1_format": y1_correct_format,
                    "y1_correct": y1_correct_answer,
                    "path": "model",
                },
            )],
            final_ob=ob_y1,
        )

        return traj_expert, traj_model

    # y1 incorrect or early termination disabled - continue to critique and y2

    # Compute y1 reward
    if expert_env.reward_y1:
        y1_reward = expert_env.format_coef * (y1_correct_format - 1) + y1_correct_answer
    else:
        y1_reward = 0.0

    # Generate expert critique (judge)
    expert_critique_text = await expert_env._get_critique(y1_response_text, y1_correct_answer)
    logger.info(f"Expert critique: {expert_critique_text[:100]}...")

    # Generate model critique (model self-critique)
    model_critique_text, model_critique_action, model_critique_ob = await model_env._get_model_critique(y1_response_text, y1_correct_answer)
    logger.info(f"Model critique: {model_critique_text[:100]}...")

    # Build feedback prompts for y2 generation
    expert_feedback_prompt = f"""Question: {expert_env.get_question()}

You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.

Your Previous Solution:
{y1_response_text}

Expert Critique:
{expert_critique_text}

Instructions:
- Write your answer as a fresh solution to the original problem. Do not refer to your previous attempt.
- Do not mention or refer to the critique or the revision process.
- Use the critique only to improve correctness, clarity, and reasoning.
- Avoid using phrases like "Correctly applying the critique..." or "Reexamining my earlier solution...", etc., as the final answer should stand alone.

Let's think step by step and output the final answer within \\boxed{{}}."""

    model_feedback_prompt = f"""Question: {model_env.get_question()}

You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.

Your Previous Solution:
{y1_response_text}

Expert Critique:
{model_critique_text}

Instructions:
- Write your answer as a fresh solution to the original problem. Do not refer to your previous attempt.
- Do not mention or refer to the critique or the revision process.
- Use the critique only to improve correctness, clarity, and reasoning.
- Avoid using phrases like "Correctly applying the critique..." or "Reexamining my earlier solution...", etc., as the final answer should stand alone.

Let's think step by step and output the final answer within \\boxed{{}}."""

    # Build observations for y2
    expert_convo = expert_env.convo_prefix + [
        {"role": "user", "content": expert_feedback_prompt},
    ]
    expert_ob_y2 = expert_env.renderer.build_generation_prompt(expert_convo)

    model_convo = model_env.convo_prefix + [
        {"role": "user", "content": model_feedback_prompt},
    ]
    model_ob_y2 = model_env.renderer.build_generation_prompt(model_convo)

    # Generate y2 for both paths
    y2_expert_action = await policy(expert_ob_y2, expert_env.stop_condition)
    y2_model_action = await policy(model_ob_y2, model_env.stop_condition)

    # Parse and grade y2s
    y2_expert_message, y2_expert_parse = expert_env.renderer.parse_response(y2_expert_action.tokens)
    y2_expert_text = y2_expert_message["content"]
    y2_expert_correct_format = float(y2_expert_parse) and float(expert_env.check_format(y2_expert_text))
    y2_expert_correct_answer = float(expert_env.check_answer(y2_expert_text))
    y2_expert_reward = expert_env.format_coef * (y2_expert_correct_format - 1) + y2_expert_correct_answer

    y2_model_message, y2_model_parse = model_env.renderer.parse_response(y2_model_action.tokens)
    y2_model_text = y2_model_message["content"]
    y2_model_correct_format = float(y2_model_parse) and float(model_env.check_format(y2_model_text))
    y2_model_correct_answer = float(model_env.check_answer(y2_model_text))
    y2_model_reward = model_env.format_coef * (y2_model_correct_format - 1) + y2_model_correct_answer

    # Build expert path trajectory
    # Transition 0: y1 (shared)
    # Transition 1: expert critique (DON'T train) + y2_expert
    expert_transitions = [
        Transition(
            ob=ob_y1,
            ac=y1_action,
            reward=y1_reward,
            episode_done=False,
            metrics={
                "stage": 1,
                "format": y1_correct_format,
                "correct": y1_correct_answer,
                "y1_format": y1_correct_format,
                "y1_correct": y1_correct_answer,
                "path": "expert",
            },
        ),
        Transition(
            ob=expert_ob_y2,
            ac=y2_expert_action,
            reward=y2_expert_reward,
            episode_done=True,
            metrics={
                "stage": 2,
                "format": y2_expert_correct_format,
                "correct": y2_expert_correct_answer,
                "y2_format": y2_expert_correct_format,
                "y2_correct": y2_expert_correct_answer,
                "path": "expert",
                "is_expert_critique": True,  # Mark: don't train the critique part
                "expert_critique_text": expert_critique_text,  # Store for future SFT
            },
        ),
    ]

    # Build model path trajectory
    # Transition 0: y1 (shared - SAME tokens!)
    # Transition 1: model critique (train on r_y2_model!) + y2_model
    model_transitions = [
        Transition(
            ob=ob_y1,
            ac=y1_action,  # IMPORTANT: Same y1 action with same tokens!
            reward=y1_reward,
            episode_done=False,
            metrics={
                "stage": 1,
                "format": y1_correct_format,
                "correct": y1_correct_answer,
                "y1_format": y1_correct_format,
                "y1_correct": y1_correct_answer,
                "path": "model",
            },
        ),
        Transition(
            ob=model_ob_y2,
            ac=y2_model_action,
            reward=y2_model_reward,
            episode_done=True,
            metrics={
                "stage": 2,
                "format": y2_model_correct_format,
                "correct": y2_model_correct_answer,
                "y2_format": y2_model_correct_format,
                "y2_correct": y2_model_correct_answer,
                "path": "model",
                "is_model_critique": True,  # Mark: train critique with advantage=r_y2_model
                "model_critique_action": model_critique_action,  # Store the critique action for training
                "model_critique_ob": model_critique_ob,  # Store the critique observation for training
                "model_critique_text": model_critique_text,
                "expert_critique_text": expert_critique_text,  # Store expert for future SFT
            },
        ),
    ]

    traj_expert = Trajectory(transitions=expert_transitions, final_ob=expert_ob_y2)
    traj_model = Trajectory(transitions=model_transitions, final_ob=model_ob_y2)

    return traj_expert, traj_model


@logtree.scope_header_decorator
async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter
) -> TrajectoryGroup:
    # Check if any environments are configured for tree rollout
    envs_G: Sequence[Env] = await env_group_builder.make_envs()
    if envs_G and getattr(envs_G[0], 'is_tree_rollout', False):
        return await do_tree_rollout(env_group_builder, policy)

    # Check if any environments are configured for dual critique rollout
    if envs_G and getattr(envs_G[0], 'dual_critique_mode', False):
        return await do_dual_critique_rollout(env_group_builder, policy, envs_G)

    # Standard group rollout (envs already created above)
    trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G, envs_G)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)

    # Log trajectory tables with final rewards
    with logtree.scope_header("Trajectory Summary"):
        for i, (traj, final_reward) in enumerate(zip(trajectories_G, rewards_G, strict=True)):
            rows = []
            step_reward_sum = 0.0
            for t_idx, t in enumerate(traj.transitions):
                step_reward_sum += t.reward
                rows.append(
                    {
                        "step": t_idx,
                        "ob_len": t.ob.length,
                        "ac_len": len(t.ac.tokens),
                        "reward": f"{t.reward:.3f}",
                    }
                )
            # Add final row with final observation and computed reward
            rows.append(
                {
                    "step": "final",
                    "ob_len": traj.final_ob.length,
                    "ac_len": "-",
                    "reward": f"{final_reward:.3f}",
                }
            )
            # Add total reward row
            rows.append(
                {
                    "step": "total",
                    "ob_len": "-",
                    "ac_len": "-",
                    "reward": f"{step_reward_sum + final_reward:.3f}",
                }
            )
            logtree.table(rows, caption=f"Trajectory {i}")

    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))


async def do_tree_rollout(
    env_group_builder, policy: TokenCompleter
) -> TrajectoryGroup:
    """
    Perform true tree-structured rollout with recursive branching.
    
    At each branching stage >= group_from_stage:
    - If episode terminates: stop this branch
    - If episode continues: branch into num_grouped_responses copies
    
    Creates exponential growth: 1 → N → N² → N³ etc.
    
    Returns single TrajectoryGroup with all tree trajectories. 
    Observation-based grouping will be handled during advantage computation.
    """
    template_envs = await env_group_builder.make_envs()
    
    # For tree rollout, we'll process each environment independently
    # Each environment represents the same problem but will generate its own tree
    if len(template_envs) != 1:
        logger.info(f"Tree rollout with {len(template_envs)} environments (same problem, multiple trees)")
    
    group_from_stage = getattr(env_group_builder, 'group_from_stage', None)
    num_grouped_responses = getattr(env_group_builder, 'num_grouped_responses', 1)
    
    logger.info(f"Tree rollout: group_from_stage={group_from_stage}, num_grouped_responses={num_grouped_responses}")
    
    # Run tree rollout for each environment in parallel (same as standard rollout)
    env_trajectory_lists = await asyncio.gather(*[
        recursive_tree_rollout(policy, env, group_from_stage, num_grouped_responses, current_stage=1) 
        for env in template_envs
    ])
    
    # Flatten the list of lists
    final_trajectories = []
    for env_trajectories in env_trajectory_lists:
        final_trajectories.extend(env_trajectories)
    
    logger.info(f"Tree rollout completed: {len(final_trajectories)} final trajectories")
    
    # Compute group rewards (same as non-tree version)
    rewards_and_metrics_G = await env_group_builder.compute_group_rewards(final_trajectories)
    rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
    
    # Create TrajectoryGroup (same as non-tree version)
    trajectory_group = TrajectoryGroup(final_trajectories, list(rewards_G), list(metrics_G))
    
    # Add tree rollout metadata
    trajectory_group.is_tree_rollout = True
    trajectory_group.group_from_stage = group_from_stage
    trajectory_group.num_grouped_responses = num_grouped_responses
    
    return trajectory_group


async def recursive_tree_rollout(
    policy: TokenCompleter, 
    env: Env, 
    group_from_stage: int, 
    num_grouped_responses: int,
    current_stage: int = 1,
    env_thunk: callable = None
) -> list[Trajectory]:
    """
    Recursively perform tree rollout, branching at each stage >= group_from_stage.
    
    Returns list of all completed trajectories from this branch.
    """
    # Take one step
    if current_stage == 1:
        ob, stop_condition = await env.initial_observation()
    else:
        # Environment should already be in the right state for continuation
        ob = getattr(env, '_current_ob', None)
        stop_condition = env.stop_condition
        if ob is None:
            raise ValueError("Environment not properly set up for continuation")
    
    ac_with_logprobs = await policy(ob, stop_condition)
    step_result = await env.step(ac_with_logprobs.tokens)
    
    transition = Transition(
        ob=ob,
        ac=ac_with_logprobs,
        reward=step_result.reward,
        episode_done=step_result.episode_done,
        metrics=step_result.metrics,
    )
    
    # If episode is done, return single completed trajectory
    if step_result.episode_done:
        return [Trajectory(transitions=[transition], final_ob=step_result.next_observation)]
    
    # If we're at a branching stage, branch
    if current_stage >= group_from_stage:
        logger.debug(f"Branching at stage {current_stage} into {num_grouped_responses} copies")
        
        all_trajectories = []
        
        for i in range(num_grouped_responses):
            # Create a copy of the environment for this branch
            env_copy = copy.deepcopy(env)
            env_copy.group_env_id = i
            
            # Set up the environment state for continuation
            env_copy._current_ob = step_result.next_observation
            
            # Recursively continue this branch
            branch_trajectories = await recursive_tree_rollout(
                policy, env_copy, group_from_stage, num_grouped_responses, current_stage + 1
            )
            
            # Prepend our transition to each trajectory from this branch
            for traj in branch_trajectories:
                traj.transitions.insert(0, transition)
            
            all_trajectories.extend(branch_trajectories)
        
        return all_trajectories
    
    else:
        # Not at branching stage yet, continue linearly
        env._current_ob = step_result.next_observation
        
        remaining_trajectories = await recursive_tree_rollout(
            policy, env, group_from_stage, num_grouped_responses, current_stage + 1
        )
        
        # Prepend our transition to each trajectory
        for traj in remaining_trajectories:
            traj.transitions.insert(0, transition)
        
        return remaining_trajectories
