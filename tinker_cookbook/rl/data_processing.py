"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import asyncio
import logging
from typing import List

import copy

import tinker
import torch
from tinker import TensorData
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.utils.misc_utils import all_same, safezip
from tinker_cookbook.supervised.common import datum_from_tokens_weights


logger = logging.getLogger(__name__)


def compute_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
    per_transition: bool = False,
    normalize_by_std: bool = False,
    use_traj_return: bool = False,
    gamma: float = 0.1,
) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups.

    Args:
        trajectory_groups_P: List of trajectory groups
        per_transition: If True, compute advantages per transition instead of per trajectory.
                       For multi-step episodes, this provides better credit assignment.
        normalize_by_std: If True, normalize advantages by standard deviation (mean 0, std 1).
                         Useful for rewards with large magnitude like log probabilities.
        use_traj_return: If True, use trajectory-level returns (final reward only) instead of
                        per-transition advantages. Useful for early termination with variable-length trajectories.
        gamma: Discount factor for computing returns (default: 0.5). Used when use_traj_return=True
              to compute discounted returns: R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    """
    if len(trajectory_groups_P) > 0:
        logger.info(f"compute_advantages: per_transition={per_transition}, use_traj_return={use_traj_return}")

    advantages_P: list[torch.Tensor] = []


    for i_group, traj_group in enumerate(trajectory_groups_P):
        # Find max number of transitions (for early termination, this varies)
        max_transitions = max(len(traj.transitions) for traj in traj_group.trajectories_G)

        # For each transition step, compute advantage across the group
        # Shape: [num_trajectories, max_transitions]
        # Pad shorter trajectories with NaN to exclude from mean calculation
        transition_returns_GT = []
        for traj in traj_group.trajectories_G:
            traj_transition_rewards = [trans.reward for trans in traj.transitions]
            # Compute discounted return: R_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
            traj_transition_returns = []
            for t in range(len(traj_transition_rewards)):
                discounted_return = sum(
                    (gamma ** (k - t)) * traj_transition_rewards[k]
                    for k in range(t, len(traj_transition_rewards))
                )
                traj_transition_returns.append(discounted_return)
            # Pad with NaN for missing transitions
            padded_returns = traj_transition_returns + [float('nan')] * (max_transitions - len(traj_transition_returns))
            transition_returns_GT.append(padded_returns)

        transition_returns_GT = torch.tensor(transition_returns_GT)  # [G, T]

        # Compute mean and center for each transition step (ignoring NaN values)
        transition_advantages_GT = torch.zeros_like(transition_returns_GT)
        for t in range(max_transitions):
            returns_t = transition_returns_GT[:, t]
            # Only compute mean over non-NaN values
            valid_mask = ~torch.isnan(returns_t)
            if valid_mask.any():
                mean_t = returns_t[valid_mask].mean()
                # Center the valid values
                transition_advantages_GT[:, t] = torch.where(
                    valid_mask,
                    returns_t - mean_t,
                    torch.tensor(0.0)  # Use 0 for padded positions (won't be used anyway)
                )

        # Normalize by std if requested (per transition step)
        if normalize_by_std:
            for t in range(max_transitions):
                adv_t = transition_advantages_GT[:, t]
                valid_mask = ~torch.isnan(transition_returns_GT[:, t])
                if valid_mask.any():
                    std = adv_t[valid_mask].std()
                    transition_advantages_GT[:, t] = torch.where(
                        valid_mask,
                        adv_t / (std + 1e-8),
                        torch.tensor(0.0)
                    )

        # Log first group for debugging
        if i_group == 0:
            logger.info(f"Per-transition advantages from return (group 0): shape={transition_advantages_GT.shape}, max_transitions={max_transitions}, normalize_by_std={normalize_by_std}")

            # Check if this is a dual critique group (even number of trajectories, likely paired)
            is_dual_critique = len(traj_group.trajectories_G) % 2 == 0 and any(
                t.metrics and t.metrics.get("is_model_critique", False)
                for traj in traj_group.trajectories_G
                for t in traj.transitions
            )

            for t in range(max_transitions):
                adv_t = transition_advantages_GT[:, t]
                returns_t = transition_returns_GT[:, t]
                valid_mask = ~torch.isnan(returns_t)
                if valid_mask.any():
                    valid_adv = adv_t[valid_mask]
                    valid_returns = returns_t[valid_mask]
                    mean_return = valid_returns.mean()
                    logger.info(f"  Transition {t} ({valid_mask.sum()}/{len(valid_mask)} trajectories):")
                    logger.info(f"    Returns: min={valid_returns.min():.3f}, max={valid_returns.max():.3f}, mean={mean_return:.3f}")
                    logger.info(f"    Advantages: min={valid_adv.min():.3f}, max={valid_adv.max():.3f}, mean={valid_adv.mean():.3f}, std={torch.std(valid_adv):.3f}")

                    if is_dual_critique:
                        # Show advantage for each trajectory at this transition
                        logger.info(f"    Per-trajectory advantages at t={t}:")
                        for i_traj in range(len(valid_mask)):
                            if valid_mask[i_traj]:
                                path_type = "expert" if i_traj % 2 == 0 else "model"
                                logger.info(f"      Traj {i_traj} ({path_type}): return={returns_t[i_traj]:.3f}, adv={adv_t[i_traj]:.3f}")

        advantages_P.append(transition_advantages_GT)

    return advantages_P

FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _to_input_targets(model_input: tinker.ModelInput) -> tuple[tinker.ModelInput, list[int]]:
    # TODO: make this work with multimodal data
    all_ints = model_input.to_ints()
    return tinker.ModelInput.from_ints(tokens=all_ints[:-1]), all_ints[1:]


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def trajectory_to_data(traj: Trajectory, traj_advantage: float | list[float], zero_out_first_transition: bool = False) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.

    Args:
        traj: The trajectory to convert
        traj_advantage: The advantage value(s) to assign to action tokens.
                       Can be a single float (trajectory-level) or list of floats (per-transition).
        zero_out_first_transition: If True, assign 0 advantage to first transition (useful for critique-based training)
    """
    # Convert traj_advantage to list if it's a single float
    if isinstance(traj_advantage, float):
        transition_advantages = [traj_advantage] * len(traj.transitions)
    else:
        transition_advantages = traj_advantage
        # For tree rollouts with early termination, trim advantages to match actual trajectory length
        if len(transition_advantages) > len(traj.transitions):
            transition_advantages = transition_advantages[:len(traj.transitions)]
        assert len(transition_advantages) == len(traj.transitions), \
            f"Number of advantages ({len(transition_advantages)}) must match number of transitions ({len(traj.transitions)})"

    class SequenceAccumulator:
        full_sequence: list[FlatObElem] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def make_datum_from_state():
        # TODO: generalize to multimodal
        all_tokens_T = _flat_ob_to_model_input(SequenceAccumulator.full_sequence)
        input_tokens_T, target_tokens_T = _to_input_targets(all_tokens_T)
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []

    for i_transition, transition in enumerate(traj.transitions):
        ob = transition.ob
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(SequenceAccumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        SequenceAccumulator.full_sequence.extend(delta_ob_flat)

        # Get advantage for this transition
        transition_advantage = transition_advantages[i_transition]

        if transition.metrics and "per_token_advantages" in transition.metrics:
            # Use per-token advantages (e.g., for rl_reweight_clip)
            per_token_advantages = transition.metrics["per_token_advantages"]

            SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
            SequenceAccumulator.sampled_logprobs.extend([0.0] * delta_ob_len + ac_with_logprobs.logprobs)
            SequenceAccumulator.advantages.extend([0] * delta_ob_len + per_token_advantages)
            SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))
        else:
            # Default behavior
            SequenceAccumulator.full_sequence.extend(ac_with_logprobs.tokens)
            SequenceAccumulator.sampled_logprobs.extend([0.0] * delta_ob_len + ac_with_logprobs.logprobs)
            SequenceAccumulator.advantages.extend([0] * delta_ob_len + [transition_advantage] * len(ac_with_logprobs.tokens))
            SequenceAccumulator.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

        # Dual critique mode: create separate training data for model critique
        if (transition.metrics and transition.metrics.get("is_model_critique", False)
            and "model_critique_action" in transition.metrics
            and "model_critique_ob" in transition.metrics):

            # Extract critique generation components
            critique_action = transition.metrics["model_critique_action"]
            critique_ob = transition.metrics["model_critique_ob"]

            # Build training data for critique generation: (critique_ob, critique_action)
            # Use the same advantage as the main transition (r_y2_model)
            critique_tokens = critique_ob.to_ints() + critique_action.tokens
            critique_input_tokens, critique_target_tokens = _to_input_targets(tinker.ModelInput.from_ints(tokens=critique_tokens))

            # Observation tokens get 0 advantage, critique action tokens get transition_advantage
            critique_advantages = [0.0] * critique_ob.length + [transition_advantage] * len(critique_action.tokens)
            critique_logprobs = [0.0] * critique_ob.length + critique_action.logprobs
            critique_mask = [0.0] * critique_ob.length + [1.0] * len(critique_action.tokens)

            # Shift for next-token prediction
            critique_advantages_shifted = critique_advantages[1:]
            critique_logprobs_shifted = critique_logprobs[1:]
            critique_mask_shifted = critique_mask[1:]

            assert (
                critique_input_tokens.length
                == len(critique_target_tokens)
                == len(critique_logprobs_shifted)
                == len(critique_advantages_shifted)
                == len(critique_mask_shifted)
            )

            critique_datum = tinker.Datum(
                model_input=critique_input_tokens,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(critique_target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(critique_logprobs_shifted)),
                    "advantages": TensorData.from_torch(torch.tensor(critique_advantages_shifted)),
                    "mask": TensorData.from_torch(torch.tensor(critique_mask_shifted)),
                },
            )
            data.append(critique_datum)

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def _split_feedback_prompt(text: str) -> tuple[str, str]:
    """
    Split a feedback prompt into question and answer parts for critique modeling.

    Args:
        text: The full feedback prompt text

    Returns:
        Tuple of (question, answer) where question is the critique generation prompt
        and answer is the critique text
    """
    start_marker = "Expert Critique:"
    end_marker = "Instructions:"

    if start_marker not in text:
        raise ValueError("The text does not contain 'Expert Critique:' marker.")

    original_instruction = "You are given your previous attempt and an expert critique of it below. Your task is to provide an improved solution based on the critique.\n\n"
    text_cleaned = text.replace(original_instruction, "")
    before, remainder = text_cleaned.split(start_marker, 1)
    new_instruction = "You are given a question and your previous attempt below. Your task is provide a critique of the attempt.\n\n"
    question = new_instruction + before + "Critique: "

    if end_marker in remainder:
        answer, _ = remainder.split(end_marker, 1)
    else:
        answer = remainder

    return question, answer.strip()


def _create_sft_distillation_data(traj: Trajectory) -> list[tinker.Datum]:
    """Create SFT distillation data: question -> final answer."""
    final_correct = traj.transitions[-1].metrics.get("correct", False)

    if len(traj.transitions) <= 1 or not final_correct:
        return []

    logger.debug("Distilling correct final answer for trajectory")
    question_tokens = traj.transitions[0].ob.to_ints()
    answer_tokens = traj.transitions[-1].ac.tokens

    all_tokens = question_tokens + answer_tokens
    weights = [0.0] * len(question_tokens) + [1.0] * len(answer_tokens)

    return [datum_from_tokens_weights(torch.tensor(all_tokens), torch.tensor(weights))]


def _create_feedback_modeling_data(traj: Trajectory, tokenizer, renderer) -> list[tinker.Datum]:
    """Create feedback modeling data: train critique generation from transitions."""
    if len(traj.transitions) < 2:
        logger.debug("Skipping feedback modeling for trajectory with < 2 transitions")
        return []

    distill_data = []

    for i in range(1, len(traj.transitions)):
        transition = traj.transitions[i]
        decoded_ob = tokenizer.decode(transition.ob.to_ints())

        # Handle dual critique mode (expert_critique_text in metadata)
        if transition.metrics and "expert_critique_text" in transition.metrics:
            is_expert_path = transition.metrics.get("is_expert_critique", False)
            if not is_expert_path:
                # Skip model path - it gets RL training on model critique
                continue

            expert_critique_text = transition.metrics["expert_critique_text"]
            start_marker = "Expert Critique:"

            if start_marker not in decoded_ob:
                logger.warning("Could not find 'Expert Critique:' marker in observation for dual critique mode")
                continue

            before_critique = decoded_ob.split(start_marker, 1)[0]
            original_instruction = "You are given your previous attempt and an expert critique of it below. Your task is to produce an improved solution using the critique.\n\n"
            before_critique = before_critique.replace(original_instruction, "")
            new_instruction = "You are given a question and your previous attempt below. Your task is provide a critique of the attempt.\n\n"
            question = new_instruction + before_critique + "Critique: "
            answer = expert_critique_text.strip()
        else:
            # Regular mode: parse critique from observation text
            question, answer = _split_feedback_prompt(decoded_ob)

        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        all_tokens, weights_tensor = renderer.build_supervised_example(conversation)
        distill_data.append(datum_from_tokens_weights(all_tokens, weights_tensor))

    if distill_data:
        logger.debug(f"Created {len(distill_data)} feedback modeling data from critique transitions")

    return distill_data


def create_distillation_data(
    traj: Trajectory,
    distillation_mode: str = "rl",
    model_name: str = "none",
) -> list[tinker.Datum]:
    """
    Create distillation training data from a trajectory.
    """
    if distillation_mode == "none":
        return []

    if distillation_mode == "sft":
        return _create_sft_distillation_data(traj)

    if distillation_mode == "feedback_modeling":
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook import renderers, model_info

        tokenizer = get_tokenizer(model_name)
        renderer_name = model_info.get_recommended_renderer_name(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
        return _create_feedback_modeling_data(traj, tokenizer, renderer)

    return []
        


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
    zero_out_first_transition: bool = False,
    distillation_mode: str = "none",
    sampling_client: tinker.SamplingClient | None = None,
    model_name: str = "none",
    use_first_turn_baseline: bool = False,
    rl_coef: float = 0.1,
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format.

    Args:
        trajectory_groups_P: List of trajectory groups
        advantages_P: List of advantage tensors
        zero_out_first_transition: If True, assign 0 advantage to first transition in multi-step trajectories
        distillation_mode: "rl" for RL distillation, "sft" for SFT distillation, "none" to disable
    """
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    # For modes that require batched sampling/logprobs, use unified batched function
    batched_modes = ["rl_reweight", "rl_reweight_mask", "rl_mask"]
    if distillation_mode in batched_modes and sampling_client is not None:
        return assemble_training_data_batched(
            trajectory_groups_P, advantages_P, zero_out_first_transition,
            distillation_mode, sampling_client, use_first_turn_baseline, rl_coef
        )

    # for sft style learning
    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        # Check if advantages_G is 2D (per-transition) or 1D (per-trajectory)
        is_per_transition = advantages_G.ndim == 2

        for i_traj, traj in enumerate(traj_group.trajectories_G):
            if is_per_transition:
                # Extract per-transition advantages for this trajectory
                traj_advantage = advantages_G[i_traj].tolist()  # [T]
            else:
                # Single advantage for whole trajectory
                traj_advantage = float(advantages_G[i_traj])

            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(traj, traj_advantage, zero_out_first_transition=zero_out_first_transition)
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

            if distillation_mode != "none":
                distill_data = create_distillation_data(traj, distillation_mode, model_name=model_name)
                if distill_data:
                    data_D.extend(distill_data)
                    metadata_D.extend([
                        dict(group_idx=i_group, traj_idx=i_traj, is_distillation=True) 
                        for _ in distill_data
                    ])

    # Log distillation statistics
    if distillation_mode != "none":
        regular_data = [d for d, m in zip(data_D, metadata_D) if not m.get("is_distillation", False)]
        distill_data = [d for d, m in zip(data_D, metadata_D) if m.get("is_distillation", False)]
        
        logger.info(f"=== Distillation Data Summary ===")
        logger.info(f"Regular training data: {len(regular_data)}")
        logger.info(f"Distillation data ({distillation_mode}): {len(distill_data)}")
        logger.info(f"Total training data: {len(data_D)}")
        logger.info(f"=== End Distillation Summary ===")

        # add number of data points to metadata
        for m in metadata_D:
            m["num_rl"] = len(regular_data)
            m["num_distill"] = len(distill_data)

    return data_D, metadata_D


def assemble_training_data_batched(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
    zero_out_first_transition: bool = False,
    distillation_mode: str = "on_policy",
    sampling_client: tinker.SamplingClient = None,
    use_first_turn_baseline: bool = False,
    rl_coef: float = 0.1,
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Unified batched version for all distillation modes that require sampling/logprobs."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    # First collect all trajectories and create regular training data
    all_trajectories = []
    traj_info = []  # (group_idx, traj_idx, traj_advantage)

    for i_group, (traj_group, advantages_G) in enumerate(safezip(trajectory_groups_P, advantages_P)):
        is_per_transition = advantages_G.ndim == 2

        for i_traj, traj in enumerate(traj_group.trajectories_G):
            if is_per_transition:
                traj_advantage = advantages_G[i_traj].tolist()
            else:
                traj_advantage = float(advantages_G[i_traj])

            # Regular training data
            new_data = trajectory_to_data(traj, traj_advantage, zero_out_first_transition=zero_out_first_transition)
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

            # Collect for batched distillation
            all_trajectories.append(traj)
            traj_info.append((i_group, i_traj, traj_advantage))


    if distillation_mode in ["rl_reweight", "rl_reweight_mask", "rl_mask"]:
        # Reweight modes: batch compute logprobs
        # Optionally compute first-turn baseline for each group: mean(r1)
        first_turn_baselines = {}
        if use_first_turn_baseline:
            from collections import defaultdict
            first_turn_rewards_by_group = defaultdict(list)
            for i, traj in enumerate(all_trajectories):
                if len(traj.transitions) > 0:
                    i_group = traj_info[i][0]
                    first_turn_rewards_by_group[i_group].append(traj.transitions[0].reward)

            first_turn_baselines = {
                group_idx: sum(rewards) / len(rewards)
                for group_idx, rewards in first_turn_rewards_by_group.items()
            }
            logger.info(f"Using first-turn baseline for distillation advantages. Computed baselines for {len(first_turn_baselines)} groups.")

        sequences = []
        sequence_info = []  # (question_length,)
        valid_indices = []

        for i, traj in enumerate(all_trajectories):
            if len(traj.transitions) > 1:
                original_question = traj.transitions[0].ob
                final_answer = traj.transitions[-1].ac
                question_tokens = original_question.to_ints()
                answer_tokens = final_answer.tokens
                full_sequence = question_tokens + answer_tokens
                sequences.append(full_sequence)
                sequence_info.append(len(question_tokens))
                valid_indices.append(i)

        if sequences:
            logger.info(f"Batch computing logprobs for {len(sequences)} {distillation_mode} sequences")
            model_inputs = [tinker.ModelInput.from_ints(tokens=seq) for seq in sequences]
            logprobs_futures = [sampling_client.compute_logprobs(mi) for mi in model_inputs]

            # Wait for all results
            all_logprobs = [future.result() for future in logprobs_futures]

            # Process results
            all_mask_fractions = []
            for seq_idx, logprobs_response in enumerate(all_logprobs):
                traj_idx = valid_indices[seq_idx]
                traj = all_trajectories[traj_idx]
                i_group, i_traj, traj_advantage = traj_info[traj_idx]

                # Extract components
                original_question = traj.transitions[0].ob
                final_answer = traj.transitions[-1].ac
                final_transition = traj.transitions[-1]
                question_length = sequence_info[seq_idx]

                # Extract logprobs for answer tokens only
                answer_logprobs = logprobs_response[question_length:]

                if distillation_mode == "rl_mask":
                    updated_final_answer = final_answer  # Use original logprobs
                else:
                    # Create updated answer with new logprobs
                    updated_final_answer = TokensWithLogprobs(
                        tokens=final_answer.tokens,
                        maybe_logprobs=answer_logprobs
                    )

                # Compute advantage
                if use_first_turn_baseline:
                    # Use r_final - mean(r_first_turn)
                    final_reward = final_transition.reward
                    first_turn_baseline = first_turn_baselines[i_group]
                    base_advantage = final_reward - first_turn_baseline
                    # Log first example for verification
                    if seq_idx == 0:
                        logger.info(f"First-turn baseline example: final_reward={final_reward:.4f}, baseline={first_turn_baseline:.4f}, advantage={base_advantage:.4f}")
                else:
                    # Use centered r2 advantage (old behavior)
                    if isinstance(traj_advantage, float):
                        base_advantage = traj_advantage
                    else:
                        base_advantage = traj_advantage[-1]

                # Handle clipping for clip modes
                if distillation_mode == "rl_reweight_mask":
                    # Clip based on absolute logprob threshold
                    logprob_threshold = -5.0
                    clip_mask = [1.0 if lp >= logprob_threshold else 0.0 for lp in answer_logprobs]
                    per_token_advantages = [base_advantage * rl_coef * mask for mask in clip_mask]
                    mask_fraction = 1.0 - sum(clip_mask) / len(clip_mask)
                    all_mask_fractions.append(mask_fraction)

                    synthetic_transition = Transition(
                        ob=original_question,
                        ac=updated_final_answer,
                        reward=final_transition.reward,
                        episode_done=True,
                        metrics={
                            "is_distillation": True,
                            "distillation_mode": distillation_mode,
                            "per_token_advantages": per_token_advantages,
                        }
                    )

                    distill_data = trajectory_to_data(
                        Trajectory(transitions=[synthetic_transition], final_ob=final_transition.ob),
                        base_advantage,
                        zero_out_first_transition=False
                    )

                    data_D.extend(distill_data)
                    metadata_D.extend([
                        dict(group_idx=i_group, traj_idx=i_traj, is_distillation=True, mask_fraction=mask_fraction)
                        for _ in distill_data
                    ])

                else:
                    # rl_reweight mode (no masking)
                    synthetic_transition = Transition(
                        ob=original_question,
                        ac=updated_final_answer,
                        reward=final_transition.reward,
                        episode_done=True,
                        metrics={"is_distillation": True, "distillation_mode": distillation_mode}
                    )

                    distill_data = trajectory_to_data(
                        Trajectory(transitions=[synthetic_transition], final_ob=final_transition.ob),
                        base_advantage,
                        zero_out_first_transition=False
                    )

                    data_D.extend(distill_data)
                    metadata_D.extend([
                        dict(group_idx=i_group, traj_idx=i_traj, is_distillation=True)
                        for _ in distill_data
                    ])

            if all_mask_fractions:
                avg_mask_fraction = sum(all_mask_fractions) / len(all_mask_fractions)
                logger.info(f"Average clip fraction: {avg_mask_fraction*100:.1f}%")

    # Log final statistics
    regular_data = [d for d, m in zip(data_D, metadata_D) if not m.get("is_distillation", False)]
    distill_data = [d for d, m in zip(data_D, metadata_D) if m.get("is_distillation", False)]

    logger.info(f"=== Batched {distillation_mode.upper()} Distillation Summary ===")
    logger.info(f"Regular training data: {len(regular_data)}")
    logger.info(f"Distillation data ({distillation_mode}): {len(distill_data)}")
    logger.info(f"Total training data: {len(data_D)}")

    for m in metadata_D:
        m["num_rl"] = len(regular_data)
        m["num_distill"] = len(distill_data)

    return data_D, metadata_D

def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups