"""Semantic-family-preserving counterfactual search for ETCE Phase 3A.1.

This module deliberately leaves the validated Phase 3A core unchanged.
It imports the frozen Phase 3A encoder/predictor/process-validity utilities and
adds one extra mutation rule:

    A_* -> A_*
    O_* -> O_*
    W_* -> W_*

The purpose is to strengthen process semantics without changing the black-box,
cohort, prefix protocol, DFG, timestamps, or search budget.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from .core import (
    CanonicalPredictor,
    DIRECT_PROXIES,
    TERMINAL_OFFER_ACTIVITIES,
    FORBIDDEN_MUTATION_TARGETS,
    flow_valid,
    process_support_score,
    sequence_edit_count,
    temporal_valid,
)


def activity_family(activity: str) -> str:
    """Return the BPIC2017 event family.

    BPIC2017 activities used here are organized as A_*, O_* and W_*.
    For defensive handling of an unexpected name, the prefix before the first
    underscore is used; if no underscore exists, the whole string is returned.
    """
    activity = str(activity)
    if "_" in activity:
        return activity.split("_", 1)[0]
    return activity


def family_preserved(factual: str, candidate: str) -> bool:
    return activity_family(factual) == activity_family(candidate)


def semantic_mutation_options(
    activities: Sequence[str],
    position: int,
    constraints: dict,
    max_options: int = 8,
):
    """Return locally valid replacements with the same event family."""
    factual = activities[position]

    if factual in TERMINAL_OFFER_ACTIVITIES or factual in DIRECT_PROXIES:
        return []

    edge_count = constraints["edge_count"]
    position_count = constraints["position_count"]
    mutable = constraints["mutable_activities"]

    prev_a = activities[position - 1] if position > 0 else None
    next_a = activities[position + 1] if position + 1 < len(activities) else None

    options = []
    for candidate in mutable:
        if candidate == factual:
            continue
        if candidate in FORBIDDEN_MUTATION_TARGETS:
            continue
        if not family_preserved(factual, candidate):
            continue
        if prev_a is not None and (prev_a, candidate) not in edge_count:
            continue
        if next_a is not None and (candidate, next_a) not in edge_count:
            continue

        pos_support = position_count.get((position + 1, candidate), 0)
        if pos_support <= 0:
            continue

        support = math.log1p(pos_support)
        if prev_a is not None:
            support += math.log1p(edge_count[(prev_a, candidate)])
        if next_a is not None:
            support += math.log1p(edge_count[(candidate, next_a)])

        options.append((support, candidate))

    options.sort(key=lambda x: (-x[0], x[1]))
    return [candidate for _, candidate in options[:max_options]]


def semantic_mutate_once(
    activities: Sequence[str],
    constraints: dict,
    rng: random.Random,
    max_options: int = 8,
):
    editable = []
    for pos in range(len(activities)):
        opts = semantic_mutation_options(
            activities,
            pos,
            constraints,
            max_options=max_options,
        )
        if opts:
            editable.append((pos, opts))

    if not editable:
        return None

    pos, options = rng.choice(editable)
    replacement = rng.choice(options)

    child = list(activities)
    child[pos] = replacement
    child = tuple(child)

    if not flow_valid(child, constraints["edge_count"]):
        return None

    return child


def semantic_crossover(
    parent_a: Sequence[str],
    parent_b: Sequence[str],
    factual: Sequence[str],
    constraints: dict,
    rng: random.Random,
):
    """Crossover is accepted only if every changed position remains in family."""
    if len(parent_a) < 3:
        return None

    cut = rng.randint(1, len(parent_a) - 1)
    child = tuple(parent_a[:cut]) + tuple(parent_b[cut:])

    for i, factual_activity in enumerate(factual):
        current = child[i]

        if factual_activity in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES:
            if current != factual_activity:
                return None
            continue

        if current != factual_activity:
            if current in FORBIDDEN_MUTATION_TARGETS:
                return None
            if not family_preserved(factual_activity, current):
                return None

    if not flow_valid(child, constraints["edge_count"]):
        return None

    return child


def _rank_key(
    probability: float,
    target_label: int,
    edit_count: int,
    support: float,
):
    target_score = probability if target_label == 1 else 1.0 - probability
    return (-target_score, edit_count, -support)


def semantic_evolutionary_counterfactual_search(
    predictor: CanonicalPredictor,
    factual_activities: Sequence[str],
    factual_timestamps: Sequence,
    constraints: dict,
    rng_seed: int,
    population_size: int = 40,
    generations: int = 20,
    mutation_rate: float = 0.70,
    crossover_rate: float = 0.50,
    max_options_per_position: int = 8,
):
    """Phase 3A.1 search with event-family preservation."""
    factual_activities = tuple(factual_activities)
    factual_timestamps = tuple(factual_timestamps)

    if len(factual_activities) != predictor.prefix:
        raise ValueError("Factual prefix length mismatch")
    if not temporal_valid(factual_timestamps):
        raise ValueError("Factual timestamps are not monotonic")

    factual_probability = float(
        predictor.predict_proba(
            [factual_activities],
            [factual_timestamps],
        )[0]
    )
    factual_pred = int(factual_probability >= 0.5)
    target_label = 1 - factual_pred

    rng = random.Random(int(rng_seed))
    edge_count = constraints["edge_count"]
    factual_flow = flow_valid(factual_activities, edge_count)

    seen = {factual_activities}
    population = []

    # Same deterministic single-edit initialization as Phase 3A, now family-preserving.
    for position in range(len(factual_activities)):
        for replacement in semantic_mutation_options(
            factual_activities,
            position,
            constraints,
            max_options=max_options_per_position,
        ):
            child = list(factual_activities)
            child[position] = replacement
            child = tuple(child)

            if child in seen:
                continue
            if not flow_valid(child, edge_count):
                continue

            seen.add(child)
            population.append(child)

    if factual_flow:
        population.append(factual_activities)

    population = population[: max(population_size * 4, population_size)]

    evaluated = 0
    all_successes = []

    for generation in range(generations + 1):
        if not population:
            break

        timestamps_batch = [factual_timestamps] * len(population)
        probs = predictor.predict_proba(population, timestamps_batch)
        evaluated += len(population)

        scored = []
        for seq, prob in zip(population, probs):
            edits = sequence_edit_count(factual_activities, seq)
            support = process_support_score(seq, edge_count)
            target_score = float(
                prob if target_label == 1 else 1.0 - prob
            )
            pred = int(prob >= 0.5)

            item = {
                "seq": seq,
                "prob": float(prob),
                "pred": pred,
                "edits": edits,
                "support": support,
                "target_score": target_score,
            }
            scored.append(item)

            if pred == target_label:
                all_successes.append(item)

        if all_successes:
            best = sorted(
                all_successes,
                key=lambda x: (
                    x["edits"],
                    -x["target_score"],
                    -x["support"],
                    x["seq"],
                ),
            )[0]

            changed_positions = [
                i
                for i, (old, new) in enumerate(
                    zip(factual_activities, best["seq"])
                )
                if old != new
            ]
            if not all(
                family_preserved(
                    factual_activities[i],
                    best["seq"][i],
                )
                for i in changed_positions
            ):
                raise AssertionError(
                    "Internal error: selected counterfactual violates family preservation"
                )

            return {
                "success": True,
                "factual_probability": factual_probability,
                "factual_prediction": factual_pred,
                "target_label": target_label,
                "factual_flow_valid": factual_flow,
                "counterfactual_activities": best["seq"],
                "counterfactual_probability": best["prob"],
                "counterfactual_prediction": best["pred"],
                "edit_count": best["edits"],
                "normalized_sequence_distance": (
                    best["edits"] / len(factual_activities)
                ),
                "temporal_distance": 0.0,
                "process_support": best["support"],
                "generation_found": generation,
                "candidates_evaluated": evaluated,
            }

        if generation == generations:
            break

        scored.sort(
            key=lambda x: _rank_key(
                x["prob"],
                target_label,
                x["edits"],
                x["support"],
            )
        )
        parents = [
            x["seq"]
            for x in scored[:population_size]
        ]

        if not parents:
            break

        children = []
        attempts = 0
        max_attempts = population_size * 30

        while (
            len(children) < population_size
            and attempts < max_attempts
        ):
            attempts += 1
            child = None

            if (
                len(parents) >= 2
                and rng.random() < crossover_rate
            ):
                pa, pb = rng.sample(parents, 2)
                child = semantic_crossover(
                    pa,
                    pb,
                    factual_activities,
                    constraints,
                    rng,
                )
            else:
                child = tuple(rng.choice(parents))

            if child is None:
                continue

            if rng.random() < mutation_rate:
                mutated = semantic_mutate_once(
                    child,
                    constraints,
                    rng,
                    max_options=max_options_per_position,
                )
                if mutated is not None:
                    child = mutated

            if child in seen:
                continue
            if not flow_valid(child, edge_count):
                continue

            valid = True
            for i, factual_activity in enumerate(factual_activities):
                current = child[i]

                if factual_activity in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES:
                    if current != factual_activity:
                        valid = False
                        break
                elif current != factual_activity:
                    if current in FORBIDDEN_MUTATION_TARGETS:
                        valid = False
                        break
                    if not family_preserved(factual_activity, current):
                        valid = False
                        break

            if not valid:
                continue

            seen.add(child)
            children.append(child)

        population = children

    return {
        "success": False,
        "factual_probability": factual_probability,
        "factual_prediction": factual_pred,
        "target_label": target_label,
        "factual_flow_valid": factual_flow,
        "counterfactual_activities": None,
        "counterfactual_probability": None,
        "counterfactual_prediction": None,
        "edit_count": None,
        "normalized_sequence_distance": None,
        "temporal_distance": 0.0,
        "process_support": None,
        "generation_found": None,
        "candidates_evaluated": evaluated,
    }
