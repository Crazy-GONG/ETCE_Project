"""Stratified local fit/eval protocol for ETCE Phase 4A.1 / Phase 4B-v2.

This module keeps the Phase 4A neighborhoods, surrogate definitions, hyperparameters,
and deterministic class-stratified 70/30 held-out evaluation protocol unchanged.
Phase 4B-v2 additionally propagates each branch's real multi-rule CF candidate list.

The local fit/eval partition remains:

- split deterministically within each black-box class using sequence hash;
- keep the factual sequence in fit;
- if a class has >=2 local samples, reserve at least one for eval;
- if a class has only one local sample, keep it in fit so the surrogate can learn it.

This avoids an avoidable Phase 4A artifact where an existing minority black-box
class can disappear entirely from the local fitting subset.
"""
from __future__ import annotations

import numpy as np

from .surrogate_core import (
    bn_branch_rules,
    sequence_hash,
    tree_branch_rules,
)


def stratified_local_split(sequences, labels, factual, eval_fraction=0.30):
    sequences = [tuple(seq) for seq in sequences]
    labels = np.asarray(labels, dtype=int)
    factual = tuple(factual)

    fit_indices = []
    eval_indices = []

    for cls in sorted(set(labels.tolist())):
        cls_indices = [
            i
            for i, value in enumerate(labels)
            if int(value) == int(cls)
        ]
        cls_indices = sorted(
            cls_indices,
            key=lambda i: (
                sequence_hash(sequences[i]),
                i,
            ),
        )

        factual_indices = [
            i
            for i in cls_indices
            if sequences[i] == factual
        ]
        nonfactual = [
            i
            for i in cls_indices
            if sequences[i] != factual
        ]

        n = len(cls_indices)
        if n <= 1:
            fit_indices.extend(cls_indices)
            continue

        n_eval = max(
            1,
            int(round(eval_fraction * n)),
        )
        n_eval = min(
            n_eval,
            n - 1,
        )

        # Factual is never placed in eval.
        eval_take = min(
            n_eval,
            len(nonfactual),
        )
        chosen_eval = nonfactual[:eval_take]
        chosen_eval_set = set(chosen_eval)

        fit_indices.extend(
            i
            for i in cls_indices
            if i not in chosen_eval_set
        )
        eval_indices.extend(chosen_eval)

    # Defensive global eval fallback if every class was singleton.
    if not eval_indices:
        movable = sorted(
            [
                i
                for i, seq in enumerate(sequences)
                if seq != factual
            ],
            key=lambda i: (
                sequence_hash(sequences[i]),
                i,
            ),
        )
        if movable:
            move = movable[0]
            fit_indices = [
                i
                for i in fit_indices
                if i != move
            ]
            eval_indices.append(move)

    fit_indices = sorted(set(fit_indices))
    eval_indices = sorted(set(eval_indices))

    if set(fit_indices) & set(eval_indices):
        raise AssertionError("Local fit/eval overlap")
    if set(fit_indices) | set(eval_indices) != set(range(len(sequences))):
        raise AssertionError("Local fit/eval partition does not cover pool")
    if factual not in [sequences[i] for i in fit_indices]:
        raise AssertionError("Factual sequence must remain in fit")

    return {
        "fit_indices": fit_indices,
        "eval_indices": eval_indices,
        "fit_sequences": [sequences[i] for i in fit_indices],
        "fit_labels": labels[fit_indices],
        "eval_sequences": [sequences[i] for i in eval_indices],
        "eval_labels": labels[eval_indices],
    }


def run_all_surrogates_stratified(
    sequences,
    bb_labels,
    factual,
    factual_label,
):
    split = stratified_local_split(
        sequences,
        bb_labels,
        factual,
        eval_fraction=0.30,
    )

    fit_sequences = split["fit_sequences"]
    fit_labels = split["fit_labels"]
    eval_sequences = split["eval_sequences"]
    eval_labels = split["eval_labels"]

    outputs = []

    outputs.append(
        tree_branch_rules(
            "LORE",
            fit_sequences,
            fit_labels,
            eval_sequences,
            eval_labels,
            factual,
            factual_label,
            max_depth=None,
            min_samples_leaf=2,
            random_state=0,
            balance=True,
        )
    )

    outputs.append(
        tree_branch_rules(
            "DT",
            fit_sequences,
            fit_labels,
            eval_sequences,
            eval_labels,
            factual,
            factual_label,
            max_depth=4,
            min_samples_leaf=5,
            random_state=0,
            balance=False,
        )
    )

    outputs.append(
        bn_branch_rules(
            fit_sequences,
            fit_labels,
            eval_sequences,
            eval_labels,
            factual,
            factual_label,
            alpha=1.0,
        )
    )

    # Phase 4B-v2 contract: every branch must expose a deterministic list of
    # distinct real counterfactual candidates while retaining the legacy
    # ``counterfactual_rule`` best-rule field. This does not alter fitting or
    # held-out evaluation; it only makes the already extracted candidates
    # available to the rule-bank writer.
    for branch in outputs:
        rules = branch.get("counterfactual_rules")
        if rules is None:
            raise AssertionError(
                f"Missing multi-rule output for {branch['branch']}"
            )
        candidate_ids = [
            rule.get("candidate_id")
            for rule in rules
        ]
        if any(value is None for value in candidate_ids):
            raise AssertionError(
                f"Missing candidate_id for {branch['branch']}"
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AssertionError(
                f"Duplicate candidate_id for {branch['branch']}"
            )
        if int(branch.get("counterfactual_rule_count", -1)) != len(rules):
            raise AssertionError(
                f"Counterfactual candidate count mismatch for {branch['branch']}"
            )

    return {
        **split,
        "branches": outputs,
        "counterfactual_rule_count_total": int(
            sum(
                len(branch["counterfactual_rules"])
                for branch in outputs
            )
        ),
    }
