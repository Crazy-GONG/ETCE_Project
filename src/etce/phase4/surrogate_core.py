"""ETCE Phase 4A local surrogate-rule extraction utilities.

Internal evidence sources:
1. Process-aware LORE branch adaptation:
   - process-valid local neighborhood
   - class-balanced fit subset
   - local decision tree
   - factual and counterfactual path rules
2. Local CART (DT) branch:
   - natural local fit subset
   - shallow CART
3. Naive-Bayes-structured Bayesian Network (BN) branch:
   - Y -> X_i categorical structure
   - Laplace-smoothed local probabilities
   - rule conditions selected from class log-likelihood contributions

Important terminology:
The LORE branch is an ETCE process-aware adaptation of the LORE workflow
(neighborhood -> local decision tree -> factual/counterfactual rules). It is
NOT the external official LORE/LORE_sa baseline. External baseline comparison
is a later experiment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from etce.phase3.core import (
    DIRECT_PROXIES,
    TERMINAL_OFFER_ACTIVITIES,
    FORBIDDEN_MUTATION_TARGETS,
    flow_valid,
    process_support_score,
    sequence_edit_count,
)
from etce.phase3.semantic_core import (
    family_preserved,
    semantic_crossover,
    semantic_mutate_once,
    semantic_mutation_options,
)


SURROGATES = ("LORE", "DT", "BN")


def sequence_key(sequence: Sequence[str]) -> str:
    return "\x1f".join(sequence)


def sequence_hash(sequence: Sequence[str]) -> str:
    return hashlib.sha256(sequence_key(sequence).encode("utf-8")).hexdigest()


def deterministic_split(sequence: Sequence[str]) -> str:
    # 70/30 fit/eval, independent of black-box label.
    value = int(sequence_hash(sequence)[:8], 16) % 10
    return "fit" if value < 7 else "eval"


def generate_process_valid_pool(
    factual: Sequence[str],
    constraints: dict,
    rng_seed: int,
    pool_size: int = 320,
    max_edits: int = 3,
    max_options_per_position: int = 8,
    crossover_rate: float = 0.35,
    max_attempts: int = 20000,
):
    """Generate a unique process-valid local pool without using black-box labels.

    Locality is controlled by the maximum Hamming edit count. All edits preserve
    event family and the same-position support rule inherited from Phase 3A.1.
    """
    factual = tuple(factual)
    rng = random.Random(int(rng_seed))
    seen = {factual}
    pool = [factual]

    # Deterministic one-edit shell first.
    for pos in range(len(factual)):
        for replacement in semantic_mutation_options(
            factual,
            pos,
            constraints,
            max_options=max_options_per_position,
        ):
            child = list(factual)
            child[pos] = replacement
            child = tuple(child)
            if child in seen:
                continue
            if sequence_edit_count(factual, child) > max_edits:
                continue
            if not flow_valid(child, constraints["edge_count"]):
                continue
            seen.add(child)
            pool.append(child)
            if len(pool) >= pool_size:
                return pool

    attempts = 0
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(pool)
        child = None

        if len(pool) >= 2 and rng.random() < crossover_rate:
            parent_b = rng.choice(pool)
            child = semantic_crossover(
                parent,
                parent_b,
                factual,
                constraints,
                rng,
            )
        else:
            child = tuple(parent)

        if child is None:
            continue

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
        edits = sequence_edit_count(factual, child)
        if edits <= 0 or edits > max_edits:
            continue
        if not flow_valid(child, constraints["edge_count"]):
            continue

        # Defensive checks against semantic/process regressions.
        valid = True
        for i, (old, new) in enumerate(zip(factual, child)):
            if old in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES:
                if new != old:
                    valid = False
                    break
            elif new != old:
                if new in FORBIDDEN_MUTATION_TARGETS:
                    valid = False
                    break
                if not family_preserved(old, new):
                    valid = False
                    break
        if not valid:
            continue

        seen.add(child)
        pool.append(child)

    return pool


def make_position_domains(sequences: Sequence[Sequence[str]]):
    if not sequences:
        raise ValueError("Empty sequence set")
    prefix = len(sequences[0])
    domains = []
    for pos in range(prefix):
        domains.append(
            tuple(sorted({seq[pos] for seq in sequences}))
        )
    return domains


def make_onehot_schema(position_domains):
    features = []
    for pos, domain in enumerate(position_domains):
        for activity in domain:
            features.append((pos, activity))
    return features


def onehot_transform(sequences, feature_meta):
    index = {
        item: j
        for j, item in enumerate(feature_meta)
    }
    X = np.zeros(
        (len(sequences), len(feature_meta)),
        dtype=np.float32,
    )
    for row, seq in enumerate(sequences):
        for pos, activity in enumerate(seq):
            j = index.get((pos, activity))
            if j is not None:
                X[row, j] = 1.0
    return X


def canonicalize_domains(allowed, position_domains):
    conditions = []
    for pos, domain in enumerate(position_domains):
        current = tuple(sorted(allowed[pos]))
        full = tuple(sorted(domain))
        if current == full:
            continue
        if not current:
            # Empty rule is unsatisfiable.
            return None
        conditions.append(
            {
                "position": int(pos + 1),
                "operator": "in",
                "values": list(current),
            }
        )
    return conditions


def apply_conditions(sequence, conditions):
    for cond in conditions:
        pos = int(cond["position"]) - 1
        if sequence[pos] not in set(cond["values"]):
            return False
    return True


def tree_path_rule(
    tree: DecisionTreeClassifier,
    x_vector,
    feature_meta,
    position_domains,
):
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold

    allowed = [
        set(domain)
        for domain in position_domains
    ]
    node = 0
    while children_left[node] != children_right[node]:
        j = int(features[node])
        pos, activity = feature_meta[j]
        threshold = float(thresholds[node])
        value = float(x_vector[j])

        if value <= threshold:
            allowed[pos].discard(activity)
            node = int(children_left[node])
        else:
            allowed[pos].intersection_update({activity})
            node = int(children_right[node])

        if not allowed[pos]:
            return None, node

    conditions = canonicalize_domains(
        allowed,
        position_domains,
    )
    return conditions, node


def enumerate_tree_leaf_rules(
    tree: DecisionTreeClassifier,
    feature_meta,
    position_domains,
):
    output = []
    base = [
        set(domain)
        for domain in position_domains
    ]

    def rec(node, allowed):
        left = int(tree.tree_.children_left[node])
        right = int(tree.tree_.children_right[node])

        if left == right:
            counts = tree.tree_.value[node][0]
            predicted_class = int(tree.classes_[int(np.argmax(counts))])
            conditions = canonicalize_domains(
                allowed,
                position_domains,
            )
            if conditions is not None:
                output.append(
                    {
                        "leaf_node": int(node),
                        "predicted_class": predicted_class,
                        "conditions": conditions,
                    }
                )
            return

        j = int(tree.tree_.feature[node])
        pos, activity = feature_meta[j]

        left_allowed = [
            set(v)
            for v in allowed
        ]
        left_allowed[pos].discard(activity)
        if left_allowed[pos]:
            rec(left, left_allowed)

        right_allowed = [
            set(v)
            for v in allowed
        ]
        right_allowed[pos].intersection_update({activity})
        if right_allowed[pos]:
            rec(right, right_allowed)

    rec(0, base)
    return output


def rule_metrics(
    conditions,
    predicted_class,
    sequences,
    bb_labels,
):
    covered = [
        i
        for i, seq in enumerate(sequences)
        if apply_conditions(seq, conditions)
    ]
    support = len(covered)
    if support == 0:
        fidelity = None
    else:
        fidelity = float(
            np.mean(
                [
                    int(bb_labels[i]) == int(predicted_class)
                    for i in covered
                ]
            )
        )
    return {
        "support": int(support),
        "fidelity": fidelity,
        "coverage": (
            0.0
            if not sequences
            else support / len(sequences)
        ),
        "covered_indices": covered,
    }


def witness_distance(
    factual,
    conditions,
    predicted_class,
    sequences,
    bb_labels,
):
    candidates = []
    for i, seq in enumerate(sequences):
        if int(bb_labels[i]) != int(predicted_class):
            continue
        if not apply_conditions(seq, conditions):
            continue
        candidates.append(
            (
                sequence_edit_count(factual, seq),
                sequence_hash(seq),
                i,
                seq,
            )
        )
    if not candidates:
        return None
    candidates.sort()
    edits, _, idx, seq = candidates[0]
    return {
        "index": idx,
        "sequence": tuple(seq),
        "edit_count": int(edits),
    }



def _counterfactual_signature(rule):
    """Canonical identity for de-duplicating candidate counterfactual rules."""
    return (
        int(rule["predicted_class"]),
        json.dumps(
            rule["conditions"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _number_unique_counterfactual_rules(candidates):
    """Keep deterministic first occurrence of each real rule and assign stable IDs.

    The caller controls candidate order. This helper never changes rule metrics and
    never recomputes support/fidelity; it only de-duplicates identical canonical
    condition sets and adds a branch-local deterministic ``candidate_id``.
    """
    output = []
    seen = set()
    for rule in candidates:
        signature = _counterfactual_signature(rule)
        if signature in seen:
            continue
        seen.add(signature)
        numbered = dict(rule)
        numbered["candidate_id"] = f"cf_{len(output):03d}"
        output.append(numbered)
    return output


def _without_candidate_id(rule):
    """Return the legacy best-rule payload without the v2-only candidate ID."""
    if rule is None:
        return None
    return {
        key: value
        for key, value in rule.items()
        if key != "candidate_id"
    }

def tree_branch_rules(
    branch_name,
    fit_sequences,
    fit_labels,
    eval_sequences,
    eval_labels,
    factual,
    factual_label,
    max_depth,
    min_samples_leaf,
    random_state=0,
    balance=False,
):
    """Fit a tree branch and extract factual/opposite-class rules."""
    fit_sequences = list(fit_sequences)
    fit_labels = np.asarray(fit_labels, dtype=int)

    if len(set(fit_labels.tolist())) < 2:
        return {
            "branch": branch_name,
            "status": "single_class_fit",
            "surrogate_fidelity_eval": None,
            "factual_rule": None,
            "counterfactual_rule": None,
            "counterfactual_rules": [],
            "counterfactual_rule_count": 0,
            "fit_size_used": len(fit_sequences),
        }

    if balance:
        by_class = {
            cls: [
                i
                for i, value in enumerate(fit_labels)
                if int(value) == cls
            ]
            for cls in (0, 1)
        }
        n = min(
            len(by_class[0]),
            len(by_class[1]),
        )
        if n == 0:
            return {
                "branch": branch_name,
                "status": "cannot_balance",
                "surrogate_fidelity_eval": None,
                "factual_rule": None,
                "counterfactual_rule": None,
                "counterfactual_rules": [],
                "counterfactual_rule_count": 0,
                "fit_size_used": 0,
            }
        # Deterministic locality-first balanced subset.
        selected = []
        for cls in (0, 1):
            ranked = sorted(
                by_class[cls],
                key=lambda i: (
                    sequence_edit_count(
                        factual,
                        fit_sequences[i],
                    ),
                    sequence_hash(
                        fit_sequences[i]
                    ),
                ),
            )
            selected.extend(ranked[:n])
        selected = sorted(selected)
        fit_sequences = [
            fit_sequences[i]
            for i in selected
        ]
        fit_labels = fit_labels[selected]

    all_sequences = (
        list(fit_sequences)
        + list(eval_sequences)
        + [tuple(factual)]
    )
    position_domains = make_position_domains(
        all_sequences
    )
    feature_meta = make_onehot_schema(
        position_domains
    )

    X_fit = onehot_transform(
        fit_sequences,
        feature_meta,
    )
    X_eval = onehot_transform(
        eval_sequences,
        feature_meta,
    )
    x_fact = onehot_transform(
        [factual],
        feature_meta,
    )[0]

    tree = DecisionTreeClassifier(
        random_state=int(random_state),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        criterion="gini",
    )
    tree.fit(
        X_fit,
        fit_labels,
    )

    if len(eval_sequences):
        pred_eval = tree.predict(
            X_eval
        ).astype(int)
        fidelity_eval = float(
            np.mean(
                pred_eval
                == np.asarray(
                    eval_labels,
                    dtype=int,
                )
            )
        )
    else:
        fidelity_eval = None

    factual_conditions, factual_leaf = tree_path_rule(
        tree,
        x_fact,
        feature_meta,
        position_domains,
    )
    factual_pred = int(
        tree.predict(
            x_fact.reshape(1, -1)
        )[0]
    )

    factual_rule = None
    if factual_conditions is not None:
        met = rule_metrics(
            factual_conditions,
            factual_pred,
            eval_sequences,
            eval_labels,
        )
        factual_rule = {
            "predicted_class": factual_pred,
            "conditions": factual_conditions,
            "support": met["support"],
            "fidelity": met["fidelity"],
            "coverage": met["coverage"],
            "leaf_node": int(factual_leaf),
            "rule_length": len(factual_conditions),
        }

    target = 1 - int(factual_label)
    leaf_rules = [
        rule
        for rule in enumerate_tree_leaf_rules(
            tree,
            feature_meta,
            position_domains,
        )
        if int(rule["predicted_class"]) == target
    ]

    counterfactual_candidates = []
    search_sequences = (
        list(fit_sequences)
        + list(eval_sequences)
    )
    search_labels = np.concatenate(
        (
            np.asarray(fit_labels, dtype=int),
            np.asarray(eval_labels, dtype=int),
        )
    )
    for leaf in leaf_rules:
        witness = witness_distance(
            factual,
            leaf["conditions"],
            target,
            search_sequences,
            search_labels,
        )
        if witness is None:
            continue
        met = rule_metrics(
            leaf["conditions"],
            target,
            eval_sequences,
            eval_labels,
        )
        counterfactual_candidates.append(
            {
                "predicted_class": target,
                "conditions": leaf["conditions"],
                "support": met["support"],
                "fidelity": met["fidelity"],
                "coverage": met["coverage"],
                "leaf_node": int(
                    leaf["leaf_node"]
                ),
                "rule_length": len(
                    leaf["conditions"]
                ),
                "witness_sequence": list(
                    witness["sequence"]
                ),
                "witness_edit_count": int(
                    witness["edit_count"]
                ),
            }
        )

    def cf_key(rule):
        fidelity = (
            -1.0
            if rule["fidelity"] is None
            else float(
                rule["fidelity"]
            )
        )
        return (
            int(
                rule[
                    "witness_edit_count"
                ]
            ),
            int(
                rule["rule_length"]
            ),
            -int(
                rule["support"]
            ),
            -fidelity,
            json.dumps(
                rule["conditions"],
                sort_keys=True,
            ),
        )

    ranked_candidates = sorted(
        counterfactual_candidates,
        key=cf_key,
    )
    counterfactual_rules = (
        _number_unique_counterfactual_rules(
            ranked_candidates
        )
    )
    counterfactual_rule = (
        None
        if not counterfactual_rules
        else _without_candidate_id(
            counterfactual_rules[0]
        )
    )

    return {
        "branch": branch_name,
        "status": "ok",
        "surrogate_fidelity_eval": fidelity_eval,
        "factual_rule": factual_rule,
        # Backward-compatible Phase 4A/4B interface: the same best CF rule.
        "counterfactual_rule": counterfactual_rule,
        # Phase 4B-v2 interface: all distinct real CF candidates.
        "counterfactual_rules": counterfactual_rules,
        "counterfactual_rule_count": len(
            counterfactual_rules
        ),
        "fit_size_used": len(
            fit_sequences
        ),
        "tree_depth": int(
            tree.get_depth()
        ),
        "tree_leaves": int(
            tree.get_n_leaves()
        ),
    }


@dataclass
class NaiveBayesBN:
    alpha: float
    classes: tuple
    class_count: dict
    position_values: list
    conditional_counts: dict
    n_samples: int

    @classmethod
    def fit(
        cls,
        sequences,
        labels,
        alpha=1.0,
    ):
        labels = [
            int(v)
            for v in labels
        ]
        classes = (0, 1)
        position_values = make_position_domains(
            sequences
        )
        class_count = Counter(labels)
        conditional_counts = {}
        for y in classes:
            for pos, domain in enumerate(
                position_values
            ):
                counts = Counter(
                    seq[pos]
                    for seq, label in zip(
                        sequences,
                        labels,
                    )
                    if label == y
                )
                conditional_counts[
                    (y, pos)
                ] = counts
        return cls(
            alpha=float(alpha),
            classes=classes,
            class_count=dict(
                class_count
            ),
            position_values=position_values,
            conditional_counts=conditional_counts,
            n_samples=len(labels),
        )

    def log_prior(self, y):
        return math.log(
            (
                self.class_count.get(
                    y,
                    0,
                )
                + self.alpha
            )
            / (
                self.n_samples
                + self.alpha
                * len(self.classes)
            )
        )

    def log_conditional(
        self,
        y,
        pos,
        value,
    ):
        domain = self.position_values[pos]
        counts = self.conditional_counts[
            (y, pos)
        ]
        numerator = (
            counts.get(
                value,
                0,
            )
            + self.alpha
        )
        denominator = (
            self.class_count.get(
                y,
                0,
            )
            + self.alpha
            * max(
                1,
                len(domain),
            )
        )
        return math.log(
            numerator
            / denominator
        )

    def log_odds(
        self,
        sequence,
        selected_positions=None,
    ):
        if selected_positions is None:
            selected_positions = range(
                len(sequence)
            )
        value = (
            self.log_prior(1)
            - self.log_prior(0)
        )
        for pos in selected_positions:
            value += (
                self.log_conditional(
                    1,
                    pos,
                    sequence[pos],
                )
                - self.log_conditional(
                    0,
                    pos,
                    sequence[pos],
                )
            )
        return float(value)

    def predict_one(
        self,
        sequence,
        selected_positions=None,
    ):
        return int(
            self.log_odds(
                sequence,
                selected_positions,
            )
            >= 0.0
        )

    def predict(self, sequences):
        return np.asarray(
            [
                self.predict_one(seq)
                for seq in sequences
            ],
            dtype=int,
        )


def bn_rule_for_reference(
    bn: NaiveBayesBN,
    reference,
    target_class,
    eval_sequences,
    eval_labels,
):
    contributions = []
    for pos in range(
        len(reference)
    ):
        contribution = (
            bn.log_conditional(
                1,
                pos,
                reference[pos],
            )
            - bn.log_conditional(
                0,
                pos,
                reference[pos],
            )
        )
        support_direction = (
            contribution
            if target_class == 1
            else -contribution
        )
        contributions.append(
            (
                -support_direction,
                pos,
                contribution,
            )
        )
    contributions.sort()

    selected = []
    predicted = int(
        bn.log_prior(1)
        - bn.log_prior(0)
        >= 0
    )

    for _, pos, _ in contributions:
        if pos not in selected:
            selected.append(pos)
        predicted = bn.predict_one(
            reference,
            selected,
        )
        if (
            predicted
            == int(target_class)
            and len(selected) >= 1
        ):
            break

    if predicted != int(target_class):
        return None

    conditions = [
        {
            "position": int(
                pos + 1
            ),
            "operator": "in",
            "values": [
                reference[pos]
            ],
        }
        for pos in sorted(
            selected
        )
    ]
    met = rule_metrics(
        conditions,
        target_class,
        eval_sequences,
        eval_labels,
    )
    return {
        "predicted_class": int(
            target_class
        ),
        "conditions": conditions,
        "support": met["support"],
        "fidelity": met["fidelity"],
        "coverage": met["coverage"],
        "rule_length": len(
            conditions
        ),
    }


def bn_branch_rules(
    fit_sequences,
    fit_labels,
    eval_sequences,
    eval_labels,
    factual,
    factual_label,
    alpha=1.0,
):
    fit_labels = np.asarray(
        fit_labels,
        dtype=int,
    )
    if len(set(fit_labels.tolist())) < 2:
        return {
            "branch": "BN",
            "status": "single_class_fit",
            "surrogate_fidelity_eval": None,
            "factual_rule": None,
            "counterfactual_rule": None,
            "counterfactual_rules": [],
            "counterfactual_rule_count": 0,
            "fit_size_used": len(
                fit_sequences
            ),
            "bn_alpha": float(alpha),
        }

    bn = NaiveBayesBN.fit(
        fit_sequences,
        fit_labels,
        alpha=float(alpha),
    )

    fidelity_eval = None
    if len(eval_sequences):
        fidelity_eval = float(
            np.mean(
                bn.predict(
                    eval_sequences
                )
                == np.asarray(
                    eval_labels,
                    dtype=int,
                )
            )
        )

    factual_pred = bn.predict_one(
        factual
    )
    factual_rule = bn_rule_for_reference(
        bn,
        factual,
        factual_pred,
        eval_sequences,
        eval_labels,
    )

    target = 1 - int(
        factual_label
    )
    candidates = []
    for seq, label in zip(
        list(fit_sequences)
        + list(eval_sequences),
        np.concatenate(
            (
                np.asarray(
                    fit_labels,
                    dtype=int,
                ),
                np.asarray(
                    eval_labels,
                    dtype=int,
                ),
            )
        ),
    ):
        if int(label) != target:
            continue
        if bn.predict_one(
            seq
        ) != target:
            continue
        candidates.append(
            (
                sequence_edit_count(
                    factual,
                    seq,
                ),
                sequence_hash(seq),
                tuple(seq),
            )
        )

    candidates.sort()

    # Preserve the exact legacy best-rule behavior: use only the nearest
    # deterministic witness for ``counterfactual_rule``.
    counterfactual_rule = None
    if candidates:
        witness = candidates[0][2]
        counterfactual_rule = (
            bn_rule_for_reference(
                bn,
                witness,
                target,
                eval_sequences,
                eval_labels,
            )
        )
        if counterfactual_rule is not None:
            counterfactual_rule[
                "witness_sequence"
            ] = list(witness)
            counterfactual_rule[
                "witness_edit_count"
            ] = int(
                sequence_edit_count(
                    factual,
                    witness,
                )
            )

    # Phase 4B-v2: derive one rule from every genuine target-class witness,
    # then de-duplicate identical canonical condition sets. Candidate order is
    # deterministic because ``candidates`` is sorted by edit distance and hash.
    multi_rule_candidates = []
    for _, _, witness in candidates:
        rule = bn_rule_for_reference(
            bn,
            witness,
            target,
            eval_sequences,
            eval_labels,
        )
        if rule is None:
            continue
        rule["witness_sequence"] = list(witness)
        rule["witness_edit_count"] = int(
            sequence_edit_count(
                factual,
                witness,
            )
        )
        multi_rule_candidates.append(rule)

    counterfactual_rules = (
        _number_unique_counterfactual_rules(
            multi_rule_candidates
        )
    )

    return {
        "branch": "BN",
        "status": "ok",
        "surrogate_fidelity_eval": fidelity_eval,
        "factual_rule": factual_rule,
        # Backward-compatible Phase 4A/4B interface.
        "counterfactual_rule": counterfactual_rule,
        # Phase 4B-v2 interface.
        "counterfactual_rules": counterfactual_rules,
        "counterfactual_rule_count": len(
            counterfactual_rules
        ),
        "fit_size_used": len(
            fit_sequences
        ),
        "bn_alpha": float(alpha),
    }


def run_all_surrogates(
    sequences,
    bb_labels,
    factual,
    factual_label,
):
    """Train/extract all three internal evidence branches.

    The neighborhood split is deterministic and label-independent.
    Factual sequence is always assigned to fit.
    """
    fit_sequences = []
    fit_labels = []
    eval_sequences = []
    eval_labels = []

    factual = tuple(factual)
    for seq, label in zip(
        sequences,
        bb_labels,
    ):
        seq = tuple(seq)
        split = (
            "fit"
            if seq == factual
            else deterministic_split(seq)
        )
        if split == "fit":
            fit_sequences.append(seq)
            fit_labels.append(int(label))
        else:
            eval_sequences.append(seq)
            eval_labels.append(int(label))

    # Defensive: if eval is empty, move deterministic non-factual rows.
    if not eval_sequences:
        movable = [
            i
            for i, seq in enumerate(
                fit_sequences
            )
            if seq != factual
        ]
        for idx in movable[-max(
            1,
            len(movable) // 4,
        ):]:
            eval_sequences.append(
                fit_sequences[idx]
            )
            eval_labels.append(
                fit_labels[idx]
            )

    outputs = []

    lore = tree_branch_rules(
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
    outputs.append(lore)

    dt = tree_branch_rules(
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
    outputs.append(dt)

    bn = bn_branch_rules(
        fit_sequences,
        fit_labels,
        eval_sequences,
        eval_labels,
        factual,
        factual_label,
        alpha=1.0,
    )
    outputs.append(bn)

    return {
        "fit_sequences": fit_sequences,
        "fit_labels": fit_labels,
        "eval_sequences": eval_sequences,
        "eval_labels": eval_labels,
        "branches": outputs,
    }
