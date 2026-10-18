"""Core utilities for ETCE Phase 3A.

Phase 3A is a software/method pilot for constrained evolutionary counterfactual
neighborhood generation. It reuses the frozen Phase 2 encoders and canonical
black-box checkpoints.

Important:
- No true label is used by the counterfactual search.
- Timestamps are preserved exactly in Phase 3A, so temporal ordering is preserved.
- Only activity substitutions are allowed in Phase 3A.
- Direct outcome proxies and terminal outcome activities are never proposed as mutations.
- Historical terminal offer activities already present in the factual prefix are immutable.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import torch

from etce.phase2.models import TabularMLP, ActivityLSTM
from etce.utils.io import read_json


DIRECT_PROXIES = frozenset({"A_Pending", "A_Denied", "A_Cancelled"})
TERMINAL_OFFER_ACTIVITIES = frozenset({"O_Accepted", "O_Refused", "O_Cancelled"})
FORBIDDEN_MUTATION_TARGETS = DIRECT_PROXIES | TERMINAL_OFFER_ACTIVITIES


@dataclass(frozen=True)
class Candidate:
    activities: tuple[str, ...]
    timestamps: tuple
    probability: float | None = None
    edit_count: int | None = None
    target_score: float | None = None
    process_support: float | None = None


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_process_constraints(results_phase3):
    results_phase3 = Path(results_phase3)
    meta = read_json(results_phase3 / "constraints/process_constraints.json")

    edge_rows = _read_csv(results_phase3 / "constraints/dfg_edges.csv")
    edge_count = {
        (row["source_activity"], row["target_activity"]): int(row["count"])
        for row in edge_rows
    }
    position_rows = _read_csv(results_phase3 / "constraints/activity_position_counts.csv")
    position_count = {
        (int(row["position"]), row["activity"]): int(row["count"])
        for row in position_rows
    }

    meta["edge_count"] = edge_count
    meta["position_count"] = position_count
    meta["mutable_activities"] = tuple(meta["mutable_activities"])
    return meta


def flow_valid(activities: Sequence[str], edge_count: dict) -> bool:
    return all((a, b) in edge_count for a, b in zip(activities, activities[1:]))


def temporal_valid(timestamps: Sequence) -> bool:
    return all(a <= b for a, b in zip(timestamps, timestamps[1:]))


def process_support_score(activities: Sequence[str], edge_count: dict) -> float:
    if len(activities) <= 1:
        return 0.0
    values = []
    for a, b in zip(activities, activities[1:]):
        count = edge_count.get((a, b), 0)
        if count <= 0:
            return float("-inf")
        values.append(math.log1p(count))
    return float(np.mean(values))


def sequence_edit_count(factual: Sequence[str], candidate: Sequence[str]) -> int:
    if len(factual) != len(candidate):
        raise ValueError("Phase 3A uses fixed-length substitution-only candidates")
    return int(sum(a != b for a, b in zip(factual, candidate)))


def encode_candidates(
    activities_batch: Sequence[Sequence[str]],
    timestamps_batch: Sequence[Sequence],
    prefix: int,
    artifact_dir,
):
    """Encode arbitrary candidate prefixes using frozen train-fit Phase 2B artifacts."""
    artifact_dir = Path(artifact_dir)
    encoder = read_json(artifact_dir / f"encoder_prefix{prefix}.json")
    tabular_scaler = joblib.load(artifact_dir / f"tabular_scaler_prefix{prefix}.joblib")
    time_scaler = joblib.load(artifact_dir / f"sequence_time_scaler_prefix{prefix}.joblib")

    activity_to_id = {str(k): int(v) for k, v in encoder["activity_to_id"].items()}
    vocab_size = len(encoder["vocabulary"]) + 1

    raw_tabular = []
    activity_sequences = []
    raw_time = []

    for activities, times in zip(activities_batch, timestamps_batch):
        if len(activities) != prefix or len(times) != prefix:
            raise ValueError("Candidate length must equal prefix")
        ids = np.asarray([activity_to_id.get(activity, 0) for activity in activities], dtype=np.int64)
        size = vocab_size
        counts = np.bincount(ids, minlength=size).astype(np.float64)
        frequency = counts / prefix
        last = np.zeros(size, dtype=np.float64)
        first = np.zeros(size, dtype=np.float64)
        last[ids[-1]] = 1.0
        first[ids[0]] = 1.0

        deltas = np.asarray(
            [0.0] + [float((b - a).total_seconds()) for a, b in zip(times, times[1:])],
            dtype=np.float64,
        )
        elapsed = np.asarray(
            [float((value - times[0]).total_seconds()) for value in times],
            dtype=np.float64,
        )
        intervals = deltas[1:]
        temporal = np.asarray(
            [
                elapsed[-1],
                intervals.mean(),
                intervals.std(),
                intervals.max(),
                deltas[-1],
                times[-1].hour,
                times[-1].weekday(),
            ],
            dtype=np.float64,
        )
        raw_tabular.append(np.concatenate((counts, frequency, last, first, temporal)))
        activity_sequences.append(ids)
        raw_time.append(np.column_stack((deltas, elapsed)))

    tabular = tabular_scaler.transform(np.vstack(raw_tabular)).astype(np.float32)
    activity_sequence = np.stack(activity_sequences).astype(np.int64)
    time_sequence = time_scaler.transform(np.vstack(raw_time)).reshape(
        len(raw_time), prefix, 2
    ).astype(np.float32)

    return tabular, activity_sequence, time_sequence, encoder


class CanonicalPredictor:
    def __init__(self, root, model_name: str, prefix: int, manifest_item: dict):
        self.root = Path(root)
        self.model_name = model_name
        self.prefix = int(prefix)
        self.seed = int(manifest_item["selected_seed"])
        self.checkpoint = self.root / manifest_item["canonical_checkpoint"]
        self.artifact_dir = self.root / "results/phase2b/artifacts"

        if model_name in ("RF", "XGBoost"):
            self.model = joblib.load(self.checkpoint)
            self.config = manifest_item.get("hyperparameters", {})
        else:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.config = payload["config"]

            # Input width is recovered from the frozen encoder.
            encoder = read_json(
                self.artifact_dir / f"encoder_prefix{self.prefix}.json"
            )
            vocab_size = len(encoder["vocabulary"]) + 1

            if model_name == "MLP":
                # Need tabular width; obtain it from the frozen scaler.
                scaler = joblib.load(
                    self.artifact_dir / f"tabular_scaler_prefix{self.prefix}.joblib"
                )
                self.model = TabularMLP(
                    int(scaler.n_features_in_),
                    tuple(self.config["hidden_dims"]),
                    float(self.config["dropout"]),
                )
            elif model_name == "LSTM":
                self.model = ActivityLSTM(
                    vocab_size,
                    int(self.config["embedding_dim"]),
                    2,
                    int(self.config["hidden_size"]),
                    int(self.config["num_layers"]),
                    float(self.config["dropout"]),
                    bool(self.config["bidirectional"]),
                )
            else:
                raise ValueError(model_name)

            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()

    def predict_proba(self, activities_batch, timestamps_batch, batch_size=512):
        tabular, activity_sequence, time_sequence, _ = encode_candidates(
            activities_batch,
            timestamps_batch,
            self.prefix,
            self.artifact_dir,
        )

        if self.model_name in ("RF", "XGBoost"):
            return np.asarray(self.model.predict_proba(tabular)[:, 1], dtype=float)

        result = []
        with torch.no_grad():
            for start in range(0, len(tabular), batch_size):
                stop = min(start + batch_size, len(tabular))
                if self.model_name == "MLP":
                    logits = self.model(torch.from_numpy(tabular[start:stop]))
                else:
                    logits = self.model(
                        torch.from_numpy(activity_sequence[start:stop]),
                        torch.from_numpy(time_sequence[start:stop]),
                    )
                result.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(result).astype(float)


def mutation_options(
    activities: Sequence[str],
    position: int,
    constraints: dict,
    max_options: int = 8,
):
    """Return locally DFG-compatible substitutions ranked by train support."""
    factual = activities[position]

    # Historical terminal offer events are observed history and are immutable in Phase 3A.
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


def mutate_once(
    activities: Sequence[str],
    constraints: dict,
    rng: random.Random,
    max_options: int = 8,
):
    editable = []
    for pos in range(len(activities)):
        opts = mutation_options(activities, pos, constraints, max_options=max_options)
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


def crossover(
    parent_a: Sequence[str],
    parent_b: Sequence[str],
    factual: Sequence[str],
    constraints: dict,
    rng: random.Random,
):
    if len(parent_a) < 3:
        return None
    cut = rng.randint(1, len(parent_a) - 1)
    child = tuple(parent_a[:cut]) + tuple(parent_b[cut:])

    # Immutable historical terminal outcomes must remain factual.
    for i, factual_activity in enumerate(factual):
        if factual_activity in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES:
            if child[i] != factual_activity:
                return None

    if any(activity in FORBIDDEN_MUTATION_TARGETS and activity != factual[i]
           for i, activity in enumerate(child)):
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
    # GA population ranking: target score first, then minimal edits, then support.
    return (-target_score, edit_count, -support)


def evolutionary_counterfactual_search(
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
    """Constrained fixed-length evolutionary search.

    Returns a dictionary containing the factual prediction, best counterfactual (if any),
    and search diagnostics. The true outcome label is never an input.
    """
    factual_activities = tuple(factual_activities)
    factual_timestamps = tuple(factual_timestamps)
    if len(factual_activities) != predictor.prefix:
        raise ValueError("Factual prefix length mismatch")
    if not temporal_valid(factual_timestamps):
        raise ValueError("Factual timestamps are not monotonic")

    factual_probability = float(
        predictor.predict_proba([factual_activities], [factual_timestamps])[0]
    )
    factual_pred = int(factual_probability >= 0.5)
    target_label = 1 - factual_pred

    rng = random.Random(int(rng_seed))
    edge_count = constraints["edge_count"]
    factual_flow = flow_valid(factual_activities, edge_count)

    seen = {factual_activities}
    population = []

    # Deterministic single-edit seeds first, ranked by train support.
    for position in range(len(factual_activities)):
        for replacement in mutation_options(
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

    # If the factual prefix is already train-DFG valid, it may act as a parent.
    if factual_flow:
        population.append(factual_activities)

    # Bound initialization.
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
            target_score = float(prob if target_label == 1 else 1.0 - prob)
            pred = int(prob >= 0.5)

            scored.append(
                {
                    "seq": seq,
                    "prob": float(prob),
                    "pred": pred,
                    "edits": edits,
                    "support": support,
                    "target_score": target_score,
                }
            )
            if pred == target_label:
                all_successes.append(scored[-1])

        if all_successes:
            # Final CF choice prioritizes minimality, then target confidence, then support.
            best = sorted(
                all_successes,
                key=lambda x: (x["edits"], -x["target_score"], -x["support"], x["seq"]),
            )[0]
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
                "normalized_sequence_distance": best["edits"] / len(factual_activities),
                "temporal_distance": 0.0,
                "process_support": best["support"],
                "generation_found": generation,
                "candidates_evaluated": evaluated,
            }

        if generation == generations:
            break

        # Keep the most target-directed parents.
        scored.sort(
            key=lambda x: _rank_key(
                x["prob"], target_label, x["edits"], x["support"]
            )
        )
        parents = [x["seq"] for x in scored[:population_size]]
        if not parents:
            break

        children = []
        attempts = 0
        max_attempts = population_size * 30
        while len(children) < population_size and attempts < max_attempts:
            attempts += 1
            child = None

            if len(parents) >= 2 and rng.random() < crossover_rate:
                pa, pb = rng.sample(parents, 2)
                child = crossover(pa, pb, factual_activities, constraints, rng)
            else:
                child = tuple(rng.choice(parents))

            if child is None:
                continue

            if rng.random() < mutation_rate:
                mutated = mutate_once(
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

            # Historical terminal offer activities are immutable.
            bad = False
            for i, factual_activity in enumerate(factual_activities):
                if factual_activity in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES:
                    if child[i] != factual_activity:
                        bad = True
                        break
            if bad:
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
