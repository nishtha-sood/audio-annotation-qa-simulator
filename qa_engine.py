"""
QA Engine: Inter-annotator agreement metrics, edge case detection,
annotator performance scoring, and consensus labeling.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from typing import List, Dict, Optional
from collections import Counter


def compute_pairwise_kappa(df: pd.DataFrame, annotator_cols: List[str]) -> np.ndarray:
    """Compute pairwise Cohen's Kappa matrix for all annotator pairs."""
    n = len(annotator_cols)
    matrix = np.ones((n, n))  # diagonal = 1.0

    for i in range(n):
        for j in range(i + 1, n):
            col_i = df[annotator_cols[i]].tolist()
            col_j = df[annotator_cols[j]].tolist()
            try:
                k = cohen_kappa_score(col_i, col_j)
            except Exception:
                k = 0.0
            matrix[i][j] = k
            matrix[j][i] = k

    return matrix


def compute_fleiss_kappa(df: pd.DataFrame, annotator_cols: List[str], labels: List[str]) -> float:
    """
    Compute Fleiss' Kappa for multi-rater categorical agreement.
    Generalization of Cohen's Kappa to multiple annotators.
    """
    n_items = len(df)
    n_raters = len(annotator_cols)
    n_categories = len(labels)

    if n_items == 0 or n_raters < 2:
        return 0.0

    # Build count matrix: n_items × n_categories
    count_matrix = np.zeros((n_items, n_categories))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    for i, row in df.iterrows():
        for col in annotator_cols:
            label = row[col]
            if label in label_to_idx:
                count_matrix[i - df.index[0], label_to_idx[label]] += 1

    # Fleiss' Kappa formula
    N = n_items
    n = n_raters

    # P_i: proportion agreement for each item
    P_i = np.sum(count_matrix * (count_matrix - 1), axis=1) / (n * (n - 1))
    P_bar = np.mean(P_i)

    # P_j: proportion of labels in category j
    P_j = np.sum(count_matrix, axis=0) / (N * n)
    P_e = np.sum(P_j ** 2)

    if P_e == 1.0:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return float(np.clip(kappa, -1.0, 1.0))


def get_consensus_label(labels_list: List[str]) -> str:
    """Return the majority-vote consensus label, or 'No Consensus' if tied."""
    if not labels_list:
        return "Unknown"
    counter = Counter(labels_list)
    most_common = counter.most_common(2)
    if len(most_common) == 1:
        return most_common[0][0]
    if most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    return "No Consensus"


def flag_edge_cases(
    df: pd.DataFrame,
    annotator_cols: List[str],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Flag tracks where annotator disagreement exceeds threshold.
    Disagreement rate = 1 - (proportion voting for majority label)
    """
    result = df.copy()

    def disagreement(row):
        vals = [row[c] for c in annotator_cols]
        counter = Counter(vals)
        majority_pct = counter.most_common(1)[0][1] / len(vals)
        return round(1 - majority_pct, 4)

    result["disagreement_rate"] = result.apply(disagreement, axis=1)
    result["flagged"] = result["disagreement_rate"] >= threshold

    return result


def compute_category_stats(
    df: pd.DataFrame,
    annotator_cols: List[str],
    labels: List[str]
) -> Dict[str, Dict]:
    """Compute per-label agreement statistics."""
    stats = {}

    for label in labels:
        # For each track, what fraction of annotators chose this label?
        label_fractions = []
        for _, row in df.iterrows():
            vals = [row[c] for c in annotator_cols]
            frac = sum(1 for v in vals if v == label) / len(vals)
            label_fractions.append(frac)

        avg_agreement = float(np.mean(label_fractions)) if label_fractions else 0.0
        # Tracks where at least one annotator used this label
        n_tracks_with_label = sum(1 for f in label_fractions if f > 0)

        stats[label] = {
            "avg_agreement": avg_agreement,
            "n_tracks": n_tracks_with_label,
            "std": float(np.std(label_fractions))
        }

    return stats


def compute_annotator_stats(
    df: pd.DataFrame,
    annotator_cols: List[str]
) -> Dict[str, Dict]:
    """Compute per-annotator performance metrics."""
    stats = {}
    n_tracks = len(df)

    # Compute consensus for each track
    consensus_labels = df[annotator_cols].apply(
        lambda row: get_consensus_label(row.tolist()), axis=1
    )

    for col in annotator_cols:
        ann_labels = df[col]

        # Agreement with consensus
        agreement_with_consensus = (ann_labels == consensus_labels).mean()

        # Self-consistency: how often does this annotator agree with the majority?
        # (same as agreement with consensus in this setup)
        agreement_rate = float(agreement_with_consensus)

        # Consistency: inverse of variance (how uniform are their labels?)
        label_counts = Counter(ann_labels.tolist())
        n_labels = len(annotator_cols[0])  # placeholder
        entropy = 0.0
        for cnt in label_counts.values():
            p = cnt / n_tracks
            if p > 0:
                entropy -= p * np.log(p + 1e-10)
        max_entropy = np.log(len(label_counts) + 1e-10)
        consistency = float(1 - (entropy / (max_entropy + 1e-10))) if max_entropy > 0 else 1.0
        consistency = np.clip(consistency, 0, 1)

        # Edge case contribution: proportion of this annotator's labels
        # that differ from consensus on flagged tracks
        edge_cases = flag_edge_cases(df, annotator_cols, threshold=0.5)
        flagged = edge_cases[edge_cases["flagged"]]
        if len(flagged) > 0:
            edge_contribution = (
                flagged[col] != consensus_labels[flagged.index]
            ).mean()
        else:
            edge_contribution = 0.0

        # Confidence: average fraction of annotators who agree with this annotator
        confidence_scores = []
        for _, row in df.iterrows():
            others = [row[c] for c in annotator_cols if c != col]
            ann_val = row[col]
            agree_pct = sum(1 for o in others if o == ann_val) / len(others) if others else 0
            confidence_scores.append(agree_pct)
        confidence = float(np.mean(confidence_scores))

        stats[col] = {
            "consistency": float(consistency),
            "agreement_rate": float(agreement_rate),
            "edge_case_contribution": float(edge_contribution),
            "confidence": float(confidence)
        }

    return stats
