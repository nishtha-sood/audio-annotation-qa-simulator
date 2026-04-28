"""
Report generator: QA export utilities.
"""

import pandas as pd
import json
import io
from typing import List, Dict
from datetime import datetime
from collections import Counter


def generate_qa_report_csv(
    df: pd.DataFrame,
    annotator_cols: List[str],
    edge_cases: pd.DataFrame
) -> bytes:
    """Generate a CSV QA report with consensus labels and disagreement flags."""
    result = df.copy()

    # Add consensus
    from qa_engine import get_consensus_label
    result["consensus_label"] = result[annotator_cols].apply(
        lambda row: get_consensus_label(row.tolist()), axis=1
    )

    # Add disagreement rate if available
    if "disagreement_rate" in edge_cases.columns:
        if "disagreement_rate" not in result.columns:
            result["disagreement_rate"] = edge_cases["disagreement_rate"]
    if "flagged" in edge_cases.columns:
        if "flagged" not in result.columns:
            result["flagged"] = edge_cases["flagged"]

    result["qa_timestamp"] = datetime.now().isoformat()

    buf = io.BytesIO()
    result.to_csv(buf, index=False)
    return buf.getvalue()


def generate_summary_stats(
    df: pd.DataFrame,
    annotator_cols: List[str],
    edge_cases: pd.DataFrame,
    fleiss_kappa: float,
    task_config: dict
) -> dict:
    """Generate a structured summary stats dict for JSON export."""
    from qa_engine import compute_pairwise_kappa, compute_category_stats

    n_flagged = int(edge_cases["flagged"].sum()) if "flagged" in edge_cases.columns else 0
    kappa_matrix = compute_pairwise_kappa(df, annotator_cols)
    cat_stats = compute_category_stats(df, annotator_cols, task_config["labels"])

    # Pairwise kappas
    n = len(annotator_cols)
    pairwise = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairwise[f"ann{i+1}_vs_ann{j+1}"] = round(float(kappa_matrix[i][j]), 4)

    overall_agreement = float(
        df[annotator_cols].apply(
            lambda row: (row == row.mode()[0]).mean(), axis=1
        ).mean()
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "task": task_config.get("description", "Unknown"),
        "dataset": {
            "total_tracks": len(df),
            "n_annotators": len(annotator_cols),
            "label_space": task_config["labels"]
        },
        "agreement_metrics": {
            "fleiss_kappa": round(fleiss_kappa, 4),
            "overall_agreement_rate": round(overall_agreement, 4),
            "pairwise_cohen_kappa": pairwise
        },
        "edge_cases": {
            "total_flagged": n_flagged,
            "flag_rate": round(n_flagged / len(df), 4) if len(df) > 0 else 0
        },
        "category_breakdown": {
            label: {
                "avg_agreement": round(v["avg_agreement"], 4),
                "n_tracks": v["n_tracks"]
            }
            for label, v in cat_stats.items()
        }
    }
