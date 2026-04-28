# Audio Annotation QA Simulator
### Portfolio Project — Nishtha Sood

A production-grade simulation of an **annotation quality assurance pipeline** for audio content — directly replicating the workflows used by ML/AI-driven platforms like Spotify's Annotation Platform Ops team.

---

## What This Demonstrates

| Skill | Implementation |
|---|---|
| **Annotation QA workflows** | Full pipeline: labeling → agreement → edge case routing → report |
| **Inter-annotator agreement** | Cohen's Kappa (pairwise) + Fleiss' Kappa (multi-rater) |
| **Ground truth definition** | Consensus voting + escalation protocol |
| **Edge case detection** | Disagreement-rate thresholding + severity classification |
| **Annotation guidelines** | Task-specific labeling rules with edge case handling |
| **ML lifecycle awareness** | Data collection → labeling → QA → model training handoff |
| **Human-in-the-loop** | Annotator performance scoring + review queue |
| **SQL-ready output** | CSV exports structured for downstream model training |

---

## Annotation Tasks Supported

1. **Mood Classification** — Happy, Sad, Energetic, Calm, Angry, Romantic
2. **Content Suitability (Age Rating)** — All Ages, Teen+, Mature, Explicit
3. **Genre Tagging** — Pop, Hip-Hop, Rock, Electronic, R&B, Jazz/Blues
4. **Podcast Topic Classification** — News, Comedy, True Crime, Education, Health/Wellness, Business
5. **Audiobook Genre** — Fiction, Non-Fiction, Self-Help, Thriller/Mystery, Sci-Fi/Fantasy, Biography

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd spotify_annotation_qa

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## App Features

### 📊 Agreement Analysis Tab
- **Fleiss' Kappa** — overall multi-rater agreement score with interpretation
- **Pairwise Cohen's Kappa heatmap** — see which annotator pairs diverge
- **Agreement distribution histogram** — visualize where edge cases cluster
- **Per-label agreement bar chart** — identify which categories are most ambiguous

### 🚩 Edge Case Review Tab
- Flagged tracks ordered by disagreement severity (High / Medium / Low)
- Suggested ground truth via consensus voting
- Edge case rate by genre — surfaces which content types are hardest to label

### 👥 Annotator Performance Tab
- **Radar chart** — consistency, agreement rate, edge case contribution, confidence
- **QA score** — composite metric for annotator quality assessment
- Identifies annotators who may need additional guideline training

### 📋 Annotation Guidelines Tab
- Structured labeling rules with decision logic per label
- Edge case handling protocol (5-step escalation flow)
- Ground truth confidence levels (High/Medium/Low → Accept/Review/Escalate)
- Live label distribution pie chart

### 📁 QA Report Export Tab
- Download full annotated dataset (CSV)
- Download edge cases only (CSV)
- Download QA summary metrics (JSON)
- Preview of flagged items

---

## Project Architecture

```
spotify_annotation_qa/
├── app.py                  # Main Streamlit application
├── data_generator.py       # Synthetic annotation dataset generation
├── qa_engine.py            # Agreement metrics, edge case detection, stats
├── report_generator.py     # CSV/JSON export utilities
├── requirements.txt
└── README.md
```

## Skills Demonstrated

`Python` · `Pandas` · `scikit-learn` · `Streamlit` · `Plotly` · `Statistical Analysis` · `Data Quality` · `QA Frameworks` · `Inter-Annotator Agreement` · `Ground Truth Definition` · `Human-in-the-Loop` · `ML Lifecycle` · `SQL-ready Data Export` · `Cross-functional Communication`

---
