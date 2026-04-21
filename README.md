# Sorvex 360 — AI-Powered Utility Workforce Prediction Pipeline

> **Predict • Prepare • Place • Post-Hire Outcomes**

Sorvex 360 is an end-to-end machine learning pipeline that generates synthetic utility workforce candidate profiles and trains predictive models to identify retention risk, safety risk, and promotion likelihood before a candidate is hired.

---

## 🎯 Project Overview

The utility industry faces a critical workforce challenge — high turnover, aging workforce, and difficulty predicting which candidates will succeed in skilled trades roles. Sorvex 360 addresses this by:

1. **Generating** a statistically grounded synthetic dataset of 10,000 utility workforce candidate profiles across 6 SOC codes using pure rule-based generation calibrated to BLS, ONET, OSHA, and WA L&I data sources
2. **Training** HistGradientBoosting models to predict 3 key post-hire outcomes
3. **Serving** predictions through a REST API that returns Low/Medium/High risk tiers per candidate
4. **Storing** all data in BigQuery and registering models in Vertex AI for production use

---

## 🏗️ Architecture

```
Data Sources (BLS, ONET, OSHA, WA L&I, USEER)
        ↓
Synthetic Pipeline (Pure Rule-Based Statistical Generation)
        ↓
10,000 Candidate Profiles (4-Phase Schema)
        ↓
Google Cloud Storage → BigQuery
        ↓
Vertex AI Workbench (Model Training)
        ↓
Vertex AI Model Registry (3 Models)
        ↓
FastAPI on Cloud Run (/predict endpoint)
        ↓
Looker Dashboard (Front End)
```

---

## 📊 Four-Phase Data Schema

| Phase | Name | Records | Key Fields |
|---|---|---|---|
| 1 | **Predict** | 10,000 | Demographics, Assessment Scores, PI Score |
| 2 | **Prepare** | 10,000 | Training Hours, Attendance, Certifications |
| 3 | **Place** | ~6,300 | Employer, Employment Type, Compliance |
| 4 | **Post-Hire** | ~6,300 | Tenure, Safety Incidents, Promotions |

Phase 3 and 4 records cover only candidates who completed training (~63% completion rate per WA L&I benchmarks). All phases joined on `CandidateID` → `PlacementID`.

---

## 🤖 Models

Three binary classifiers trained using HistGradientBoostingClassifier with class-weight balancing and optimal F1 threshold tuning:

| Model | Target | AUC-ROC | CV AUC | Threshold |
|---|---|---|---|---|
| Retention | `Tenure_1Year` | 0.865 | 0.861 ± 0.008 | 0.337 |
| Safety | `OSHA_Recordable_Incident` | 0.774 | 0.776 ± 0.018 | 0.154 |
| Promotion | `PromotionWithin24Months` | 0.743 | 0.746 ± 0.004 | 0.188 |

**Key predictors:**
- Retention: `Sorvex360PI_ScoreAtHire`, `AttendanceRate`, `TotalTrainingHours`, `UnionStatus`
- Safety: `SafetyCommitmentScore` (dominant), `SOC_Code`, `Orientation_LOTO_Completed`
- Promotion: `AttendanceRate`, `TotalTrainingHours`, `Sorvex360PI_Score`

> **Note on safety model:** The OSHA model is optimized for recall over precision — it is intentionally designed to over-flag candidates as safety risks rather than miss a true incident. At 4.7% positive rate this is the appropriate tradeoff for a safety-critical application.

---

## 🔌 API

**POST `/predict`** — Takes a candidate profile, returns Low/Medium/High risk tiers

```json
{
  "candidate_summary": {
    "soc_code": "49-9051.00",
    "pi_score": 66.5,
    "overall_score": 72.3,
    "overall_tier": "Low Risk"
  },
  "predictions": {
    "retention":  { "risk_tier": "Low",    "score": 81 },
    "safety":     { "risk_tier": "Low",    "score": 74 },
    "promotion":  { "risk_tier": "Medium", "score": 58 }
  }
}
```

---

## ☁️ GCP Infrastructure

| Service | Purpose |
|---|---|
| Cloud Storage | Raw data, validated files, model artifacts |
| BigQuery | `sorvex_raw`, `sorvex_staging`, `sorvex_ml`, `sorvex_logs` datasets |
| Vertex AI Workbench | Model training environment |
| Vertex AI Model Registry | 3 registered models (retention, safety, promotion) |
| Cloud Run | FastAPI prediction service |

---

## 📁 Repo Structure

```
AI_capstone/
├── api/
│   ├── main.py            # FastAPI prediction service
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Cloud Run deployment
├── notebooks/
│   ├── Sorvex360_Synthetic_Pipeline_V3.ipynb  # 10,000 record rule-based generation
│   ├── Sorvex360_VertexAI_Training.ipynb      # Model training + Vertex AI registration
│   ├── Sorvex360_Synthetic_Pipeline.py        # .py version of synthetic pipeline
│   ├── Sorvex360_VertexAI_Training.py         # .py version of training pipeline
│   ├── Sorvex360_OSHA_EDA.py                  # OSHA severe injury EDA
│   └── Sorvex360_FactoryWorker_EDA.py         # Factory worker behavior EDA
└── docs/
    └── Sorvex360_Blueprint.docx               # Full field-by-field data blueprint
```

---

## 🛠️ Tech Stack

- **Python** — pandas, scikit-learn, FastAPI
- **Google Cloud Platform** — Cloud Storage, BigQuery, Vertex AI, Cloud Run
- **Data Sources** — BLS, ONET, OSHA, WA L&I Apprenticeship, USEER, CPWR
- **ML** — HistGradientBoostingClassifier with permutation importance analysis

---

## 📈 Data Sources

| Source | Purpose |
|---|---|
| BLS Tenure Tables 2024 | Tenure distributions by age, education, occupation |
| USEER 2025 | Utility workforce demographic distributions |
| ONET (6 SOC codes) | Skills and knowledge profiles per utility role |
| WA L&I Apprenticeship | 131K apprentice records — training hours and completion rates |
| OSHA Severe Injury | 3,295 utility incident records — safety parameters |
| CareerBuilder (Strict) | 257 utility job postings — employer and role distributions |

---

## 📋 SOC Codes Covered

| SOC Code | Role | Share |
|---|---|---|
| 49-2022.00 | Telecom Equipment Installer/Repairer | 28% |
| 49-9051.00 | Power Line Installer/Repairer | 20% |
| 51-8013.00 | Power Plant Operator | 16% |
| 51-8031.00 | Water/Wastewater Treatment Plant Operator | 16% |
| 51-8092.00 | Gas Plant Operator | 12% |
| 43-5041.00 | Meter Reader, Utilities | 8% |

---

## 🚀 Running the API Locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

API docs available at `http://localhost:8080/docs`

---

## 📊 Synthetic Data Design Notes

The dataset uses pure rule-based generation — no generative AI or GAN-based synthesis. Every distribution, correlation, and outcome rate is directly calibrated to real-world data sources:

- Assessment score distributions by SOC code from ONET
- Training hours and completion rates from WA L&I 131K apprentice records
- OSHA incident rates by SOC from OSHA Severe Injury dataset
- Tenure distributions from BLS 2024 tenure tables by age band
- Demographic distributions from USEER 2025 utility workforce report

This approach ensures all ML signal chains are intentional, auditable, and free of GAN noise artifacts.

---

*Built as an AI/ML capstone project — May 2026*
