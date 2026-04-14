# Sorvex 360 — AI-Powered Utility Workforce Prediction Pipeline

> **Predict • Prepare • Place • Post-Hire Outcomes**

Sorvex 360 is an end-to-end machine learning pipeline that generates synthetic utility workforce candidate profiles and trains predictive models to identify retention risk, safety risk, and promotion likelihood before a candidate is hired.

---

## 🎯 Project Overview

The utility industry faces a critical workforce challenge — high turnover, aging workforce, and difficulty predicting which candidates will succeed in skilled trades roles. Sorvex 360 addresses this by:

1. **Generating** a statistically grounded synthetic dataset of 5,000 utility workforce candidate profiles across 6 SOC codes
2. **Training** gradient boosting models to predict 3 key post-hire outcomes
3. **Serving** predictions through a REST API that returns Low/Medium/High risk tiers per candidate
4. **Storing** all data in BigQuery and registering models in Vertex AI for production use

---

## 🏗️ Architecture

```
Data Sources (BLS, ONET, OSHA, WA L&I, USEER)
        ↓
Synthetic Pipeline (CTGAN + Statistical Seed Generation)
        ↓
5,000 Candidate Profiles (4-Phase Schema)
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
| 1 | **Predict** | 5,000 | Demographics, Assessment Scores, PI Score |
| 2 | **Prepare** | 5,000 | Training Hours, Attendance, Certifications |
| 3 | **Place** | 5,000 | Employer, Employment Type, Compliance |
| 4 | **Post-Hire** | 5,000 | Tenure, Safety Incidents, Promotions |

All phases joined on `CandidateID` → `PlacementID`.

---

## 🤖 Models

Three binary classifiers trained using Gradient Boosting with class-weight balancing:

| Model | Target | AUC-ROC | Description |
|---|---|---|---|
| Retention | `Tenure_1Year` | 0.70 | Will candidate stay 12+ months? |
| Safety | `OSHA_Recordable_Incident` | 0.53 | OSHA incident likelihood |
| Promotion | `PromotionWithin24Months` | 0.55 | Promotion within 24 months |

**Key features:** `Sorvex360PI_Score`, `Completed`, `AttendanceRate`, `SafetyCommitmentScore`, `TotalTrainingHours`, `ReadinessDelta`

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
│   ├── Sorvex360_Synthetic_Pipeline.py    # 5,000 record generation (CTGAN)
│   ├── Sorvex360_VertexAI_Training.py     # Model training + Vertex AI registration
│   ├── Sorvex360_OSHA_EDA.py              # OSHA severe injury EDA
│   └── Sorvex360_FactoryWorker_EDA.py     # Factory worker behavior EDA
└── docs/
    └── Sorvex360_Blueprint.docx           # Full field-by-field data blueprint
```

---

## 🛠️ Tech Stack

- **Python** — pandas, scikit-learn, CTGAN (SDV), FastAPI
- **Google Cloud Platform** — Cloud Storage, BigQuery, Vertex AI, Cloud Run
- **Data Sources** — BLS, ONET, OSHA, WA L&I Apprenticeship, USEER, CPWR
- **ML** — Gradient Boosting Classifier, CTGAN synthetic data generation

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

## 🚀 Running the API Locally

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

API docs available at `http://localhost:8080/docs`

---

*Built as an AI/ML capstone project — May 2026*
