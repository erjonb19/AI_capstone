# Sorvex 360 Prediction API — v1.2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from google.cloud import storage
import pandas as pd
import numpy as np
import joblib
import os
import tempfile

app = FastAPI(
    title="Sorvex 360 Prediction API",
    description="Predicts retention, safety, and promotion risk for utility workforce candidates",
    version="1.0.0"
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID  = os.getenv("PROJECT_ID", "sorvex360-493312")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sorvex360-raw-data")
REGION      = os.getenv("REGION", "us-central1")

# ── Model cache — loaded lazily on first request ───────────────────────────────
models = {}

def load_models():
    global models
    if models:
        return
    print("Loading models from GCS...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    model_files = {
        "retention":  "models/retention/model.joblib",
        "safety":     "models/safety/model.joblib",
        "promotion":  "models/promotion/model.joblib",
    }
    for name, gcs_path in model_files.items():
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            bucket.blob(gcs_path).download_to_filename(tmp.name)
            models[name] = joblib.load(tmp.name)
        print(f"  Loaded: {name}")

# ── Request schema ────────────────────────────────────────────────────────────
class CandidateProfile(BaseModel):
    SOC_Code:                str   = Field(..., example="49-9051.00")
    Age:                     int   = Field(..., example=35)
    Gender:                  str   = Field(..., example="Male")
    State:                   str   = Field(..., example="TX")
    EducationLevel:          str   = Field(..., example="Associate Technical")
    CognitiveScore:          int   = Field(..., ge=50, le=100, example=72)
    SimulationScore:         int   = Field(..., ge=50, le=100, example=70)
    BehavioralScore:         int   = Field(..., ge=50, le=100, example=68)
    SituationalScore:        int   = Field(..., ge=50, le=100, example=67)
    Sorvex360PI_Score:       float = Field(..., example=66.5)
    HasPriorTradeExperience: int   = Field(..., ge=0, le=1, example=1)
    LongestJobTenure:        float = Field(..., example=4.5)
    CDL_Status:              int   = Field(..., ge=0, le=1, example=0)
    VeteranStatus:           int   = Field(..., ge=0, le=1, example=0)
    ApprenticeshipInterest:  int   = Field(..., ge=0, le=1, example=1)
    HasValidLicense:         int   = Field(..., ge=0, le=1, example=1)
    CanPassDrugScreen:       int   = Field(..., ge=0, le=1, example=1)
    CanPassBackgroundCheck:  int   = Field(..., ge=0, le=1, example=1)
    MostRecentIndustry:      str   = Field(..., example="Construction")
    TrainingSource:          str   = Field(..., example="IBEW JATC")
    SourceOfCandidate:       str   = Field(..., example="WorkforceProgram")
    TotalTrainingHours:             Optional[int]   = Field(default=6400)
    AttendanceRate:                 Optional[float] = Field(default=0.80)
    PassedRequiredModules:          Optional[int]   = Field(default=1)
    CertificationsEarned:           Optional[int]   = Field(default=2)
    SimulationPerformance:          Optional[int]   = Field(default=65)
    SafetyCommitmentScore:          Optional[float] = Field(default=3.0)
    TeamworkScore:                  Optional[float] = Field(default=3.2)
    ReliabilityScore:               Optional[float] = Field(default=3.3)
    PhysicalTestResult:             Optional[int]   = Field(default=1)
    Lift50lbsTest:                  Optional[int]   = Field(default=1)
    Completed:                      Optional[int]   = Field(default=1)
    ReadinessDelta:                 Optional[float] = Field(default=5.0)
    Sorvex360PI_Score_AtCompletion: Optional[float] = Field(default=None)
    UnionStatus:                    Optional[int]   = Field(default=0)
    EmploymentType:                 Optional[str]   = Field(default="Full-Time")
    RoleRequires_CDL:               Optional[int]   = Field(default=0)
    RoleRequires_OSHA10:            Optional[int]   = Field(default=0)
    RoleRequires_CPR:               Optional[int]   = Field(default=0)
    PreHire_Verified_MVR:           Optional[int]   = Field(default=1)
    PreHire_Verified_DrugScreen:    Optional[int]   = Field(default=1)
    PreHire_Verified_Background:    Optional[int]   = Field(default=1)
    Orientation_LOTO_Completed:     Optional[int]   = Field(default=1)
    Orientation_PPE_Fitted:         Optional[int]   = Field(default=1)
    Apprenticeship_Registered:      Optional[int]   = Field(default=0)
    Sorvex360PI_ScoreAtHire:        Optional[float] = Field(default=None)

# ── Risk tier logic ───────────────────────────────────────────────────────────
def get_risk_tier(probability: float, outcome: str) -> str:
    if outcome == "retention":
        if probability >= 0.75: return "Low"
        if probability >= 0.50: return "Medium"
        return "High"
    else:
        if probability >= 0.60: return "High"
        if probability >= 0.35: return "Medium"
        return "Low"

def get_risk_score(probability: float, outcome: str) -> int:
    if outcome == "retention":
        return int(probability * 100)
    else:
        return int((1 - probability) * 100)

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "version": "1.0.0"
    }

@app.get("/")
def root():
    return {
        "service": "Sorvex 360 Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict"
    }

# ── Prediction endpoint ────────────────────────────────────────────────────────
@app.post("/predict")
def predict(candidate: CandidateProfile):
    load_models()

    pi_at_completion = candidate.Sorvex360PI_Score_AtCompletion or candidate.Sorvex360PI_Score
    pi_at_hire       = candidate.Sorvex360PI_ScoreAtHire or pi_at_completion

    features = {
        "Age":                            candidate.Age,
        "CognitiveScore":                 candidate.CognitiveScore,
        "SimulationScore":                candidate.SimulationScore,
        "BehavioralScore":                candidate.BehavioralScore,
        "SituationalScore":               candidate.SituationalScore,
        "Sorvex360PI_Score":              candidate.Sorvex360PI_Score,
        "HasPriorTradeExperience":        candidate.HasPriorTradeExperience,
        "LongestJobTenure":               candidate.LongestJobTenure,
        "CDL_Status":                     candidate.CDL_Status,
        "VeteranStatus":                  candidate.VeteranStatus,
        "ApprenticeshipInterest":         candidate.ApprenticeshipInterest,
        "CanPassDrugScreen":              candidate.CanPassDrugScreen,
        "CanPassBackgroundCheck":         candidate.CanPassBackgroundCheck,
        "OSHA10_Status":                  0,
        "CPR_Status":                     0,
        "TotalTrainingHours":             candidate.TotalTrainingHours,
        "AttendanceRate":                 candidate.AttendanceRate,
        "PassedRequiredModules":          candidate.PassedRequiredModules,
        "CertificationsEarned":           candidate.CertificationsEarned,
        "SimulationPerformance":          candidate.SimulationPerformance,
        "SafetyCommitmentScore":          candidate.SafetyCommitmentScore,
        "TeamworkScore":                  candidate.TeamworkScore,
        "ReliabilityScore":               candidate.ReliabilityScore,
        "PhysicalTestResult":             candidate.PhysicalTestResult,
        "Lift50lbsTest":                  candidate.Lift50lbsTest,
        "Completed":                      candidate.Completed,
        "ReadinessDelta":                 candidate.ReadinessDelta,
        "Sorvex360PI_Score_AtCompletion": pi_at_completion,
        "UnionStatus":                    candidate.UnionStatus,
        "RoleRequires_CDL":               candidate.RoleRequires_CDL,
        "RoleRequires_OSHA10":            candidate.RoleRequires_OSHA10,
        "RoleRequires_CPR":               candidate.RoleRequires_CPR,
        "PreHire_Verified_MVR":           candidate.PreHire_Verified_MVR,
        "PreHire_Verified_DrugScreen":    candidate.PreHire_Verified_DrugScreen,
        "PreHire_Verified_Background":    candidate.PreHire_Verified_Background,
        "Orientation_LOTO_Completed":     candidate.Orientation_LOTO_Completed,
        "Orientation_PPE_Fitted":         candidate.Orientation_PPE_Fitted,
        "Apprenticeship_Registered":      candidate.Apprenticeship_Registered,
        "Sorvex360PI_ScoreAtHire":        pi_at_hire,
        "SOC_Code":                       candidate.SOC_Code,
        "Gender":                         candidate.Gender,
        "EducationLevel":                 candidate.EducationLevel,
        "MostRecentIndustry":             candidate.MostRecentIndustry,
        "TrainingSource":                 candidate.TrainingSource,
        "SourceOfCandidate":              candidate.SourceOfCandidate,
        "EmploymentType":                 candidate.EmploymentType,
        "HasValidLicense":                str(candidate.HasValidLicense),
    }

    X = pd.DataFrame([features])

    model_map = {
        "retention":  ("Tenure_1Year",            "Will candidate stay 1+ year?"),
        "safety":     ("OSHA_Recordable_Incident", "OSHA incident likelihood"),
        "promotion":  ("PromotionWithin24Months",  "Promotion within 24 months"),
    }

    predictions = {}
    for model_name, (outcome_label, description) in model_map.items():
        model  = models[model_name]
        proba  = float(model.predict_proba(X)[0][1])
        tier   = get_risk_tier(proba, model_name)
        score  = get_risk_score(proba, model_name)
        predictions[model_name] = {
            "outcome":     outcome_label,
            "description": description,
            "probability": round(proba, 4),
            "risk_tier":   tier,
            "score":       score,
        }

    retention_score = predictions["retention"]["score"]
    safety_score    = predictions["safety"]["score"]
    promotion_score = predictions["promotion"]["score"]
    composite_score = round((retention_score * 0.5 + safety_score * 0.3 + promotion_score * 0.2), 1)

    if composite_score >= 70:
        overall_tier = "Low Risk"
    elif composite_score >= 50:
        overall_tier = "Medium Risk"
    else:
        overall_tier = "High Risk"

    return {
        "candidate_summary": {
            "soc_code":      candidate.SOC_Code,
            "pi_score":      candidate.Sorvex360PI_Score,
            "overall_score": composite_score,
            "overall_tier":  overall_tier,
        },
        "predictions": predictions,
        "model_version": "v1.0",
    }
