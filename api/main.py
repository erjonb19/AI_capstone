# Sorvex 360 — v3.0
# Adds: multi-page app, Jinja2 templates, session auth, role-based access
# Keeps: all v2.1 prediction endpoints unchanged

from fastapi import FastAPI, HTTPException, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from google.cloud import storage
import pandas as pd
import numpy as np
import joblib
import shap
import os
import tempfile

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sorvex 360",
    description="Predictive Workforce Intelligence — Utilities Sector",
    version="3.0.0"
)

# Session middleware — secret key for signing cookies
SECRET_KEY = os.getenv("SECRET_KEY", "sorvex360-session-secret-change-in-prod")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=86400)

# CORS — keep open for existing frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID  = os.getenv("PROJECT_ID",  "sorvex360-493312")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sorvex360-raw-data")
REGION      = os.getenv("REGION",      "us-central1")
API_BASE    = os.getenv("API_BASE",    "https://sorvex360-predict-839675931790.us-central1.run.app")

# ── Hardcoded users ───────────────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "sorvex360",  "role": "admin",  "client": "Utility Corp A"},
    "viewer":  {"password": "viewer123",  "role": "viewer", "client": "Utility Corp A"},
    "clientb": {"password": "clientb123", "role": "viewer", "client": "Utility Corp B"},
}

# ── Auth helpers ──────────────────────────────────────────────────────────────
def get_session(request: Request) -> dict:
    return request.session.get("user", {})

def require_auth(request: Request) -> Optional[RedirectResponse]:
    """Returns a redirect if not authenticated, else None."""
    if not request.session.get("user"):
        return RedirectResponse(url="/login", status_code=302)
    return None

def require_admin(request: Request) -> Optional[RedirectResponse]:
    """Returns a redirect if not admin."""
    user = request.session.get("user", {})
    if not user:
        return RedirectResponse(url="/login", status_code=302)
    if user.get("role") != "admin":
        return RedirectResponse(url="/dashboard", status_code=302)
    return None

def render_page(request: Request, template: str, active_page: str, **kwargs):
    """Helper to render a protected page with common context."""
    return templates.TemplateResponse(template, {
        "request":      request,
        "session":      get_session(request),
        "active_page":  active_page,
        "api_base":     API_BASE,
        **kwargs
    })

# ── Model + SHAP cache ────────────────────────────────────────────────────────
models          = {}
shap_explainers = {}

def load_models():
    global models, shap_explainers
    if models:
        return
    print("Loading models from GCS...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    model_files = {
        "retention": "models/retention/model.joblib",
        "safety":    "models/safety/model.joblib",
        "promotion": "models/promotion/model.joblib",
    }
    for name, gcs_path in model_files.items():
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            bucket.blob(gcs_path).download_to_filename(tmp.name)
            models[name] = joblib.load(tmp.name)
        print(f"  Loaded: {name}")

    print("Building SHAP explainers...")
    background = _build_background_data()
    for name, pipeline in models.items():
        try:
            preprocessor     = pipeline.named_steps['preprocessor']
            classifier       = pipeline.named_steps['classifier']
            X_bg_transformed = preprocessor.transform(background)
            explainer        = shap.TreeExplainer(classifier, data=X_bg_transformed)
            shap_explainers[name] = (explainer, preprocessor)
            print(f"  SHAP ready: {name}")
        except Exception as e:
            print(f"  SHAP failed for {name}: {e}")
            shap_explainers[name] = None


def _build_background_data() -> pd.DataFrame:
    np.random.seed(360)
    n = 50
    soc_codes = np.random.choice(
        ['49-2022.00','49-9051.00','51-8013.00','51-8031.00','51-8092.00','43-5041.00'],
        n, p=[0.28, 0.20, 0.16, 0.16, 0.12, 0.08]
    )
    df = pd.DataFrame({
        "Age":                            np.random.normal(43, 10, n).clip(18, 60).astype(int),
        "CognitiveScore":                 np.random.normal(70, 9, n).clip(50, 100).astype(int),
        "SimulationScore":                np.random.normal(71, 9, n).clip(50, 100).astype(int),
        "BehavioralScore":                np.random.normal(69, 9, n).clip(50, 100).astype(int),
        "SituationalScore":               np.random.normal(69, 8, n).clip(50, 100).astype(int),
        "Sorvex360PI_Score":              np.random.normal(70, 8, n).clip(0, 100),
        "HasPriorTradeExperience":        np.random.binomial(1, 0.72, n),
        "LongestJobTenure":               np.random.lognormal(1.5, 0.6, n).clip(0.1, 35),
        "CDL_Status":                     np.random.binomial(1, 0.12, n),
        "VeteranStatus":                  np.random.binomial(1, 0.12, n),
        "ApprenticeshipInterest":         np.random.binomial(1, 0.55, n),
        "CanPassDrugScreen":              np.random.binomial(1, 0.88, n),
        "CanPassBackgroundCheck":         np.random.binomial(1, 0.85, n),
        "OSHA10_Status":                  np.zeros(n, dtype=int),
        "CPR_Status":                     np.zeros(n, dtype=int),
        "TotalTrainingHours":             np.random.normal(5500, 1500, n).clip(500, 10000).astype(int),
        "AttendanceRate":                 np.random.beta(7, 2, n).clip(0.50, 0.99),
        "PassedRequiredModules":          np.random.binomial(1, 0.60, n),
        "CertificationsEarned":           np.random.poisson(2.5, n).clip(0, 6),
        "SimulationPerformance":          np.random.normal(63, 9, n).clip(0, 100).astype(int),
        "SafetyCommitmentScore":          np.random.normal(3.1, 0.65, n).clip(1, 5),
        "TeamworkScore":                  np.random.normal(3.2, 0.7, n).clip(1, 5),
        "ReliabilityScore":               np.random.normal(3.3, 0.65, n).clip(1, 5),
        "PhysicalTestResult":             np.random.binomial(1, 0.82, n),
        "Lift50lbsTest":                  np.random.binomial(1, 0.85, n),
        "Completed":                      np.ones(n, dtype=int),
        "ReadinessDelta":                 np.random.normal(5, 5, n),
        "Sorvex360PI_Score_AtCompletion": np.random.normal(75, 8, n).clip(0, 100),
        "UnionStatus":                    np.random.binomial(1, 0.18, n),
        "RoleRequires_CDL":               np.random.binomial(1, 0.12, n),
        "RoleRequires_OSHA10":            np.random.binomial(1, 0.18, n),
        "RoleRequires_CPR":               np.random.binomial(1, 0.12, n),
        "PreHire_Verified_MVR":           np.random.binomial(1, 0.88, n),
        "PreHire_Verified_DrugScreen":    np.random.binomial(1, 0.92, n),
        "PreHire_Verified_Background":    np.random.binomial(1, 0.90, n),
        "Orientation_LOTO_Completed":     np.random.binomial(1, 0.65, n),
        "Orientation_PPE_Fitted":         np.random.binomial(1, 0.90, n),
        "Apprenticeship_Registered":      np.random.binomial(1, 0.35, n),
        "Sorvex360PI_ScoreAtHire":        np.random.normal(75, 8, n).clip(0, 100),
        "SOC_Code":                       soc_codes,
        "Gender":                         np.random.choice(['Male','Female','Other'], n, p=[0.731,0.259,0.010]),
        "EducationLevel":                 np.random.choice(
                                              ['HS/GED','Vocational Certificate','Associate Technical',
                                               'Associate General',"Bachelor's",'Apprenticeship'],
                                              n, p=[0.32,0.18,0.17,0.06,0.08,0.19]),
        "MostRecentIndustry":             np.random.choice(
                                              ['Utility','Construction','Manufacturing','Military','Other'],
                                              n, p=[0.35,0.25,0.18,0.12,0.10]),
        "TrainingSource":                 np.random.choice(
                                              ['IBEW JATC','Trade Union','Community College',
                                               'Employer Program','Military','Self-taught'],
                                              n, p=[0.28,0.20,0.18,0.15,0.12,0.07]),
        "SourceOfCandidate":              np.random.choice(
                                              ['WorkforceProgram','SelfReferred','TradeUnion',
                                               'MilitaryTransition','CommunityCollege','Other'],
                                              n, p=[0.25,0.22,0.20,0.12,0.13,0.08]),
        "EmploymentType":                 np.random.choice(['Full-Time','Part-Time','Contractor'],
                                              n, p=[0.90,0.07,0.03]),
        "HasValidLicense":                np.random.choice(['0','1'], n, p=[0.22,0.78]),
    })
    return df


# ── Cohort percentile ─────────────────────────────────────────────────────────
SOC_SCORE_PARAMS = {
    '49-9051.00': {'mean': 64.2, 'std': 11.8},
    '49-2022.00': {'mean': 67.5, 'std': 11.2},
    '51-8013.00': {'mean': 71.3, 'std': 10.5},
    '51-8031.00': {'mean': 70.1, 'std': 10.8},
    '51-8092.00': {'mean': 69.8, 'std': 11.0},
    '43-5041.00': {'mean': 65.4, 'std': 12.1},
}

def get_cohort_percentile(composite_score: float, soc_code: str) -> dict:
    from scipy import stats
    params = SOC_SCORE_PARAMS.get(soc_code, {'mean': 68.0, 'std': 11.5})
    percentile = int(stats.norm.cdf(composite_score, params['mean'], params['std']) * 100)
    percentile = max(1, min(99, percentile))
    if percentile >= 75:   label = "Top 25%"
    elif percentile >= 50: label = "Above Average"
    elif percentile >= 25: label = "Below Average"
    else:                  label = "Bottom 25%"
    return {
        "percentile":  percentile,
        "label":       label,
        "soc_mean":    round(params['mean'], 1),
        "description": f"Scores higher than {percentile}% of {_soc_label(soc_code)} candidates"
    }

def _soc_label(soc_code: str) -> str:
    return {
        '49-9051.00': 'Power Line Installer',
        '49-2022.00': 'Telecom Installer',
        '51-8013.00': 'Power Plant Operator',
        '51-8031.00': 'Water Treatment Operator',
        '51-8092.00': 'Gas Plant Operator',
        '43-5041.00': 'Meter Reader',
    }.get(soc_code, 'utility worker')


# ── SHAP factor extraction ────────────────────────────────────────────────────
FEATURE_LABELS = {
    "AttendanceRate":"Attendance Rate","SafetyCommitmentScore":"Safety Commitment",
    "Sorvex360PI_Score":"PI Score","Sorvex360PI_ScoreAtHire":"PI Score at Hire",
    "TotalTrainingHours":"Training Hours","CognitiveScore":"Cognitive Score",
    "BehavioralScore":"Behavioral Score","SituationalScore":"Situational Score",
    "SimulationScore":"Simulation Score","LongestJobTenure":"Job Tenure History",
    "Age":"Age","HasPriorTradeExperience":"Prior Trade Experience",
    "CertificationsEarned":"Certifications Earned","ReadinessDelta":"Readiness Change",
    "UnionStatus":"Union Status","PassedRequiredModules":"Passed Training Modules",
    "ReliabilityScore":"Reliability Score","TeamworkScore":"Teamwork Score",
    "PhysicalTestResult":"Physical Test","Orientation_LOTO_Completed":"LOTO Orientation",
    "Apprenticeship_Registered":"Apprenticeship Registered","VeteranStatus":"Veteran Status",
    "CDL_Status":"CDL License","SimulationPerformance":"Simulation Performance",
    "Completed":"Completed Training",
}

def get_shap_factors(model_name: str, X: pd.DataFrame, n_factors: int = 3) -> list:
    entry = shap_explainers.get(model_name)
    if entry is None:
        return []
    try:
        explainer, preprocessor = entry
        X_transformed = preprocessor.transform(X)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]
        original_feature_names = list(X.columns)
        numeric_features = [c for c in original_feature_names
                            if X[c].dtype in [np.float64, np.float32, np.int64, np.int32, int, float]]
        try:
            n_numeric = len(preprocessor.named_transformers_['num'].transform(
                X[numeric_features]).T)
        except Exception:
            n_numeric = len(numeric_features)
        vals_numeric = vals[:n_numeric]
        top_idx = np.argsort(np.abs(vals_numeric))[::-1][:n_factors]
        factors = []
        for idx in top_idx:
            if idx >= len(numeric_features): continue
            name  = numeric_features[idx]
            value = float(vals_numeric[idx])
            factors.append({
                "feature":    name,
                "label":      FEATURE_LABELS.get(name, name),
                "shap_value": round(value, 4),
                "direction":  "increases" if value > 0 else "decreases",
                "raw_value":  float(X.iloc[0][name]) if isinstance(X.iloc[0][name], (np.integer, np.floating)) else X.iloc[0][name],
            })
        return factors
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")
        return []


# ── Request schemas ───────────────────────────────────────────────────────────
class CandidateProfile(BaseModel):
    SOC_Code:                str   = Field(..., example="49-9051.00")
    Age:                     int   = Field(..., example=35)
    Gender:                  str   = Field(..., example="Male")
    State:                   str   = Field(..., example="TX")
    EducationLevel:          str   = Field(..., example="Associate Technical")
    CognitiveScore:          int   = Field(..., ge=50, le=100)
    SimulationScore:         int   = Field(..., ge=50, le=100)
    BehavioralScore:         int   = Field(..., ge=50, le=100)
    SituationalScore:        int   = Field(..., ge=50, le=100)
    Sorvex360PI_Score:       float
    HasPriorTradeExperience: int   = Field(..., ge=0, le=1)
    LongestJobTenure:        float
    CDL_Status:              int   = Field(..., ge=0, le=1)
    VeteranStatus:           int   = Field(..., ge=0, le=1)
    ApprenticeshipInterest:  int   = Field(..., ge=0, le=1)
    HasValidLicense:         int   = Field(..., ge=0, le=1)
    CanPassDrugScreen:       int   = Field(..., ge=0, le=1)
    CanPassBackgroundCheck:  int   = Field(..., ge=0, le=1)
    MostRecentIndustry:      str
    TrainingSource:          str
    SourceOfCandidate:       str
    TotalTrainingHours:             Optional[int]   = 6400
    AttendanceRate:                 Optional[float] = 0.80
    PassedRequiredModules:          Optional[int]   = 1
    CertificationsEarned:           Optional[int]   = 2
    SimulationPerformance:          Optional[int]   = 65
    SafetyCommitmentScore:          Optional[float] = 3.0
    TeamworkScore:                  Optional[float] = 3.2
    ReliabilityScore:               Optional[float] = 3.3
    PhysicalTestResult:             Optional[int]   = 1
    Lift50lbsTest:                  Optional[int]   = 1
    Completed:                      Optional[int]   = 1
    ReadinessDelta:                 Optional[float] = 5.0
    Sorvex360PI_Score_AtCompletion: Optional[float] = None
    UnionStatus:                    Optional[int]   = 0
    EmploymentType:                 Optional[str]   = "Full-Time"
    RoleRequires_CDL:               Optional[int]   = 0
    RoleRequires_OSHA10:            Optional[int]   = 0
    RoleRequires_CPR:               Optional[int]   = 0
    PreHire_Verified_MVR:           Optional[int]   = 1
    PreHire_Verified_DrugScreen:    Optional[int]   = 1
    PreHire_Verified_Background:    Optional[int]   = 1
    Orientation_LOTO_Completed:     Optional[int]   = 1
    Orientation_PPE_Fitted:         Optional[int]   = 1
    Apprenticeship_Registered:      Optional[int]   = 0
    Sorvex360PI_ScoreAtHire:        Optional[float] = None

class ExplainRequest(BaseModel):
    candidate:    CandidateProfile
    predictions:  Dict[str, Any]
    cohort:       Dict[str, Any]
    shap_factors: Dict[str, Any]

class CompareRequest(BaseModel):
    candidate_a: CandidateProfile
    candidate_b: CandidateProfile


# ── Risk tier/score logic ─────────────────────────────────────────────────────
def get_risk_tier(probability: float, outcome: str) -> str:
    if outcome == "retention":
        if probability >= 0.75: return "Low"
        if probability >= 0.50: return "Medium"
        return "High"
    elif outcome == "safety":
        if probability >= 0.60: return "High"
        if probability >= 0.35: return "Medium"
        return "Low"
    else:
        if probability >= 0.60: return "Low"
        if probability >= 0.35: return "Medium"
        return "High"

def get_risk_score(probability: float, outcome: str) -> int:
    if outcome == "retention": return int(probability * 100)
    elif outcome == "safety":  return int((1 - probability) * 100)
    else:                      return int(probability * 100)


# ── Build feature DataFrame ───────────────────────────────────────────────────
def build_features(candidate: CandidateProfile) -> pd.DataFrame:
    pi_at_completion = candidate.Sorvex360PI_Score_AtCompletion or candidate.Sorvex360PI_Score
    pi_at_hire       = candidate.Sorvex360PI_ScoreAtHire or pi_at_completion
    return pd.DataFrame([{
        "Age": candidate.Age, "CognitiveScore": candidate.CognitiveScore,
        "SimulationScore": candidate.SimulationScore, "BehavioralScore": candidate.BehavioralScore,
        "SituationalScore": candidate.SituationalScore, "Sorvex360PI_Score": candidate.Sorvex360PI_Score,
        "HasPriorTradeExperience": candidate.HasPriorTradeExperience,
        "LongestJobTenure": candidate.LongestJobTenure, "CDL_Status": candidate.CDL_Status,
        "VeteranStatus": candidate.VeteranStatus, "ApprenticeshipInterest": candidate.ApprenticeshipInterest,
        "CanPassDrugScreen": candidate.CanPassDrugScreen, "CanPassBackgroundCheck": candidate.CanPassBackgroundCheck,
        "OSHA10_Status": 0, "CPR_Status": 0,
        "TotalTrainingHours": candidate.TotalTrainingHours, "AttendanceRate": candidate.AttendanceRate,
        "PassedRequiredModules": candidate.PassedRequiredModules, "CertificationsEarned": candidate.CertificationsEarned,
        "SimulationPerformance": candidate.SimulationPerformance, "SafetyCommitmentScore": candidate.SafetyCommitmentScore,
        "TeamworkScore": candidate.TeamworkScore, "ReliabilityScore": candidate.ReliabilityScore,
        "PhysicalTestResult": candidate.PhysicalTestResult, "Lift50lbsTest": candidate.Lift50lbsTest,
        "Completed": candidate.Completed, "ReadinessDelta": candidate.ReadinessDelta,
        "Sorvex360PI_Score_AtCompletion": pi_at_completion, "UnionStatus": candidate.UnionStatus,
        "RoleRequires_CDL": candidate.RoleRequires_CDL, "RoleRequires_OSHA10": candidate.RoleRequires_OSHA10,
        "RoleRequires_CPR": candidate.RoleRequires_CPR, "PreHire_Verified_MVR": candidate.PreHire_Verified_MVR,
        "PreHire_Verified_DrugScreen": candidate.PreHire_Verified_DrugScreen,
        "PreHire_Verified_Background": candidate.PreHire_Verified_Background,
        "Orientation_LOTO_Completed": candidate.Orientation_LOTO_Completed,
        "Orientation_PPE_Fitted": candidate.Orientation_PPE_Fitted,
        "Apprenticeship_Registered": candidate.Apprenticeship_Registered,
        "Sorvex360PI_ScoreAtHire": pi_at_hire,
        "SOC_Code": candidate.SOC_Code, "Gender": candidate.Gender,
        "EducationLevel": candidate.EducationLevel, "MostRecentIndustry": candidate.MostRecentIndustry,
        "TrainingSource": candidate.TrainingSource, "SourceOfCandidate": candidate.SourceOfCandidate,
        "EmploymentType": candidate.EmploymentType, "HasValidLicense": str(candidate.HasValidLicense),
    }])


# ── Run predictions ───────────────────────────────────────────────────────────
def run_predictions(candidate: CandidateProfile) -> dict:
    X = build_features(candidate)
    model_map = {
        "retention": ("Tenure_1Year",            "Will candidate stay 1+ year?"),
        "safety":    ("OSHA_Recordable_Incident", "OSHA incident likelihood"),
        "promotion": ("PromotionWithin24Months",  "Promotion within 24 months"),
    }
    predictions = {}
    for model_name, (outcome_label, description) in model_map.items():
        model        = models[model_name]
        proba        = float(model.predict_proba(X)[0][1])
        tier         = get_risk_tier(proba, model_name)
        score        = get_risk_score(proba, model_name)
        shap_factors = get_shap_factors(model_name, X, n_factors=3)
        predictions[model_name] = {
            "outcome":      outcome_label,
            "description":  description,
            "probability":  round(proba, 4),
            "risk_tier":    tier,
            "score":        score,
            "shap_factors": shap_factors,
        }

    retention_score = predictions["retention"]["score"]
    safety_score    = predictions["safety"]["score"]
    promotion_score = predictions["promotion"]["score"]
    composite_score = round((retention_score * 0.5 + safety_score * 0.3 + promotion_score * 0.2), 1)
    overall_tier    = "Low Risk" if composite_score >= 70 else "Medium Risk" if composite_score >= 50 else "High Risk"
    cohort          = get_cohort_percentile(composite_score, candidate.SOC_Code)

    return {
        "candidate_summary": {
            "soc_code":      candidate.SOC_Code,
            "pi_score":      candidate.Sorvex360PI_Score,
            "overall_score": composite_score,
            "overall_tier":  overall_tier,
        },
        "predictions":   predictions,
        "cohort":        cohort,
        "model_version": "v3.0",
    }


# ── Gemini helper ─────────────────────────────────────────────────────────────
def call_gemini(prompt: str, max_tokens: int = 1024) -> str:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.4,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini call failed: {str(e)}")

def _shap_summary_text(shap_factors: dict) -> str:
    lines = []
    for model_name, factors in shap_factors.items():
        if not factors: continue
        label = {"retention":"Retention","safety":"Safety","promotion":"Promotion"}[model_name]
        factor_text = ", ".join([
            f"{f['label']} ({'+' if f['direction']=='increases' else '-'})"
            for f in factors
        ])
        lines.append(f"{label}: {factor_text}")
    return " | ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTES (protected)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if request.session.get("user"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return RedirectResponse(url="/login", status_code=302)

# ── Login ──────────────────────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse(url="/dashboard", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = USERS.get(username)
    if not user or user["password"] != password:
        return templates.TemplateResponse("login.html", {
            "request":  request,
            "error":    "Invalid username or password.",
            "username": username,
        })
    request.session["user"] = {
        "username": username,
        "role":     user["role"],
        "client":   user["client"],
    }
    return RedirectResponse(url="/dashboard", status_code=302)

# ── Logout ────────────────────────────────────────────────────────────────────
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)

# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "dashboard.html", "dashboard")

# ── Candidate Intelligence ────────────────────────────────────────────────────
@app.get("/candidates", response_class=HTMLResponse)
async def candidates(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "candidates.html", "candidates")

# ── Placement Readiness ───────────────────────────────────────────────────────
@app.get("/placement", response_class=HTMLResponse)
async def placement(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "placement.html", "placement")

# ── Batch Upload ──────────────────────────────────────────────────────────────
@app.get("/upload", response_class=HTMLResponse)
async def upload(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "upload.html", "upload")

# ── Hotspot Module ────────────────────────────────────────────────────────────
@app.get("/hotspot", response_class=HTMLResponse)
async def hotspot(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "hotspot.html", "hotspot")

# ── Single Screener ───────────────────────────────────────────────────────────
@app.get("/screener", response_class=HTMLResponse)
async def screener(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "screener.html", "screener")

# ── About ─────────────────────────────────────────────────────────────────────
@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    redir = require_auth(request)
    if redir: return redir
    return render_page(request, "about.html", "about")

# ── Admin (admin only) ────────────────────────────────────────────────────────
@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    redir = require_admin(request)
    if redir: return redir
    return render_page(request, "admin.html", "admin")


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS (unchanged from v2.1)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(models.keys()),
        "shap_ready":    [k for k, v in shap_explainers.items() if v is not None],
        "version":       "3.0.0"
    }

@app.post("/predict")
def predict(candidate: CandidateProfile):
    load_models()
    return run_predictions(candidate)

@app.post("/explain")
def explain(req: ExplainRequest):
    load_models()
    c            = req.candidate
    preds        = req.predictions
    cohort       = req.cohort
    shap_factors = req.shap_factors
    soc_label    = _soc_label(c.SOC_Code)
    shap_text    = _shap_summary_text(shap_factors)
    percentile   = cohort.get("percentile", 50)

    retention_tier = preds.get("retention", {}).get("risk_tier", "Unknown")
    safety_tier    = preds.get("safety",    {}).get("risk_tier", "Unknown")
    promotion_tier = preds.get("promotion", {}).get("risk_tier", "Unknown")
    overall_tier   = preds.get("overall_tier", "Medium Risk")

    summary_prompt = f"""You are a workforce analytics assistant for a utility staffing platform.
Write a concise 2-3 sentence plain English summary for a hiring manager about a {soc_label} candidate.
Be specific, professional, and actionable. No bullet points.

Candidate: Age {c.Age}, {c.EducationLevel}, PI Score {c.Sorvex360PI_Score}
Attendance: {c.AttendanceRate:.0%}, Safety Commitment: {c.SafetyCommitmentScore}/5
Training Hours: {c.TotalTrainingHours:,}, Certifications: {c.CertificationsEarned}
Prior Experience: {'Yes' if c.HasPriorTradeExperience else 'No'}, Veteran: {'Yes' if c.VeteranStatus else 'No'}

Results: {overall_tier} overall (top {100-percentile}% of cohort)
Retention: {retention_tier} | Safety: {safety_tier} | Promotion: {promotion_tier}
Key drivers: {shap_text}

Write 2-3 sentences. End with one specific recommendation."""

    plan_prompt = f"""You are a workforce development specialist at a utility staffing firm.
Create a 90-day onboarding plan for a new {soc_label} hire.
Use exactly three sections: Days 1-30, Days 31-60, Days 61-90.
Each section: 2-3 specific actionable items targeting their risk areas.

Risk profile:
Retention: {retention_tier} | Safety: {safety_tier} | Promotion: {promotion_tier}
Attendance: {c.AttendanceRate:.0%} | Safety Commitment: {c.SafetyCommitmentScore}/5
Training Hours: {c.TotalTrainingHours:,}

Generate the plan:"""

    summary = call_gemini(summary_prompt, max_tokens=400)
    plan    = call_gemini(plan_prompt,    max_tokens=2048)

    return {"summary": summary, "onboarding_plan": plan}

@app.post("/compare")
def compare(req: CompareRequest):
    load_models()
    result_a = run_predictions(req.candidate_a)
    result_b = run_predictions(req.candidate_b)
    a, b     = req.candidate_a, req.candidate_b

    def tier_summary(r):
        p = r["predictions"]
        return (f"Overall {r['candidate_summary']['overall_tier']} "
                f"(score {r['candidate_summary']['overall_score']}) | "
                f"Retention: {p['retention']['risk_tier']} | "
                f"Safety: {p['safety']['risk_tier']} | "
                f"Promotion: {p['promotion']['risk_tier']} | "
                f"Cohort: {r['cohort']['percentile']}th percentile")

    prompt = f"""You are a utility workforce hiring advisor. Compare two candidates and give a clear recommendation.
Format: 3 short paragraphs — Candidate A (2 sentences), Candidate B (2 sentences), Recommendation (2 sentences).

Candidate A — {_soc_label(a.SOC_Code)}:
PI Score: {a.Sorvex360PI_Score} | Attendance: {a.AttendanceRate:.0%} | Safety: {a.SafetyCommitmentScore}/5
Training: {a.TotalTrainingHours:,}hrs | Certs: {a.CertificationsEarned} | Prior Exp: {'Yes' if a.HasPriorTradeExperience else 'No'}
{tier_summary(result_a)}

Candidate B — {_soc_label(b.SOC_Code)}:
PI Score: {b.Sorvex360PI_Score} | Attendance: {b.AttendanceRate:.0%} | Safety: {b.SafetyCommitmentScore}/5
Training: {b.TotalTrainingHours:,}hrs | Certs: {b.CertificationsEarned} | Prior Exp: {'Yes' if b.HasPriorTradeExperience else 'No'}
{tier_summary(result_b)}

Provide your recommendation:"""

    commentary  = call_gemini(prompt, max_tokens=400)
    recommended = "A" if result_a["candidate_summary"]["overall_score"] >= result_b["candidate_summary"]["overall_score"] else "B"

    return {
        "candidate_a": result_a,
        "candidate_b": result_b,
        "comparison":  commentary,
        "recommended": recommended,
    }
