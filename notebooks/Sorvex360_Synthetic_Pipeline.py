# # 🏗️ Sorvex 360 — Synthetic Candidate Pipeline
# **Predict • Prepare • Place • Post-Hire Outcomes**
# 
# ## Architecture
# - **Stage 1:** Generate 1,000 seed records using blueprint parameters (rule-based statistical draws)
# - **Stage 2:** Train CTGAN on seed records and synthesize remaining 4,000
# - **Output:** 5,000 synthetic candidate profiles across all 4 phases
# 
# ## Join Keys
# - `CandidateID` → links Phase 1, 2, 3
# - `PlacementID` → links Phase 3 to Phase 4
# 
# ---
# **Runtime:** ~45-60 min on Colab T4 GPU (CTGAN training is the bottleneck)

# ── Cell 1 — Install & Import ────────────────────────────────────────────────
!pip install sdv pandas numpy scipy --quiet

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

warnings.filterwarnings('ignore')
np.random.seed(360)
random.seed(360)

OUTPUT_DIR = Path('/content/sorvex360_outputs/')
OUTPUT_DIR.mkdir(exist_ok=True)

SEED_N    = 1000   # Stage 1 seed records
TARGET_N  = 5000   # Final target
CTGAN_N   = TARGET_N - SEED_N  # Records CTGAN generates

print(f'✅ Ready. Seed: {SEED_N:,} | CTGAN: {CTGAN_N:,} | Total: {TARGET_N:,}')

# ── Cell 2 — SOC Configuration ───────────────────────────────────────────────
# All score parameters, CDL rates, and role-specific settings per SOC
# Source: ONET skills/knowledge EDA + CareerBuilder analysis + Blueprint

SOC_CONFIG = {
    '49-2022.00': {
        'label':           'Telecom Equipment Installer/Repairer',
        'share':           0.28,
        'cognitive_mean':  68, 'cognitive_sd': 9,
        'simulation_mean': 72, 'simulation_sd': 9,
        'behavioral_mean': 67, 'behavioral_sd': 9,
        'situational_mean':70, 'situational_sd': 8,
        'cdl_rate':        0.02,
        'union_rate':      0.08,
        'training_hrs_mean': 4500, 'training_hrs_sd': 1200,
        'physical_pass':   0.85,
        'loto_rate':       0.40,
        'cpr_rate':        0.06,
        'osha10_rate':     0.12,
        'pole_climb':      False,
    },
    '49-9051.00': {
        'label':           'Power Line Installer/Repairer',
        'share':           0.20,
        'cognitive_mean':  67, 'cognitive_sd': 9,
        'simulation_mean': 71, 'simulation_sd': 9,
        'behavioral_mean': 68, 'behavioral_sd': 9,
        'situational_mean':67, 'situational_sd': 9,
        'cdl_rate':        0.25,
        'union_rate':      0.22,
        'training_hrs_mean': 7000, 'training_hrs_sd': 1500,
        'physical_pass':   0.75,
        'loto_rate':       0.90,
        'cpr_rate':        0.08,
        'osha10_rate':     0.45,
        'pole_climb':      True,
    },
    '51-8013.00': {
        'label':           'Power Plant Operator',
        'share':           0.16,
        'cognitive_mean':  75, 'cognitive_sd': 8,
        'simulation_mean': 76, 'simulation_sd': 8,
        'behavioral_mean': 72, 'behavioral_sd': 8,
        'situational_mean':73, 'situational_sd': 8,
        'cdl_rate':        0.02,
        'union_rate':      0.18,
        'training_hrs_mean': 6000, 'training_hrs_sd': 1800,
        'physical_pass':   0.82,
        'loto_rate':       0.98,
        'cpr_rate':        0.18,
        'osha10_rate':     0.15,
        'pole_climb':      False,
    },
    '51-8031.00': {
        'label':           'Water/Wastewater Treatment Plant Operator',
        'share':           0.16,
        'cognitive_mean':  72, 'cognitive_sd': 8,
        'simulation_mean': 74, 'simulation_sd': 8,
        'behavioral_mean': 71, 'behavioral_sd': 8,
        'situational_mean':72, 'situational_sd': 8,
        'cdl_rate':        0.02,
        'union_rate':      0.12,
        'training_hrs_mean': 5500, 'training_hrs_sd': 1600,
        'physical_pass':   0.80,
        'loto_rate':       0.85,
        'cpr_rate':        0.35,
        'osha10_rate':     0.10,
        'pole_climb':      False,
    },
    '51-8092.00': {
        'label':           'Gas Plant Operator',
        'share':           0.12,
        'cognitive_mean':  73, 'cognitive_sd': 8,
        'simulation_mean': 75, 'simulation_sd': 8,
        'behavioral_mean': 73, 'behavioral_sd': 8,
        'situational_mean':72, 'situational_sd': 8,
        'cdl_rate':        0.40,
        'union_rate':      0.40,
        'training_hrs_mean': 6500, 'training_hrs_sd': 1700,
        'physical_pass':   0.82,
        'loto_rate':       0.95,
        'cpr_rate':        0.20,
        'osha10_rate':     0.15,
        'pole_climb':      False,
    },
    '43-5041.00': {
        'label':           'Meter Reader, Utilities',
        'share':           0.08,
        'cognitive_mean':  63, 'cognitive_sd': 8,
        'simulation_mean': 58, 'simulation_sd': 9,
        'behavioral_mean': 70, 'behavioral_sd': 8,
        'situational_mean':65, 'situational_sd': 8,
        'cdl_rate':        0.02,
        'union_rate':      0.10,
        'training_hrs_mean': 2500, 'training_hrs_sd': 800,
        'physical_pass':   0.92,
        'loto_rate':       0.20,
        'cpr_rate':        0.05,
        'osha10_rate':     0.08,
        'pole_climb':      False,
    },
}

SOC_CODES  = list(SOC_CONFIG.keys())
SOC_SHARES = [SOC_CONFIG[s]['share'] for s in SOC_CODES]
print('✅ SOC config loaded:', len(SOC_CONFIG), 'roles')
for soc, cfg in SOC_CONFIG.items():
    print(f"  {soc} — {cfg['label']} ({cfg['share']*100:.0f}%)")

# ── Cell 3 — Helper Functions ────────────────────────────────────────────────

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def bernoulli(rate):
    return np.random.random() < rate

def normal_score(mean, sd, lo=50, hi=100):
    return int(clamp(round(np.random.normal(mean, sd)), lo, hi))

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    return start + timedelta(days=np.random.randint(0, (end - start).days))

def age_tenure_mean(age):
    """BLS median tenure by age band (years)"""
    if age < 25: return 1.0
    if age < 35: return 2.7
    if age < 45: return 4.6
    if age < 55: return 7.0
    return 9.6

def tenure_1yr_rate(age):
    """Probability of passing 1-year mark by age band (BLS)"""
    if age < 25: return 0.65
    if age < 35: return 0.72
    if age < 45: return 0.82
    if age < 55: return 0.87
    return 0.90

def compute_pi_score(cog, sim, beh, sit, has_prior, edu):
    """Sorvex360PI_Score weighted composite (0-100)"""
    edu_bonus = {'HS/GED': 0, 'Vocational Certificate': 3,
                 'Associate Technical': 8, 'Associate General': 4,
                 "Bachelor's": 5, 'Apprenticeship': 6}
    score = (
        cog * 0.30 +
        beh * 0.20 +
        sit * 0.20 +
        sim * 0.15 +
        (10 if has_prior else 0) * 0.10 / 0.10 * 0.10 +
        edu_bonus.get(edu, 0)
    )
    return round(clamp(score, 0, 100), 1)

print('✅ Helper functions ready.')

# ── Cell 4 — Phase 1: Generate PREDICT Records ───────────────────────────────
print(f'Generating {SEED_N:,} Phase 1 (PREDICT) seed records...\n')

EDU_LEVELS   = ['HS/GED','Vocational Certificate','Associate Technical','Associate General',"Bachelor's",'Apprenticeship']
EDU_WEIGHTS  = [0.32, 0.18, 0.17, 0.06, 0.08, 0.19]

SOURCE_LABELS   = ['WorkforceProgram','SelfReferred','TradeUnion','MilitaryTransition','CommunityCollege','Other']
SOURCE_WEIGHTS  = [0.25, 0.22, 0.20, 0.12, 0.13, 0.08]

INDUSTRY_LABELS  = ['Utility','Construction','Manufacturing','Military','Other']
INDUSTRY_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.10]

TRAINING_LABELS  = ['IBEW JATC','Trade Union','Community College','Employer Program','Military','Self-taught']
TRAINING_WEIGHTS = [0.28, 0.20, 0.18, 0.15, 0.12, 0.07]

STATE_LABELS  = ['CA','TX','VA','NC','WA','FL','IL','PA','MI','MD','TN','Other']
STATE_WEIGHTS = [0.14, 0.13, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.12]

phase1_records = []

for i in range(SEED_N):
    candidate_id = f'SORV-{i+1:05d}'
    soc = np.random.choice(SOC_CODES, p=SOC_SHARES)
    cfg = SOC_CONFIG[soc]

    # Demographics
    age    = int(clamp(round(np.random.normal(43.1, 10.5)), 18, 60))
    gender = np.random.choice(['Male','Female','Other'], p=[0.731, 0.259, 0.010])
    state  = np.random.choice(STATE_LABELS, p=STATE_WEIGHTS)
    lang   = np.random.choice(['English','Spanish','Other'], p=[0.813, 0.161, 0.026])
    vet    = bernoulli(0.12)

    # Education & Credentials
    edu    = np.random.choice(EDU_LEVELS, p=EDU_WEIGHTS)
    cdl    = bernoulli(cfg['cdl_rate'])
    osha10 = bernoulli(cfg['osha10_rate'])
    cpr    = bernoulli(cfg['cpr_rate'])

    # Assessment Scores — base from SOC config
    cog = normal_score(cfg['cognitive_mean'],   cfg['cognitive_sd'])
    sim = normal_score(cfg['simulation_mean'],  cfg['simulation_sd'])
    beh = normal_score(cfg['behavioral_mean'],  cfg['behavioral_sd'])
    sit = normal_score(cfg['situational_mean'], cfg['situational_sd'])

    # Score modifiers
    if age >= 50:
        cog = int(clamp(cog * 0.95, 50, 100))  # APA aging: -5%
    if edu == 'Associate Technical':
        cog = int(clamp(cog * 1.05, 50, 100))  # CAST-R: +25% mechanical, approx +5% composite

    # Prior work
    prior_exp_rate = {(18,25): 0.30, (26,35): 0.65, (36,45): 0.82, (46,61): 0.91}
    for (lo, hi), rate in prior_exp_rate.items():
        if lo <= age < hi:
            has_prior = bernoulli(rate)
            break

    industry   = np.random.choice(INDUSTRY_LABELS, p=INDUSTRY_WEIGHTS)
    tenure_yrs = max(0.1, np.random.lognormal(
        np.log(max(0.1, age_tenure_mean(age))), 0.6
    ))
    longest_tenure = round(clamp(tenure_yrs, 0.1, 35), 1)
    train_src  = np.random.choice(TRAINING_LABELS, p=TRAINING_WEIGHTS)
    appr_int   = bernoulli(0.55)

    # Screening eligibility
    has_license = bernoulli(0.78)
    drug_screen = bernoulli(0.88)
    background  = bernoulli(0.85)

    # Motivation
    source = np.random.choice(SOURCE_LABELS, p=SOURCE_WEIGHTS)

    # PI Score
    pi_score = compute_pi_score(cog, sim, beh, sit, has_prior, edu)

    phase1_records.append({
        'CandidateID':             candidate_id,
        'SOC_Code':                soc,
        'JobFamily':               cfg['label'],
        'Age':                     age,
        'Gender':                  gender,
        'State':                   state,
        'PrimaryLanguage':         lang,
        'VeteranStatus':           int(vet),
        'EducationLevel':          edu,
        'CDL_Status':              int(cdl),
        'OSHA10_Status':           int(osha10),
        'CPR_Status':              int(cpr),
        'CognitiveScore':          cog,
        'SimulationScore':         sim,
        'BehavioralScore':         beh,
        'SituationalScore':        sit,
        'HasPriorTradeExperience': int(has_prior),
        'MostRecentIndustry':      industry,
        'LongestJobTenure':        longest_tenure,
        'TrainingSource':          train_src,
        'ApprenticeshipInterest':  int(appr_int),
        'HasValidLicense':         int(has_license),
        'CanPassDrugScreen':       int(drug_screen),
        'CanPassBackgroundCheck':  int(background),
        'SourceOfCandidate':       source,
        'Sorvex360PI_Score':       pi_score,
    })

phase1_df = pd.DataFrame(phase1_records)
print(f'✅ Phase 1 complete: {len(phase1_df):,} records')
print(f'\nSOC distribution:')
print(phase1_df['SOC_Code'].value_counts())
print(f'\nPI Score stats:')
print(phase1_df['Sorvex360PI_Score'].describe().round(2))

# ── Cell 5 — Phase 2: Generate PREPARE Records ───────────────────────────────
print(f'Generating Phase 2 (PREPARE) records conditioned on Phase 1...\n')

EMPLOYERS = {
    '49-2022.00': ['Charter Communications','US Cellular','Intrado','Avacend Corporation','Leidos'],
    '49-9051.00': ['Pike Corporation','NPL Construction Co.','Q3 Contracting Inc.','Davey Tree Expert Co.','Altec Industries'],
    '51-8013.00': ['Exelon Corporation','BGIS North America','Exide Technologies','Veolia North America','General Dynamics IT'],
    '51-8031.00': ['Veolia North America','BGIS North America','NPL Construction Co.','Exelon Corporation','US Cellular'],
    '51-8092.00': ['Exelon Corporation','General Dynamics IT','Veolia North America','Leidos','GPAC'],
    '43-5041.00': ['Exelon Corporation','US Cellular','Charter Communications','Veolia North America','GPAC'],
}

phase2_records = []

for _, p1 in phase1_df.iterrows():
    soc = p1['SOC_Code']
    cfg = SOC_CONFIG[soc]
    pi  = p1['Sorvex360PI_Score']
    age = p1['Age']
    edu = p1['EducationLevel']

    # Training hours — WA L&I distribution by SOC
    train_hrs = int(clamp(round(np.random.normal(
        cfg['training_hrs_mean'], cfg['training_hrs_sd'])), 500, 10000))

    # Attendance — Beta distribution conditioned on BehavioralScore
    beh_norm = p1['BehavioralScore'] / 100
    att_mean = 0.65 + beh_norm * 0.25  # range 0.65-0.90
    att_alpha = att_mean * 8
    att_beta  = (1 - att_mean) * 8
    attendance = round(clamp(np.random.beta(att_alpha, att_beta), 0.50, 0.99), 3)

    # Start/End dates
    start_date = random_date(2020, 2024)
    weeks      = train_hrs / 40
    end_date   = start_date + timedelta(weeks=weeks)

    # Passed modules — conditioned on attendance + cognitive
    pass_prob = 0.30
    if attendance > 0.85 and p1['CognitiveScore'] > 60:
        pass_prob = 0.78
    elif attendance > 0.75:
        pass_prob = 0.55
    elif p1['CognitiveScore'] > 65:
        pass_prob = 0.50
    passed_modules = bernoulli(pass_prob)

    # Certifications earned — conditioned on passed_modules
    if passed_modules:
        certs = int(np.random.poisson(4.1))
    else:
        certs = int(np.random.poisson(1.1))
    certs = clamp(certs, 0, 6)

    # Simulation performance
    sim_perf = clamp(round(np.random.normal(
        p1['SimulationScore'] * 0.9, 8)), 0, 100)

    # Instructor scores — factory worker trait distributions
    teamwork    = round(clamp(np.random.normal(3.2, 0.7), 1, 5), 1)
    # SafetyCommitmentScore — EEI SCL: +30% for experienced workers
    safety_base = round(clamp(np.random.normal(
        2.8 + p1['BehavioralScore'] / 100 * 1.5, 0.65), 1, 5), 1)
    if p1['LongestJobTenure'] > 5:
        safety_base = round(clamp(safety_base * 1.10, 1, 5), 1)  # EEI SCL +30% → ~10% on 5pt scale
    reliability = round(clamp(np.random.normal(
        3.3 + (attendance - 0.75) * 2, 0.65), 1, 5), 1)

    # Physical tests — by SOC + age adjustment
    phys_rate = cfg['physical_pass']
    if age >= 50: phys_rate *= 0.92
    phys_pass  = bernoulli(phys_rate)
    pole_climb = bernoulli(0.72) if cfg['pole_climb'] else None
    lift50     = bernoulli(phys_rate * 1.05)
    colorblind = bernoulli(0.08 if p1['Gender'] == 'Male' else 0.005)

    # Completion status — WA L&I: Completed=46.3% raw, adjusted upward
    completed_prob = 0.30
    if passed_modules and attendance > 0.80:
        completed_prob = 0.85
    elif passed_modules:
        completed_prob = 0.65
    elif attendance > 0.75:
        completed_prob = 0.40
    completed = bernoulli(completed_prob)

    dropout_reason = None
    if not completed:
        dropout_reason = np.random.choice(
            ['Personal','Work Conflict','Performance','Medical','Unknown'],
            p=[0.35, 0.25, 0.20, 0.10, 0.10]
        )

    # PI Score at completion
    if completed:
        delta = np.random.normal(8, 5)   # positive training impact
    else:
        delta = np.random.normal(-8, 4)  # negative
    pi_at_completion = round(clamp(pi + delta, 0, 100), 1)

    phase2_records.append({
        'CandidateID':                    p1['CandidateID'],
        'TrainingProvider':               np.random.choice(EMPLOYERS[soc]),
        'StartDate':                      start_date.strftime('%Y-%m-%d'),
        'EndDate':                        end_date.strftime('%Y-%m-%d'),
        'TotalTrainingHours':             train_hrs,
        'AttendanceRate':                 attendance,
        'PassedRequiredModules':          int(passed_modules),
        'CertificationsEarned':           certs,
        'SimulationPerformance':          sim_perf,
        'TeamworkScore':                  teamwork,
        'SafetyCommitmentScore':          safety_base,
        'ReliabilityScore':               reliability,
        'PhysicalTestResult':             int(phys_pass),
        'PoleClimbResult':                int(pole_climb) if pole_climb is not None else None,
        'Lift50lbsTest':                  int(lift50),
        'ColorBlindness':                 int(colorblind),
        'Completed':                      int(completed),
        'DropoutReason':                  dropout_reason,
        'Sorvex360PI_Score_AtCompletion': pi_at_completion,
        'ReadinessDelta':                 round(pi_at_completion - pi, 1),
    })

phase2_df = pd.DataFrame(phase2_records)
print(f'✅ Phase 2 complete: {len(phase2_df):,} records')
print(f'Completion rate: {phase2_df["Completed"].mean():.1%}')
print(f'Avg ReadinessDelta: {phase2_df["ReadinessDelta"].mean():.2f}')

# ── Cell 6 — Phase 3: Generate PLACE Records ─────────────────────────────────
print('Generating Phase 3 (PLACE) records for completed candidates...\n')

# Merge Phase 1 + 2
merged12 = phase1_df.merge(phase2_df, on='CandidateID')
completed_df = merged12[merged12['Completed'] == 1].copy()
print(f'Candidates proceeding to placement: {len(completed_df):,} / {len(merged12):,}')

EMPLOYER_NAMES = {
    '49-2022.00': ['Charter Communications','US Cellular','Intrado','Avacend Corporation','Leidos','General Dynamics IT'],
    '49-9051.00': ['Pike Corporation','NPL Construction Co.','Q3 Contracting Inc.','Davey Tree Expert Co.','Altec Industries','PDS Tech Inc.'],
    '51-8013.00': ['Exelon Corporation','BGIS North America','Exide Technologies','Veolia North America','General Dynamics IT'],
    '51-8031.00': ['Veolia North America','BGIS North America','NPL Construction Co.','Exelon Corporation'],
    '51-8092.00': ['Exelon Corporation','General Dynamics IT','Veolia North America','Leidos','GPAC'],
    '43-5041.00': ['Exelon Corporation','US Cellular','Charter Communications','Veolia North America'],
}

phase3_records = []
placement_counter = 1

for _, row in completed_df.iterrows():
    soc = row['SOC_Code']
    cfg = SOC_CONFIG[soc]

    placement_id = f'PLC-{placement_counter:05d}'
    placement_counter += 1

    # Employment type by SOC
    if soc == '49-2022.00':
        emp_type = np.random.choice(['Full-Time','Part-Time','Contractor'], p=[0.86, 0.04, 0.10])
    elif soc == '49-9051.00':
        emp_type = np.random.choice(['Full-Time','Part-Time'], p=[0.93, 0.07])
    else:
        emp_type = np.random.choice(['Full-Time','Part-Time'], p=[0.92, 0.08])

    union = bernoulli(cfg['union_rate'])

    # Hire date = end_date + placement lag
    end_dt    = datetime.strptime(row['EndDate'], '%Y-%m-%d')
    lag_days  = np.random.randint(7, 91)
    hire_date = end_dt + timedelta(days=lag_days)

    # Compliance requirements
    req_cdl    = bernoulli(cfg['cdl_rate'] * 1.1)
    req_dot    = int(req_cdl)
    req_osha10 = bernoulli(cfg['osha10_rate'] * 1.1)
    req_cpr    = bernoulli(cfg['cpr_rate'] * 1.1)

    # Pre-hire verification
    mvr_ver  = bernoulli(0.92 if row['HasValidLicense'] else 0.05)
    drug_ver = bernoulli(0.96 if row['CanPassDrugScreen'] else 0.08)
    bg_ver   = bernoulli(0.94 if row['CanPassBackgroundCheck'] else 0.05)
    cert_ver = bernoulli(0.88 if row['CertificationsEarned'] > 0 else 0.10)

    # Orientation
    safety_orient_date = hire_date + timedelta(days=np.random.randint(0, 4))
    loto_comp = bernoulli(cfg['loto_rate'])
    ppe_fit   = bernoulli(0.95 if soc != '43-5041.00' else 0.70)

    # Apprenticeship registration
    appr_reg = bernoulli(0.85 if (row['ApprenticeshipInterest'] and union) else
                         0.40 if row['ApprenticeshipInterest'] else 0.05)

    phase3_records.append({
        'CandidateID':              row['CandidateID'],
        'PlacementID':              placement_id,
        'EmployerName':             np.random.choice(EMPLOYER_NAMES[soc]),
        'JobFamily':                row['JobFamily'],
        'EmploymentType':           emp_type,
        'UnionStatus':              int(union),
        'HireDate':                 hire_date.strftime('%Y-%m-%d'),
        'RoleRequires_CDL':         int(req_cdl),
        'RoleRequires_DOTMedical':  req_dot,
        'RoleRequires_OSHA10':      int(req_osha10),
        'RoleRequires_CPR':         int(req_cpr),
        'PreHire_Verified_MVR':     int(mvr_ver),
        'PreHire_Verified_DrugScreen': int(drug_ver),
        'PreHire_Verified_Background': int(bg_ver),
        'PreHire_Verified_Certificates': int(cert_ver),
        'Orientation_SiteSafety_Date': safety_orient_date.strftime('%Y-%m-%d'),
        'Orientation_LOTO_Completed': int(loto_comp),
        'Orientation_PPE_Fitted':   int(ppe_fit),
        'Apprenticeship_Registered': int(appr_reg),
        'Sorvex360PI_ScoreAtHire':  row['Sorvex360PI_Score_AtCompletion'],
    })

phase3_df = pd.DataFrame(phase3_records)
print(f'✅ Phase 3 complete: {len(phase3_df):,} placement records')
print(f'Union rate: {phase3_df["UnionStatus"].mean():.1%}')
print(f'Employment type:\n{phase3_df["EmploymentType"].value_counts()}')

# ── Cell 7 — Phase 4: Generate POST-HIRE OUTCOMES ────────────────────────────
print('Generating Phase 4 (POST-HIRE OUTCOMES) records...\n')

# Merge all phases
merged123 = completed_df.merge(phase3_df, on='CandidateID')

phase4_records = []

for _, row in merged123.iterrows():
    soc   = row['SOC_Code']
    pi    = row['Sorvex360PI_ScoreAtHire']
    age   = row['Age']
    union = row['UnionStatus']

    # ── Tenure ───────────────────────────────────────────────────────────────
    # BLS: Utilities median 4.9yr, Install/Maint 3.9yr
    base_tenure_yr = 4.9 if soc in ['51-8013.00','51-8031.00','51-8092.00','43-5041.00'] else 3.9
    if union: base_tenure_yr *= 1.35  # Union +35%

    # PI Score influence on tenure
    pi_mult = 0.7 + (pi / 100) * 0.6  # range 0.7-1.3
    tenure_yr = max(0.1, np.random.weibull(1.5) * base_tenure_yr * pi_mult)
    tenure_days = int(clamp(round(tenure_yr * 365), 1, 7300))

    hire_dt = datetime.strptime(row['HireDate'], '%Y-%m-%d')
    sep_dt  = hire_dt + timedelta(days=tenure_days)

    # Milestone flags
    t90d   = int(tenure_days >= 90)
    t6mo   = int(tenure_days >= 180)
    t1yr   = int(bernoulli(tenure_1yr_rate(age) * pi_mult))

    # Employment status
    if tenure_days > 1000:
        status = np.random.choice(['Active','Separated','On Leave'], p=[0.65, 0.28, 0.07])
    else:
        status = np.random.choice(['Active','Separated'], p=[0.40, 0.60])

    # ── Safety ───────────────────────────────────────────────────────────────
    # BLS base: 1.9 per 100 FTE = 1.9% annual
    safety_score = row['SafetyCommitmentScore']
    osha_rate = 0.019
    if safety_score < 2.5 and tenure_days < 365:
        osha_rate = 0.032  # higher risk — new + low safety commitment
    elif safety_score > 4.0 and tenure_days > 730:
        osha_rate = 0.008  # lower risk — experienced + high safety commitment
    osha_incident = int(bernoulli(osha_rate * (tenure_days / 365)))

    # Total safety incidents
    if osha_incident:
        total_safety = int(np.random.poisson(2.1))
    else:
        total_safety = int(np.random.poisson(0.3))

    total_violations = int(np.random.poisson(0.8 * (1 - safety_score / 5)))
    total_preventable = int(min(total_safety, np.random.binomial(total_safety, 0.30)))

    # ── Attendance ───────────────────────────────────────────────────────────
    # BLS anchor: 4-6 unscheduled days/year for utility workers
    att = row['AttendanceRate']
    if att > 0.90:
        absence_lambda = 3.0
    elif att > 0.80:
        absence_lambda = 5.0
    else:
        absence_lambda = 9.0
    unscheduled_abs = int(np.random.poisson(absence_lambda))
    probation = int(bernoulli(
        0.65 if pi < 55 else 0.25 if pi < 70 else 0.08
    ))

    # ── Drug screens ─────────────────────────────────────────────────────────
    screens_lambda = 2.5 if row['CDL_Status'] else 1.2
    screens_taken  = int(np.random.poisson(screens_lambda))
    screens_passed = sum(bernoulli(0.975) for _ in range(screens_taken))

    # ── Recertification ──────────────────────────────────────────────────────
    recert_safety = int(bernoulli(0.88 if safety_score > 3.5 else 0.62))
    recert_cpr    = int(bernoulli(0.91))

    # ── Manager feedback ─────────────────────────────────────────────────────
    # Factory EDA: Mean=2.976, SD=0.560, Normal
    if pi > 80:
        mgr_mean = 3.8
    elif pi > 60:
        mgr_mean = 3.0
    else:
        mgr_mean = 2.3
    mgr_score  = round(clamp(np.random.normal(mgr_mean, 0.56), 1, 5), 2)

    # Promotion — Factory EDA: 20.8% rate
    promo_prob = 0.05
    if mgr_score > 3.5 and tenure_days > 365:
        promo_prob = 0.35
    elif mgr_score > 2.5 and tenure_days > 365:
        promo_prob = 0.15
    promotion = int(bernoulli(promo_prob))

    # Time to competency — WA L&I training hours / 160 hrs per month
    comp_months = round(clamp(
        np.random.lognormal(np.log(row['TotalTrainingHours'] / 160), 0.4), 1, 60
    ), 1)

    phase4_records.append({
        'CandidateID':                  row['CandidateID'],
        'PlacementID':                  row['PlacementID'],
        'TenureInDays':                 tenure_days,
        'Tenure_90Day':                 t90d,
        'Tenure_6Month':                t6mo,
        'Tenure_1Year':                 t1yr,
        'EmploymentStatus':             status,
        'OSHA_Recordable_Incident':     osha_incident,
        'TotalSafetyIncidents':         total_safety,
        'TotalComplianceViolations':    total_violations,
        'TotalPreventableAccidents':    total_preventable,
        'UnscheduledAbsences':          unscheduled_abs,
        'ProbationStatus':              probation,
        'RandomDrugScreens_Taken':      screens_taken,
        'RandomDrugScreens_Passed':     screens_passed,
        'Recert_Safety_PassedFirstTry': recert_safety,
        'Recert_CPR_PassedFirstTry':    recert_cpr,
        'ManagerFitFeedback_Score':     mgr_score,
        'PromotionWithin24Months':      promotion,
        'Time_To_Competency_Months':    comp_months,
    })

phase4_df = pd.DataFrame(phase4_records)
print(f'✅ Phase 4 complete: {len(phase4_df):,} outcome records')
print(f'Tenure_1Year rate: {phase4_df["Tenure_1Year"].mean():.1%}')
print(f'OSHA incident rate: {phase4_df["OSHA_Recordable_Incident"].mean():.2%}')
print(f'Promotion rate: {phase4_df["PromotionWithin24Months"].mean():.1%}')
print(f'Probation rate: {phase4_df["ProbationStatus"].mean():.1%}')
print(f'Avg ManagerFitFeedback: {phase4_df["ManagerFitFeedback_Score"].mean():.3f}')

# ── Cell 8 — Validate Seed Records ───────────────────────────────────────────
print('=== SEED RECORD VALIDATION ===')

# Blueprint targets
checks = [
    ('OSHA incident rate', phase4_df['OSHA_Recordable_Incident'].mean(), 0.01, 0.03),
    ('Tenure_1Year rate', phase4_df['Tenure_1Year'].mean(), 0.70, 0.85),
    ('Promotion rate', phase4_df['PromotionWithin24Months'].mean(), 0.18, 0.25),
    ('Probation rate', phase4_df['ProbationStatus'].mean(), 0.20, 0.30),
    ('Manager score mean', phase4_df['ManagerFitFeedback_Score'].mean(), 2.7, 3.3),
    ('Completion rate', phase2_df['Completed'].mean(), 0.45, 0.65),
    ('Union rate', phase3_df['UnionStatus'].mean(), 0.15, 0.28),
    ('PI Score mean', phase1_df['Sorvex360PI_Score'].mean(), 60, 80),
]

all_pass = True
for name, val, lo, hi in checks:
    status = '✅' if lo <= val <= hi else '❌'
    if status == '❌': all_pass = False
    print(f'{status} {name}: {val:.3f} (target {lo:.2f}-{hi:.2f})')

# Correlation checks
print('\n=== CORRELATION CHECKS ===')
full_seed = phase1_df.merge(phase2_df, on='CandidateID').merge(
    phase3_df, on='CandidateID', how='left').merge(
    phase4_df, on='CandidateID', how='left')

corr_checks = [
    ('CognitiveScore → PI_Score', 'CognitiveScore', 'Sorvex360PI_Score', 0.55),
    ('Age → LongestJobTenure', 'Age', 'LongestJobTenure', 0.35),
]
for name, c1, c2, threshold in corr_checks:
    r = full_seed[[c1, c2]].corr().iloc[0, 1]
    status = '✅' if abs(r) >= threshold else '⚠️ '
    print(f'{status} {name}: r={r:.3f} (min |{threshold}|)')

print(f'\n{"✅ All validation checks passed!" if all_pass else "⚠️  Some checks failed — review before CTGAN training"}')
print(f'\nSeed records ready for CTGAN: {len(full_seed):,}')

# ── Cell 9 — Train CTGAN & Synthesize Remaining Records ──────────────────────
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

print('Preparing CTGAN training data...')

# Build flat training table — drop string/date columns CTGAN can't handle well
drop_cols = ['CandidateID','PlacementID','SOC_Code','JobFamily','Gender',
             'State','PrimaryLanguage','EducationLevel','MostRecentIndustry',
             'TrainingSource','SourceOfCandidate','TrainingProvider',
             'EmployerName','EmploymentType','StartDate','EndDate',
             'HireDate','Orientation_SiteSafety_Date','EmploymentStatus',
             'DropoutReason']

train_df = full_seed.drop(columns=[c for c in drop_cols if c in full_seed.columns])
train_df = train_df.dropna()

print(f'Training table: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns')
print(f'Columns: {list(train_df.columns)}')

# Auto-detect metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_df)

# Train CTGAN
print(f'\nTraining CTGAN on {len(train_df):,} seed records...')
print('(This takes ~20-40 minutes on Colab CPU, ~10-15 min on T4 GPU)')

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=300,
    batch_size=500,
    verbose=True
)
synthesizer.fit(train_df)
print('\n✅ CTGAN training complete')

# Generate synthetic records
print(f'Generating {CTGAN_N:,} synthetic records...')
synthetic_numeric = synthesizer.sample(num_rows=CTGAN_N)
print(f'✅ Generated: {len(synthetic_numeric):,} records')

# ── Cell 10 — Rebuild Full Synthetic Records ──────────────────────────────────
print('Rebuilding categorical and ID fields for CTGAN records...')

n = len(synthetic_numeric)

# Assign IDs
synthetic_numeric['CandidateID'] = [f'SORV-S{i+1:05d}' for i in range(n)]

# Re-assign SOC based on synthetic scores (closest match to SOC mean profile)
def assign_soc(row):
    cog = row.get('CognitiveScore', 68)
    distances = {}
    for soc, cfg in SOC_CONFIG.items():
        distances[soc] = abs(cog - cfg['cognitive_mean'])
    return min(distances, key=distances.get)

synthetic_numeric['SOC_Code'] = synthetic_numeric.apply(assign_soc, axis=1)
synthetic_numeric['JobFamily'] = synthetic_numeric['SOC_Code'].map(
    {k: v['label'] for k, v in SOC_CONFIG.items()}
)

# Re-assign categorical fields using original distributions
synthetic_numeric['Gender']         = np.random.choice(['Male','Female','Other'], n, p=[0.731, 0.259, 0.010])
synthetic_numeric['State']          = np.random.choice(STATE_LABELS, n, p=STATE_WEIGHTS)
synthetic_numeric['PrimaryLanguage']= np.random.choice(['English','Spanish','Other'], n, p=[0.813, 0.161, 0.026])
synthetic_numeric['EducationLevel'] = np.random.choice(EDU_LEVELS, n, p=EDU_WEIGHTS)
synthetic_numeric['MostRecentIndustry'] = np.random.choice(INDUSTRY_LABELS, n, p=INDUSTRY_WEIGHTS)
synthetic_numeric['TrainingSource'] = np.random.choice(TRAINING_LABELS, n, p=TRAINING_WEIGHTS)
synthetic_numeric['SourceOfCandidate'] = np.random.choice(SOURCE_LABELS, n, p=SOURCE_WEIGHTS)
synthetic_numeric['EmploymentStatus'] = np.random.choice(
    ['Active','Separated','On Leave','Transferred'], n, p=[0.55, 0.35, 0.05, 0.05])

# Clamp numeric fields to valid ranges
score_fields = ['CognitiveScore','SimulationScore','BehavioralScore','SituationalScore']
for f in score_fields:
    if f in synthetic_numeric.columns:
        synthetic_numeric[f] = synthetic_numeric[f].clip(50, 100).round().astype(int)

if 'AttendanceRate' in synthetic_numeric.columns:
    synthetic_numeric['AttendanceRate'] = synthetic_numeric['AttendanceRate'].clip(0.50, 0.99).round(3)
if 'TenureInDays' in synthetic_numeric.columns:
    synthetic_numeric['TenureInDays'] = synthetic_numeric['TenureInDays'].clip(1, 7300).round().astype(int)
if 'ManagerFitFeedback_Score' in synthetic_numeric.columns:
    synthetic_numeric['ManagerFitFeedback_Score'] = synthetic_numeric['ManagerFitFeedback_Score'].clip(1, 5).round(2)

print(f'✅ Rebuilt {len(synthetic_numeric):,} CTGAN records with categorical fields')

# ── Cell 11 — Combine Seed + Synthetic & Split into Phase CSVs ───────────────
print('Combining seed + synthetic records and splitting into phase files...')

# Combine full_seed (seed records) + synthetic_numeric (CTGAN records)
all_records = pd.concat([full_seed, synthetic_numeric], ignore_index=True)
mask = all_records['CandidateID'].isna()
all_records.loc[mask, 'CandidateID'] = [f'SORV-S{i:05d}' for i in range(mask.sum())]

print(f'Total records: {len(all_records):,}')

# Phase 1 columns
p1_cols = ['CandidateID','SOC_Code','JobFamily','Age','Gender','State',
           'PrimaryLanguage','VeteranStatus','EducationLevel','CDL_Status',
           'OSHA10_Status','CPR_Status','CognitiveScore','SimulationScore',
           'BehavioralScore','SituationalScore','HasPriorTradeExperience',
           'MostRecentIndustry','LongestJobTenure','TrainingSource',
           'ApprenticeshipInterest','HasValidLicense','CanPassDrugScreen',
           'CanPassBackgroundCheck','SourceOfCandidate','Sorvex360PI_Score']

# Phase 2 columns
p2_cols = ['CandidateID','TrainingProvider','StartDate','EndDate',
           'TotalTrainingHours','AttendanceRate','PassedRequiredModules',
           'CertificationsEarned','SimulationPerformance','TeamworkScore',
           'SafetyCommitmentScore','ReliabilityScore','PhysicalTestResult',
           'PoleClimbResult','Lift50lbsTest','ColorBlindness','Completed',
           'DropoutReason','Sorvex360PI_Score_AtCompletion','ReadinessDelta']

# Phase 3 columns
p3_cols = ['CandidateID','PlacementID','EmployerName','JobFamily',
           'EmploymentType','UnionStatus','HireDate','RoleRequires_CDL',
           'RoleRequires_DOTMedical','RoleRequires_OSHA10','RoleRequires_CPR',
           'PreHire_Verified_MVR','PreHire_Verified_DrugScreen',
           'PreHire_Verified_Background','PreHire_Verified_Certificates',
           'Orientation_SiteSafety_Date','Orientation_LOTO_Completed',
           'Orientation_PPE_Fitted','Apprenticeship_Registered',
           'Sorvex360PI_ScoreAtHire']

# Phase 4 columns
p4_cols = ['CandidateID','PlacementID','TenureInDays','Tenure_90Day',
           'Tenure_6Month','Tenure_1Year','EmploymentStatus',
           'OSHA_Recordable_Incident','TotalSafetyIncidents',
           'TotalComplianceViolations','TotalPreventableAccidents',
           'UnscheduledAbsences','ProbationStatus','RandomDrugScreens_Taken',
           'RandomDrugScreens_Passed','Recert_Safety_PassedFirstTry',
           'Recert_CPR_PassedFirstTry','ManagerFitFeedback_Score',
           'PromotionWithin24Months','Time_To_Competency_Months']

# Save phase files
for phase_name, cols, source_df in [
    ('Phase1_Predict',    p1_cols, all_records),
    ('Phase2_Prepare',    p2_cols, all_records),
    ('Phase3_Place',      p3_cols, all_records),
    ('Phase4_PostHire',   p4_cols, all_records),
]:
    avail = [c for c in cols if c in source_df.columns]
    df_out = source_df[avail].dropna(subset=['CandidateID'])
    path = OUTPUT_DIR / f'Sorvex360_{phase_name}.csv'
    df_out.to_csv(path, index=False)
    print(f'  ✅ {phase_name}: {len(df_out):,} rows x {len(avail)} cols → {path.name}')

# Save master flat file
all_records.to_csv(OUTPUT_DIR / 'Sorvex360_Master_Flat.csv', index=False)
print(f'  ✅ Master flat file: {len(all_records):,} rows x {len(all_records.columns)} cols')

# ── Cell 12 — Final Validation Report ────────────────────────────────────────
import matplotlib.pyplot as plt

print('=== SORVEX 360 FINAL VALIDATION REPORT ===')
print(f'Total records: {len(all_records):,}')
print(f'Unique candidates: {all_records["CandidateID"].nunique():,}')

p1_final = pd.read_csv(OUTPUT_DIR / 'Sorvex360_Phase1_Predict.csv')
p4_final = pd.read_csv(OUTPUT_DIR / 'Sorvex360_Phase4_PostHire.csv')

print('\n--- Phase 1 Stats ---')
print(p1_final['Sorvex360PI_Score'].describe().round(2))
print(f'\nSOC distribution:')
print(p1_final['SOC_Code'].value_counts(normalize=True).round(3))

print('\n--- Phase 4 Outcome Rates ---')
outcomes = {
    'Tenure_1Year':             ('70-85%', 0.70, 0.85),
    'OSHA_Recordable_Incident': ('1-3%',   0.01, 0.03),
    'PromotionWithin24Months':  ('18-25%', 0.18, 0.25),
    'ProbationStatus':          ('20-30%', 0.20, 0.30),
}
for field, (target, lo, hi) in outcomes.items():
    if field in p4_final.columns:
        val = p4_final[field].mean()
        status = '✅' if lo <= val <= hi else '⚠️ '
        print(f'{status} {field}: {val:.2%} (target {target})')

# Plot PI Score distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

p1_final['Sorvex360PI_Score'].hist(bins=40, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('PI Score Distribution')
axes[0].set_xlabel('Score (0-100)')

p1_final['SOC_Code'].value_counts().plot(kind='barh', ax=axes[1], color='coral')
axes[1].set_title('Records by SOC Code')

if 'TenureInDays' in p4_final.columns:
    (p4_final['TenureInDays'] / 365).hist(bins=40, ax=axes[2], color='green', edgecolor='white')
    axes[2].set_title('Tenure Distribution (Years)')
    axes[2].set_xlabel('Years')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'validation_charts.png', dpi=150)
plt.show()

print('\n✅ Sorvex 360 synthetic pipeline complete!')
print(f'All outputs saved to: {OUTPUT_DIR}')

# ── Cell 13 — Download All Outputs ───────────────────────────────────────────
from google.colab import files
import os

print('Downloading all output files...')
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = OUTPUT_DIR / f
    print(f'  Downloading: {f}')
    files.download(str(path))

print('\n✅ All downloads triggered.')
