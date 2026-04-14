# ============================================================
# Sorvex 360 — Vertex AI Training Pipeline
# Predict • Prepare • Place • Post-Hire Outcomes
# ============================================================
#
# Architecture:
# - Train scikit-learn models locally in Workbench (no training job costs)
# - Register models to Vertex AI Model Registry (free)
# - Deploy endpoint only to test, then delete immediately
#
# Three prediction targets:
# - Tenure_1Year           — will candidate stay 12+ months? (binary)
# - OSHA_Recordable_Incident — safety risk flag (binary)
# - PromotionWithin24Months  — promotion likelihood (binary)
#
# Before running: update PROJECT_ID, BUCKET_NAME, REGION in Cell 1

# ── Cell 1 — Config & Install ─────────────────────────────────────────────────
PROJECT_ID  = 'sorvex360-493312'      # GCP project ID
BUCKET_NAME = 'sorvex360-raw-data'    # GCS bucket name
REGION      = 'us-central1'           # region
BQ_DATASET  = 'sorvex_ml'
BQ_TABLE    = 'training_table'
MODEL_NAME  = 'sorvex360-retention-model'

import subprocess
subprocess.run(['pip', 'install', 'google-cloud-aiplatform', 'google-cloud-bigquery',
                'google-cloud-storage', 'scikit-learn', 'pandas', 'numpy', 'joblib', '--quiet'])

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = Path('/tmp/sorvex_models')
MODEL_DIR.mkdir(exist_ok=True)

print(f'✅ Config set. Project: {PROJECT_ID} | Region: {REGION}')

# ── Cell 2 — Load Data from GCS ───────────────────────────────────────────────
from google.cloud import storage

GCS_PATH = 'incoming/Sorvex360_Master_Clean.csv'  # file location in bucket

print(f'Loading data from gs://{BUCKET_NAME}/{GCS_PATH}...')

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)
blob   = bucket.blob(GCS_PATH)
blob.download_to_filename('/tmp/master_clean.csv')

df = pd.read_csv('/tmp/master_clean.csv')
print(f'✅ Loaded: {len(df):,} rows x {len(df.columns)} columns')
print(f'\nTarget variable distributions:')
for target in ['Tenure_1Year', 'OSHA_Recordable_Incident', 'PromotionWithin24Months']:
    rate = df[target].mean()
    print(f'  {target}: {rate:.1%} positive')

# ── Cell 3 — Feature Engineering ──────────────────────────────────────────────
print('Engineering features...')

NUMERIC_FEATURES = [
    'Age', 'CognitiveScore', 'SimulationScore', 'BehavioralScore',
    'SituationalScore', 'Sorvex360PI_Score', 'LongestJobTenure',
    'HasPriorTradeExperience', 'VeteranStatus', 'CDL_Status',
    'OSHA10_Status', 'CPR_Status', 'ApprenticeshipInterest',
    'CanPassDrugScreen', 'CanPassBackgroundCheck',
    'TotalTrainingHours', 'AttendanceRate', 'PassedRequiredModules',
    'CertificationsEarned', 'SimulationPerformance', 'TeamworkScore',
    'SafetyCommitmentScore', 'ReliabilityScore', 'PhysicalTestResult',
    'Lift50lbsTest', 'Completed', 'ReadinessDelta',
    'Sorvex360PI_Score_AtCompletion',
    'UnionStatus', 'RoleRequires_CDL', 'RoleRequires_OSHA10',
    'RoleRequires_CPR', 'PreHire_Verified_MVR', 'PreHire_Verified_DrugScreen',
    'PreHire_Verified_Background', 'Orientation_LOTO_Completed',
    'Orientation_PPE_Fitted', 'Apprenticeship_Registered',
    'Sorvex360PI_ScoreAtHire',
]

CATEGORICAL_FEATURES = [
    'SOC_Code', 'Gender', 'EducationLevel', 'MostRecentIndustry',
    'TrainingSource', 'SourceOfCandidate', 'EmploymentType',
    'HasValidLicense',  # stored as string in this dataset
]

TARGETS = {
    'Tenure_1Year':             'retention',
    'OSHA_Recordable_Incident': 'safety',
    'PromotionWithin24Months':  'promotion',
}

NUMERIC_FEATURES    = [f for f in NUMERIC_FEATURES    if f in df.columns]
CATEGORICAL_FEATURES = [f for f in CATEGORICAL_FEATURES if f in df.columns]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

X = df[ALL_FEATURES].copy()
X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(0)
X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna('Unknown')

print(f'Numeric features: {len(NUMERIC_FEATURES)}')
print(f'Categorical features: {len(CATEGORICAL_FEATURES)}')
print(f'Feature matrix shape: {X.shape}')
print('✅ Features ready.')

# ── Cell 4 — Build Preprocessing Pipeline ─────────────────────────────────────
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, NUMERIC_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES),
])

print('✅ Preprocessing pipeline built.')

# ── Cell 5 — Train Models for All 3 Targets ───────────────────────────────────
trained_models = {}
results = {}

print('✅ Preprocessor ready')

for target, label in TARGETS.items():
    print(f'\n── Training: {target} ({label}) ──────────────────────')

    y = df[target].astype(int)
    pos_rate = y.mean()
    print(f'  Positive rate: {pos_rate:.1%}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=360, stratify=y
    )

    # Sample weights to handle class imbalance
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=360
        ))
    ])

    # Train with sample weights
    model.fit(X_train, y_train,
              classifier__sample_weight=sample_weights)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(f'  AUC-ROC: {auc:.4f}')
    print(f'  Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['No','Yes']))

    # 3-fold CV (faster than 5-fold, use n_jobs=1 to avoid hanging)
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc', n_jobs=1)
    print(f'  3-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    model_path = MODEL_DIR / f'sorvex360_{label}_model.joblib'
    joblib.dump(model, model_path)

    trained_models[target] = model
    results[target] = {
        'label':    label,
        'auc':      round(auc, 4),
        'cv_mean':  round(cv_scores.mean(), 4),
        'cv_std':   round(cv_scores.std(), 4),
        'model_path': str(model_path)
    }

print('\n✅ All 3 models trained and saved.')

# ── Cell 6 — Feature Importance ───────────────────────────────────────────────
print('=== FEATURE IMPORTANCE ANALYSIS ===')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (target, label) in zip(axes, TARGETS.items()):
    model = trained_models[target]
    clf   = model.named_steps['classifier']
    prep  = model.named_steps['preprocessor']

    cat_names    = prep.named_transformers_['cat']['onehot'].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERIC_FEATURES + list(cat_names)

    importances = clf.feature_importances_
    top_idx     = np.argsort(importances)[-15:]
    top_names   = [feature_names[i] for i in top_idx]
    top_imp     = importances[top_idx]

    ax.barh(top_names, top_imp, color='steelblue')
    ax.set_title(f'{target}\nTop 15 Features', fontsize=11)
    ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Feature importance charts saved.')

# ── Cell 7 — Save Results Summary ─────────────────────────────────────────────
print('=== MODEL PERFORMANCE SUMMARY ===')
for target, res in results.items():
    print(f"\n{target}:")
    print(f"  AUC-ROC: {res['auc']}")
    print(f"  CV AUC:  {res['cv_mean']} ± {res['cv_std']}")

with open(MODEL_DIR / 'model_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n✅ Results saved to model_results.json')

# ── Cell 8 — Upload Models to GCS ─────────────────────────────────────────────
print('Uploading models to GCS...')

bucket = storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME)

for target, res in results.items():
    local_path = res['model_path']
    gcs_path   = f"models/sorvex360_{res['label']}_model.joblib"
    bucket.blob(gcs_path).upload_from_filename(local_path)
    print(f'  ✅ gs://{BUCKET_NAME}/{gcs_path}')

bucket.blob('models/model_results.json').upload_from_filename(
    str(MODEL_DIR / 'model_results.json'))
bucket.blob('models/feature_importance.png').upload_from_filename(
    str(MODEL_DIR / 'feature_importance.png'))

print(f'\n✅ All models uploaded to gs://{BUCKET_NAME}/models/')

# ── Cell 9 — Register Models in Vertex AI Model Registry ──────────────────────
from google.cloud import aiplatform

aiplatform.init(project=PROJECT_ID, location=REGION)

print('Registering models in Vertex AI Model Registry...\n')

registered_models = {}

for target, res in results.items():
    label = res['label']

    # Copy each model to its own subfolder as model.joblib
    src_blob  = bucket.blob(f'models/sorvex360_{label}_model.joblib')
    dest_blob = bucket.blob(f'models/{label}/model.joblib')
    bucket.copy_blob(src_blob, bucket, dest_blob.name)
    print(f'  Copied to gs://{BUCKET_NAME}/models/{label}/model.joblib')

    model = aiplatform.Model.upload(
        display_name=f'sorvex360-{label}-v1',
        artifact_uri=f'gs://{BUCKET_NAME}/models/{label}/',
        serving_container_image_uri=(
            'us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest'
        ),
        description=(
            f'Sorvex 360 {target} prediction model. '
            f'GBM classifier. AUC={res["auc"]}. '
            f'Trained on 5,000 synthetic utility workforce profiles.'
        ),
        labels={
            'target':   label,
            'version':  'v1',
            'pipeline': 'sorvex360',
        }
    )

    registered_models[target] = model.resource_name
    print(f'  ✅ Registered: {model.display_name}')
    print(f'     Resource: {model.resource_name}')
    print(f'     AUC: {res["auc"]}\n')

print('✅ All models registered in Vertex AI Model Registry.')

# ── Cell 10 — Load Training Data into BigQuery ────────────────────────────────
from google.cloud import bigquery

bq_client = bigquery.Client(project=PROJECT_ID)

print('Loading clean master file into BigQuery...')

table_id = f'{PROJECT_ID}.sorvex_raw.master_clean'
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
)

gcs_uri  = f'gs://{BUCKET_NAME}/incoming/Sorvex360_Master_Clean.csv'
load_job = bq_client.load_table_from_uri(gcs_uri, table_id, job_config=job_config)
load_job.result()

table = bq_client.get_table(table_id)
print(f'✅ Loaded {table.num_rows:,} rows into {table_id}')

# ── Cell 11 — Test Prediction (optional) ──────────────────────────────────────
# Set DEPLOY_TEST = True only when you want to test a live prediction
# Endpoint will be deleted after test to avoid ongoing costs
DEPLOY_TEST = False

if DEPLOY_TEST:
    print('Deploying retention model for test prediction...')
    retention_model = aiplatform.Model(model_name=registered_models['Tenure_1Year'])

    endpoint = retention_model.deploy(
        machine_type='n1-standard-2',
        min_replica_count=1,
        max_replica_count=1,
        deployed_model_display_name='sorvex360-retention-test',
    )

    test_instance = X.iloc[0][ALL_FEATURES].tolist()
    prediction    = endpoint.predict(instances=[test_instance])

    print(f'\n=== TEST PREDICTION ===')
    print(f'Prediction (Tenure_1Year): {prediction.predictions}')
    print(f'Actual: {df.iloc[0]["Tenure_1Year"]}')

    print('\nDeleting endpoint to stop billing...')
    endpoint.delete(force=True)
    print('✅ Endpoint deleted.')
else:
    print('DEPLOY_TEST=False — skipping endpoint deployment.')

# ── Cell 12 — Final Summary ────────────────────────────────────────────────────
print('=' * 60)
print('SORVEX 360 — VERTEX AI PIPELINE COMPLETE')
print('=' * 60)

print('\n📊 Models Trained:')
for target, res in results.items():
    print(f"  {target}")
    print(f"    AUC-ROC: {res['auc']} | CV: {res['cv_mean']} ± {res['cv_std']}")

print(f'\n☁️  GCS Artifacts:')
print(f'  gs://{BUCKET_NAME}/incoming/Sorvex360_Master_Clean.csv')
print(f'  gs://{BUCKET_NAME}/models/sorvex360_retention_model.joblib')
print(f'  gs://{BUCKET_NAME}/models/sorvex360_safety_model.joblib')
print(f'  gs://{BUCKET_NAME}/models/sorvex360_promotion_model.joblib')

print(f'\n🗄️  BigQuery Tables:')
print(f'  {PROJECT_ID}.sorvex_raw.master_clean')
print(f'  {PROJECT_ID}.sorvex_ml.training_table')

print(f'\n🤖  Vertex AI Model Registry:')
for target, resource_name in registered_models.items():
    print(f'  {target}: {resource_name}')

print('\n✅ Pipeline complete!')
