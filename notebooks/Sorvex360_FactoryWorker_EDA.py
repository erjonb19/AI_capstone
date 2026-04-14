# ============================================================
# Sorvex 360 — Factory Worker Behavior EDA
# ============================================================
#
# Goal: Aggregate daily behavioral records into per-worker
# summary statistics to parameterize Phase 4 fields.
#
# Target fields:
# - UnscheduledAbsences
# - ManagerFitFeedback_Score
# - PromotionWithin24Months
# - Time_To_Competency_Months
# - ProbationStatus
#
# Instructions: Upload your factory worker CSV when prompted

# ── Cell 1 — Install & Import ────────────────────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', 'pandas', 'matplotlib', 'seaborn', 'scipy', '--quiet'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
from google.colab import files

OUTPUT_DIR = Path('/content/factory_eda_outputs/')
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

# Upload file
uploaded = files.upload()
FILE_PATH = list(uploaded.keys())[0]
print(f'✅ Loaded: {FILE_PATH}')

# ── Cell 2 — Load & Preview ──────────────────────────────────────────────────
# latin-1 encoding handles Windows special characters in this dataset
df = pd.read_csv(FILE_PATH, low_memory=False, encoding='latin-1')

print(f'Total rows: {len(df):,}')
print(f'Unique workers (sub_ID): {df["sub_ID"].nunique():,}')
print(f'Columns: {list(df.columns)}')
print(f'\nEvent type distribution:')
print(df['record_comptype'].value_counts())
print(f'\nConf matrix breakdown:')
print(df['record_conf_matrix_h'].value_counts())

# ── Cell 3 — Worker Trait Distributions ──────────────────────────────────────
# Static per-worker fields — take first record per worker
worker_traits = df.groupby('sub_ID').first()[[
    'sub_age', 'sub_sex', 'sub_shift', 'sub_role',
    'sub_health_h', 'sub_commitment_h', 'sub_perceptiveness_h',
    'sub_dexterity_h', 'sub_sociality_h', 'sub_goodness_h',
    'sub_strength_h', 'sub_openmindedness_h', 'sub_workstyle_h'
]].reset_index()

print(f'Unique workers: {len(worker_traits):,}')
print(f'\nAge distribution:\n{worker_traits["sub_age"].describe()}')
print(f'\nSex distribution:\n{worker_traits["sub_sex"].value_counts(normalize=True).round(3)}')

# Numeric traits only (sub_workstyle_h is categorical — Group A/B/C/D)
trait_cols = [
    'sub_health_h', 'sub_commitment_h', 'sub_perceptiveness_h',
    'sub_dexterity_h', 'sub_sociality_h', 'sub_goodness_h',
    'sub_strength_h', 'sub_openmindedness_h'
]

print(f'\nTrait distributions (0-1 scale):')
print(worker_traits[trait_cols].describe().round(3))

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, col in enumerate(trait_cols):
    ax = axes[i//3][i%3]
    worker_traits[col].hist(bins=30, ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(col.replace('sub_','').replace('_h','').title())
# Plot workstyle (categorical) in last subplot
axes[2][2].bar(worker_traits['sub_workstyle_h'].value_counts().index,
               worker_traits['sub_workstyle_h'].value_counts().values, color='steelblue')
axes[2][2].set_title('Workstyle')
plt.suptitle('Worker Trait Distributions', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trait_distributions.png', dpi=150)
plt.show()

# ── Cell 4 — Efficacy Distribution ───────────────────────────────────────────
efficacy_df = df[df['record_comptype'] == 'Efficacy'].copy()
efficacy_df['recorded_efficacy'] = pd.to_numeric(efficacy_df['recorded_efficacy'], errors='coerce')
efficacy_df['actual_efficacy_h'] = pd.to_numeric(efficacy_df['actual_efficacy_h'], errors='coerce')

print('=== EFFICACY DISTRIBUTIONS ===')
print(f'\nRecorded efficacy:\n{efficacy_df["recorded_efficacy"].describe().round(3)}')
print(f'\nActual efficacy (ground truth):\n{efficacy_df["actual_efficacy_h"].describe().round(3)}')

efficacy_df['efficacy_gap'] = efficacy_df['actual_efficacy_h'] - efficacy_df['recorded_efficacy']
print(f'\nSupervisor accuracy gap:\n{efficacy_df["efficacy_gap"].describe().round(3)}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
efficacy_df['recorded_efficacy'].hist(bins=40, ax=axes[0], color='steelblue', edgecolor='white')
axes[0].set_title('Recorded Efficacy')
efficacy_df['actual_efficacy_h'].hist(bins=40, ax=axes[1], color='coral', edgecolor='white')
axes[1].set_title('Actual Efficacy (Ground Truth)')
efficacy_df['efficacy_gap'].hist(bins=40, ax=axes[2], color='green', edgecolor='white')
axes[2].set_title('Supervisor Accuracy Gap')
axes[2].axvline(0, color='red', linestyle='--', label='Perfect accuracy')
axes[2].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'efficacy_distributions.png', dpi=150)
plt.show()

# ── Cell 5 — Per-Worker Aggregation ──────────────────────────────────────────
print('Building per-worker summary stats...')

df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
df['recorded_efficacy_num'] = pd.to_numeric(df['recorded_efficacy'], errors='coerce')
df['actual_efficacy_num']   = pd.to_numeric(df['actual_efficacy_h'], errors='coerce')

total_days = df.groupby('sub_ID')['event_date'].nunique().rename('total_days')
presence   = df[df['record_comptype'] == 'Presence'].groupby('sub_ID').size().rename('presence_days')

eff_stats = df[df['record_comptype'] == 'Efficacy'].groupby('sub_ID').agg(
    mean_recorded_efficacy=('recorded_efficacy_num', 'mean'),
    mean_actual_efficacy=('actual_efficacy_num', 'mean'),
    std_efficacy=('recorded_efficacy_num', 'std'),
    efficacy_records=('recorded_efficacy_num', 'count')
).round(4)

feats = df[df['record_comptype'] == 'Feat'].groupby('sub_ID').size().rename('feat_count')
tp    = df[df['record_conf_matrix_h'] == 'True Positive'].groupby('sub_ID').size().rename('tp_count')
fp    = df[df['record_conf_matrix_h'] == 'False Positive'].groupby('sub_ID').size().rename('fp_count')
fn    = df[df['record_conf_matrix_h'] == 'False Negative'].groupby('sub_ID').size().rename('fn_count')

worker_summary = pd.concat([
    worker_traits.set_index('sub_ID'),
    total_days, presence, eff_stats, feats, tp, fp, fn
], axis=1).fillna(0).reset_index()

worker_summary['feat_rate'] = (
    worker_summary['feat_count'] / worker_summary['total_days'].clip(lower=1)
).round(4)

worker_summary['supervisor_accuracy'] = (
    worker_summary['tp_count'] / (
        worker_summary['tp_count'] + worker_summary['fp_count'] + worker_summary['fn_count']
    ).clip(lower=1)
).round(4)

print(f'Per-worker summary built: {len(worker_summary):,} workers')
worker_summary.to_csv(OUTPUT_DIR / 'worker_summary.csv', index=False)
print('✅ Saved worker_summary.csv')

# ── Cell 6 — Manager Feedback Score ──────────────────────────────────────────
print('=== MANAGER FIT FEEDBACK SCORE ===')
print(worker_summary['mean_recorded_efficacy'].describe().round(4))

e_min = worker_summary['mean_recorded_efficacy'].min()
e_max = worker_summary['mean_recorded_efficacy'].max()
worker_summary['manager_score_1_5'] = (
    1 + 4 * (worker_summary['mean_recorded_efficacy'] - e_min) / (e_max - e_min)
).round(2)

print(f'\nManagerFitFeedback_Score (1-5 normalized):')
print(worker_summary['manager_score_1_5'].describe().round(3))

fig, ax = plt.subplots(figsize=(10, 5))
worker_summary['manager_score_1_5'].hist(bins=40, ax=ax, color='steelblue', edgecolor='white')
ax.set_title('ManagerFitFeedback_Score Distribution (1-5 Scale)')
ax.axvline(worker_summary['manager_score_1_5'].mean(), color='red', linestyle='--',
           label=f'Mean: {worker_summary["manager_score_1_5"].mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'manager_score_distribution.png', dpi=150)
plt.show()

# ── Cell 7 — Absence Distribution ────────────────────────────────────────────
# NOTE: absence_rate is a DATA ARTIFACT in this dataset (96% mean)
# The Presence and Efficacy event types are logged separately per day
# causing inflated "absence" counts. Use BLS data for UnscheduledAbsences instead.
print('NOTE: Absence rate from this dataset is not reliable for Sorvex 360.')
print('Use BLS utility industry absence anchor (4-6 days/year) instead.')

# ── Cell 8 — Trait Correlations with Outcomes ─────────────────────────────────
print('=== TRAIT -> OUTCOME CORRELATIONS ===')

outcome_cols = ['mean_recorded_efficacy', 'feat_rate', 'supervisor_accuracy']
corr_cols    = trait_cols + outcome_cols
corr_matrix  = worker_summary[corr_cols].corr().round(3)

trait_outcome_corr = corr_matrix.loc[trait_cols, outcome_cols]
print(trait_outcome_corr.to_string())

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(trait_outcome_corr, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax, linewidths=0.5)
ax.set_title('Worker Trait -> Outcome Correlations\n(for CTGAN conditioning)', fontsize=13)
ax.set_yticklabels([c.replace('sub_','').replace('_h','').title() for c in trait_cols], rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trait_outcome_correlations.png', dpi=150)
plt.show()
trait_outcome_corr.to_csv(OUTPUT_DIR / 'trait_outcome_correlations.csv')

# ── Cell 9 — Promotion & Probation Proxies ────────────────────────────────────
eff_75 = worker_summary['mean_recorded_efficacy'].quantile(0.75)
worker_summary['promotion_proxy'] = (
    (worker_summary['mean_recorded_efficacy'] >= eff_75) &
    (worker_summary['feat_count'] > 0)
).astype(int)

promo_rate = worker_summary['promotion_proxy'].mean()
print(f'Promotion proxy rate: {promo_rate:.3f} ({promo_rate*100:.1f}%)')

# Probation proxy: bottom quartile efficacy in first 30 days
worker_start = df.groupby('sub_ID')['event_date'].min().rename('start_date')
df = df.join(worker_start, on='sub_ID')
df['days_in'] = (df['event_date'] - df['start_date']).dt.days

early_efficacy = df[
    (df['days_in'] <= 30) & (df['record_comptype'] == 'Efficacy')
].groupby('sub_ID')['recorded_efficacy_num'].mean().rename('early_efficacy')

eff_25 = early_efficacy.quantile(0.25)
probation_flags = (early_efficacy <= eff_25).astype(int).rename('probation_proxy')
probation_rate  = probation_flags.mean()
print(f'Probation proxy rate: {probation_rate:.3f} ({probation_rate*100:.1f}%)')

# ── Cell 10 — Export Parameters ───────────────────────────────────────────────
summary_params = {
    'source': 'Factory Workers Daily Performance & Attrition Dataset',
    'total_workers': len(worker_summary),
    'ManagerFitFeedback_Score': {
        'scale':  '1-5 (normalized from recorded_efficacy)',
        'mean':   round(worker_summary['manager_score_1_5'].mean(), 3),
        'std':    round(worker_summary['manager_score_1_5'].std(), 3),
        'median': round(worker_summary['manager_score_1_5'].median(), 3)
    },
    'PromotionWithin24Months': {
        'rate':     round(promo_rate, 4),
        'criteria': 'top quartile efficacy + at least 1 feat event'
    },
    'ProbationStatus': {
        'rate':     round(float(probation_rate), 4),
        'criteria': 'bottom 25% efficacy in first 30 days'
    },
    'UnscheduledAbsences': {
        'NOTE': 'Do NOT use absence_rate from this dataset — data artifact (96% mean)',
        'Use':  'BLS utility industry anchor: 4-6 unscheduled days/year'
    },
    'trait_outcome_correlations': trait_outcome_corr.to_dict()
}

with open(OUTPUT_DIR / 'sorvex360_phase4_factory_params.json', 'w') as f:
    json.dump(summary_params, f, indent=2)

print(json.dumps(
    {k: v for k, v in summary_params.items() if k != 'trait_outcome_correlations'},
    indent=2
))
print(f'\n✅ All outputs saved to {OUTPUT_DIR}')

# Download all outputs
for f in OUTPUT_DIR.iterdir():
    files.download(str(f))
