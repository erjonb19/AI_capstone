# ============================================================
# Sorvex 360 — OSHA Severe Injury EDA
# ============================================================
#
# Goal: Extract utility-specific injury distributions from the
# OSHA Severe Injury dataset to parameterize Phase 4 fields.
#
# Target fields:
# - OSHA_Recordable_Incident
# - TotalSafetyIncidents
# - TotalPreventableAccidents
# - TotalComplianceViolations
#
# Instructions: Upload your OSHA dataset CSV when prompted

# ── Cell 1 — Install & Import ────────────────────────────────────────────────
import subprocess
subprocess.run(['pip', 'install', 'pandas', 'matplotlib', 'seaborn', '--quiet'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from google.colab import files

OUTPUT_DIR = Path('/content/osha_eda_outputs/')
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style='whitegrid')

# Upload file
uploaded = files.upload()
FILE_PATH = list(uploaded.keys())[0]
print(f'✅ Loaded: {FILE_PATH}')

# ── Cell 2 — Load & Preview ──────────────────────────────────────────────────
df = pd.read_csv(FILE_PATH, low_memory=False)

print(f'Total rows: {len(df):,}')
print(f'Columns ({len(df.columns)}): {list(df.columns)}')
print(f'\nNull counts:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# ── Cell 3 — Filter to Utility NAICS ─────────────────────────────────────────
# NAICS 221x = Core utilities (electric, natural gas, water/wastewater)
# NAICS 2371x = Utility construction crews (lineworkers, NPL, Pike, etc.)
# NAICS 9261x = Government-operated utilities (public water, municipal power)
df['naics_str'] = df['Primary NAICS'].astype(str)

util_primary      = df['naics_str'].str.startswith('221')
util_construction = df['naics_str'].str.startswith('2371')
util_govt         = df['naics_str'].str.startswith('9261')

util_df = df[util_primary | util_construction | util_govt].copy()

util_df['naics_group'] = 'Other'
util_df.loc[util_df['naics_str'].str.startswith('221'),  'naics_group'] = 'Core Utility (221x)'
util_df.loc[util_df['naics_str'].str.startswith('2371'), 'naics_group'] = 'Utility Construction (2371x)'
util_df.loc[util_df['naics_str'].str.startswith('9261'), 'naics_group'] = 'Govt Utility (9261x)'

naics_labels = {
    '2211': 'Electric Power Generation/Transmission',
    '2212': 'Natural Gas Distribution',
    '2213': 'Water/Wastewater Systems'
}

print(f'Utility records: {len(util_df):,} / {len(df):,} total ({len(util_df)/len(df)*100:.1f}%)')
print(f'\nBy group:\n{util_df["naics_group"].value_counts()}')
print(f'\nHospitalization rate by group:')
print(util_df.groupby('naics_group')['Hospitalized'].mean().round(3))

# ── Cell 4 — Severity Distribution ───────────────────────────────────────────
print('=== SEVERITY DISTRIBUTION (Utility Workforce) ===')

hosp_rate = util_df['Hospitalized'].mean()
amp_rate  = util_df['Amputation'].mean()
eye_rate  = util_df['Loss of Eye'].mean()

print(f'\nHospitalization rate: {hosp_rate:.3f} ({hosp_rate*100:.1f}%)')
print(f'Amputation rate:      {amp_rate:.3f} ({amp_rate*100:.1f}%)')
print(f'Loss of eye rate:     {eye_rate:.3f} ({eye_rate*100:.1f}%)')

util_df['severity_tier'] = 'Minor (No Hospitalization)'
util_df.loc[util_df['Hospitalized'] == 1, 'severity_tier'] = 'Moderate (Hospitalized)'
util_df.loc[(util_df['Amputation'] == 1) | (util_df['Loss of Eye'] == 1), 'severity_tier'] = 'Severe (Permanent Injury)'

tier_dist = util_df['severity_tier'].value_counts(normalize=True).round(4)
print(f'\nSeverity tiers:\n{tier_dist}')

severity_params = {
    'hospitalization_rate': round(hosp_rate, 4),
    'amputation_rate':      round(amp_rate, 4),
    'loss_of_eye_rate':     round(eye_rate, 4),
    'severity_tiers':       tier_dist.to_dict()
}
with open(OUTPUT_DIR / 'severity_params.json', 'w') as f:
    json.dump(severity_params, f, indent=2)
print('✅ Saved severity_params.json')

# ── Cell 5 — Event Type Distribution ─────────────────────────────────────────
print('=== EVENT TYPE DISTRIBUTION ===')

event_dist = util_df['EventTitle'].value_counts()
print(f'\nTop 20 event types:\n{event_dist.head(20).to_string()}')

fig, ax = plt.subplots(figsize=(12, 6))
event_dist.head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 OSHA Event Types — Utility Workforce', fontsize=14)
ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'event_type_distribution.png', dpi=150)
plt.show()

def categorize_event(title):
    if pd.isna(title): return 'Other'
    t = title.lower()
    if any(k in t for k in ['electric','arc','shock','voltage']): return 'Electrical'
    if any(k in t for k in ['fall','slip','trip']):               return 'Fall'
    if any(k in t for k in ['caught','compress','pinch']):        return 'Caught/Compressed'
    if any(k in t for k in ['struck','hit','impact']):            return 'Struck By'
    if any(k in t for k in ['exposure','inhal','chemical','burn']): return 'Exposure/Chemical'
    return 'Other'

util_df['event_category'] = util_df['EventTitle'].apply(categorize_event)
cat_dist = util_df['event_category'].value_counts(normalize=True).round(4)
print(f'\nSorvex 360 event categories:\n{cat_dist}')
cat_dist.to_csv(OUTPUT_DIR / 'event_category_weights.csv')

# ── Cell 6 — Injury Nature Distribution ──────────────────────────────────────
nature_dist = util_df['NatureTitle'].value_counts()
print(f'\nTop 15 injury natures:\n{nature_dist.head(15).to_string()}')

fig, ax = plt.subplots(figsize=(12, 6))
nature_dist.head(12).plot(kind='barh', ax=ax, color='coral')
ax.set_title('Top 12 Injury Natures — Utility Workforce', fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'injury_nature_distribution.png', dpi=150)
plt.show()
nature_dist.to_csv(OUTPUT_DIR / 'injury_nature_distribution.csv')

# ── Cell 7 — Body Part Distribution ──────────────────────────────────────────
body_dist = util_df['Part of Body Title'].value_counts()
print(f'\nTop 15 body parts:\n{body_dist.head(15).to_string()}')
body_dist.to_csv(OUTPUT_DIR / 'body_part_distribution.csv')

# ── Cell 8 — Incident Rate by NAICS ──────────────────────────────────────────
util_df['naics_4'] = util_df['naics_str'].str[:4]
by_naics = util_df.groupby('naics_4').agg(
    total_incidents=('ID', 'count'),
    hospitalized=('Hospitalized', 'sum'),
    hosp_rate=('Hospitalized', 'mean'),
).round(3)
print(f'\nIncident rate by NAICS:\n{by_naics.to_string()}')
by_naics.to_csv(OUTPUT_DIR / 'incident_rate_by_naics.csv')

# ── Cell 9 — Incident Trends Over Time ───────────────────────────────────────
util_df['EventDate'] = pd.to_datetime(util_df['EventDate'], errors='coerce')
util_df['year'] = util_df['EventDate'].dt.year

yearly = util_df.groupby('year').agg(
    total=('ID', 'count'),
    hosp_rate=('Hospitalized', 'mean')
).round(3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
yearly['total'].plot(ax=axes[0], marker='o', color='steelblue')
axes[0].set_title('Total Utility Incidents by Year')
yearly['hosp_rate'].plot(ax=axes[1], marker='o', color='coral')
axes[1].set_title('Hospitalization Rate by Year')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'incident_trends.png', dpi=150)
plt.show()

# ── Cell 10 — Export Sorvex 360 Parameters ───────────────────────────────────
summary = {
    'source': 'OSHA Severe Injury Dataset — Utility NAICS 221x + 2371x + 9261x',
    'total_utility_records': len(util_df),
    'OSHA_Recordable_Incident': {
        'hospitalization_rate': round(util_df['Hospitalized'].mean(), 4),
        'amputation_rate':      round(util_df['Amputation'].mean(), 4),
        'loss_of_eye_rate':     round(util_df['Loss of Eye'].mean(), 4),
    },
    'TotalPreventableAccidents_severity_tiers': tier_dist.to_dict(),
    'TotalComplianceViolations_event_categories': cat_dist.to_dict(),
    'top_injury_natures': nature_dist.head(10).to_dict(),
    'top_body_parts':     body_dist.head(10).to_dict(),
}

with open(OUTPUT_DIR / 'sorvex360_phase4_osha_params.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f'\n✅ All outputs saved to {OUTPUT_DIR}')

# Download all outputs
for f in OUTPUT_DIR.iterdir():
    files.download(str(f))
