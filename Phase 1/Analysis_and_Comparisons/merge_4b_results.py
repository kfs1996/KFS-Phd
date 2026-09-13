import pandas as pd
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 1. Parse Phase 2-B and Phase 4-B from the Markdown artifact
md_path = r"C:\Users\Fskir\.gemini\antigravity\brain\36ba3f84-34f2-4102-99ae-df58671f3ac7\phase2b_vs_phase4b_comparison.md"
with open(md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []
current_dataset = None
for line in lines:
    if line.startswith("## PROMISE Dataset"):
        current_dataset = "PROMISE"
    elif line.startswith("## FNFC Dataset"):
        current_dataset = "FNFC"
    elif line.startswith("|") and "Algorithm" not in line and "---" not in line:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4 and current_dataset:
            algo = parts[0]
            emb = parts[1]
            acc_2b = parts[2].replace('%', '')
            acc_4b = parts[3].replace('%', '')
            data.append({
                'Dataset': current_dataset,
                'Algorithm': algo,
                'Vectorization': emb,
                'Phase 2-B (1-Stage Native)': float(acc_2b),
                'Phase 4-B (2-Stage Native)': float(acc_4b)
            })

df_base = pd.DataFrame(data)

# 2. Read Phase 4-B-New
df_new = pd.read_csv('FINAL_Phase4_B_New_SelfCorrecting_Results.csv')
df_new['Phase 4-B-New'] = (df_new['Phase 4-B-New'] * 100).round(2)

# Merge
merged = pd.merge(df_base, df_new, on=['Dataset', 'Algorithm', 'Vectorization'], how='inner')

# Format
for col in ['Phase 2-B (1-Stage Native)', 'Phase 4-B (2-Stage Native)', 'Phase 4-B-New']:
    merged[col] = merged[col].astype(str) + '%'

print("### PROMISE DATASET")
print(merged[merged['Dataset'] == 'PROMISE'].sort_values('Phase 4-B-New', ascending=False).to_markdown(index=False))
print("\n### FNFC DATASET")
print(merged[merged['Dataset'] == 'FNFC'].sort_values('Phase 4-B-New', ascending=False).to_markdown(index=False))
