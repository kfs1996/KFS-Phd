import pandas as pd
import re
import os

log_path = r"C:\Users\Fskir\.gemini\antigravity\brain\36ba3f84-34f2-4102-99ae-df58671f3ac7\.system_generated\tasks\task-5959.log"

phase2c_data = []

# 1. Parse from the original frozen log
if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r'\[(.*?)\] (.*?) \+ (.*?) -> Phase 2-C Acc: ([\d\.]+)', line)
            if match:
                dataset, algo, vectorization, acc = match.groups()
                phase2c_data.append({
                    'Dataset': dataset.strip(),
                    'Algorithm': algo.strip(),
                    'Vectorization': vectorization.strip(),
                    'Phase 2-C': float(acc)
                })

# 2. Add the newly generated missing data (FNFC GloVe & Word2Vec)
missing_data_path = r"D:\phd presentations\phase4code\FINAL_Phase2_C_MissingData.csv"
if os.path.exists(missing_data_path):
    df_missing = pd.read_csv(missing_data_path)
    for _, row in df_missing.iterrows():
        phase2c_data.append({
            'Dataset': row['Dataset'],
            'Algorithm': row['Algorithm'],
            'Vectorization': row['Vectorization'],
            'Phase 2-C': float(row['Phase 2-C'])
        })

df_2c = pd.DataFrame(phase2c_data)
# Drop duplicates in case of retries, keeping the latest successful run
df_2c = df_2c.drop_duplicates(subset=['Dataset', 'Algorithm', 'Vectorization'], keep='last')

# Load Phase 4-2C
df_42c = pd.read_csv(r"D:\phd presentations\phase4code\FINAL_Phase4_2C_Simple_Optuna_Results.csv")

# Merge
merged = pd.merge(df_42c, df_2c, on=['Dataset', 'Algorithm', 'Vectorization'], how='left')

# Sort by Phase 2-C highest to lowest first
merged = merged.sort_values(by=['Dataset', 'Phase 2-C'], ascending=[True, False])

# Format
merged['Phase 4-2C (Adaptive 2-Stage)'] = (merged['Phase 4-2C (Adaptive 2-Stage)'] * 100).round(2).astype(str) + '%'
merged['Phase 2-C'] = merged['Phase 2-C'].apply(lambda x: f"{(x * 100):.2f}%" if pd.notnull(x) else "N/A")

# Sort for output
for dataset in ['PROMISE', 'FNFC']:
    print(f"\n### --- {dataset} DATASET (Phase 2-C vs Phase 4-2C) Highest to Lowest ---")
    sub_df = merged[merged['Dataset'] == dataset][['Algorithm', 'Vectorization', 'Phase 2-C', 'Phase 4-2C (Adaptive 2-Stage)']]
    print(sub_df.to_markdown(index=False))

