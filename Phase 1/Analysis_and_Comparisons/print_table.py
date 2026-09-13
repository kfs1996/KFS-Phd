import pandas as pd

df = pd.read_csv('ULTIMATE_3WAY_COMPARISON.csv')

# Format the floats to percentages
cols_to_format = ['Phase 2-A (Flat)', 'Phase 4-A (Standard 2-Stage)', 'Phase 4-A-New (Self-Correcting)']
for col in cols_to_format:
    df[col] = (df[col] * 100).round(2).astype(str) + '%'

print("### PROMISE DATASET")
print(df[df['Dataset'] == 'PROMISE'].sort_values('Phase 4-A-New (Self-Correcting)', ascending=False).to_markdown(index=False))

print("\n### FNFC DATASET")
print(df[df['Dataset'] == 'FNFC'].sort_values('Phase 4-A-New (Self-Correcting)', ascending=False).to_markdown(index=False))
