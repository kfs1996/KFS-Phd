import pandas as pd

# Load Phase 4-2C
df_42c = pd.read_csv("FINAL_Phase4_2C_Simple_Optuna_Results.csv")
df_42c["Phase 4-2C (Adaptive 2-Stage)"] = (df_42c["Phase 4-2C (Adaptive 2-Stage)"] * 100).round(2).astype(str) + "%"

# Check if we can find Phase 2-A (Flat Baseline) to compare since 2-C full 72-grid isn't saved
df_ultimate = pd.read_csv("ULTIMATE_3WAY_COMPARISON.csv")
df_ultimate["Phase 2-A (Flat)"] = (df_ultimate["Phase 2-A (Flat)"] * 100).round(2).astype(str) + "%"

# Merge
merged = pd.merge(df_ultimate[['Dataset', 'Algorithm', 'Vectorization', 'Phase 2-A (Flat)']], df_42c, on=['Dataset', 'Algorithm', 'Vectorization'], how='inner')

with open("phase4_2c_full_comparison.md", "w", encoding="utf-8") as f:
    f.write("# Full Comparison: Phase 2-A (Flat Baseline) vs Phase 4-2C (Simple Adaptive 2-Stage)\n\n")
    f.write("*(Note: The full 72-combination matrix for Phase 2-C was not persisted in the workspace, so Phase 2-A is used as the 1-Stage flat baseline for comparison).* \n\n")
    
    f.write("## PROMISE DATASET\n")
    f.write(merged[merged["Dataset"] == "PROMISE"].sort_values("Phase 4-2C (Adaptive 2-Stage)", ascending=False).to_markdown(index=False))
    
    f.write("\n\n## FNFC DATASET\n")
    f.write(merged[merged["Dataset"] == "FNFC"].sort_values("Phase 4-2C (Adaptive 2-Stage)", ascending=False).to_markdown(index=False))
    
print("Artifact generated!")
