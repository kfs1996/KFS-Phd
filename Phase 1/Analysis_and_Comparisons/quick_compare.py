import pandas as pd

try:
    df_new = pd.read_csv('FINAL_Phase4_A_New_SelfCorrecting_Results.csv')
    df_new = df_new.rename(columns={'Average': 'Phase 4-A-New'})
    
    # Try to load Phase 4 Optuna as Phase 4-A baseline
    df_4a = pd.read_csv('FINAL_Phase4_2A_Classical_Optuna_Results.csv')
    df_4a = df_4a.rename(columns={'Average': 'Phase 4-A (Optuna)'})
    
    # Merge them
    merged = pd.merge(df_4a[['Dataset', 'Algorithm', 'Vectorization', 'Phase 4-A (Optuna)']], 
                      df_new[['Dataset', 'Algorithm', 'Vectorization', 'Phase 4-A-New']], 
                      on=['Dataset', 'Algorithm', 'Vectorization'])
                      
    merged['Diff'] = merged['Phase 4-A-New'] - merged['Phase 4-A (Optuna)']
    
    print("--- TOP PROMISE RESULTS ---")
    print(merged[merged['Dataset'] == 'PROMISE'].sort_values('Phase 4-A-New', ascending=False).head(5).to_markdown(index=False))
    
    print("\n--- TOP FNFC RESULTS ---")
    print(merged[merged['Dataset'] == 'FNFC'].sort_values('Phase 4-A-New', ascending=False).head(5).to_markdown(index=False))
except Exception as e:
    print("Error:", e)
