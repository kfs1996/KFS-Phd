import os
import pandas as pd
import numpy as np
import warnings
import optuna
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_base_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def run_class_wise_analysis():
    print("Loading FNFC and TF-IDF...")
    df = pd.read_csv(r"D:\phd presentations\datasets\FNFC.csv", encoding='latin1')
    label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
    text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
    
    df[label_col] = df[label_col].astype(str).str.strip()
    df['text'] = df[text_col].astype(str).fillna("")
    y_multi = df[label_col].values
    
    vec = TfidfVectorizer(max_features=5000)
    X = vec.fit_transform(df['text'].tolist()).toarray()
    
    # ---- PHASE 2-C (FLAT) ----
    print("\nCalculating Phase 2-C (Flat) Class-Wise Accuracy...")
    def obj_2c(trial):
        w_dict = get_base_weights(y_multi)
        for cls in np.unique(y_multi):
            w_dict[cls] *= trial.suggest_float(f'w_{cls}', 0.5, 2.0)
        c_val = trial.suggest_float('C', 0.1, 10.0, log=True)
        
        model = LinearSVC(C=c_val, class_weight=w_dict, random_state=42)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        preds = cross_val_predict(model, X, y_multi, cv=skf)
        return accuracy_score(y_multi, preds)

    study_2c = optuna.create_study(direction="maximize")
    study_2c.optimize(obj_2c, n_trials=15)
    
    # Re-run best 2C model
    w_dict_2c = get_base_weights(y_multi)
    for cls in np.unique(y_multi):
        w_dict_2c[cls] *= study_2c.best_params[f'w_{cls}']
    best_model_2c = LinearSVC(C=study_2c.best_params['C'], class_weight=w_dict_2c, random_state=42)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    preds_2c = cross_val_predict(best_model_2c, X, y_multi, cv=skf)
    print("\n### Phase 2-C (Flat) Classification Report ###")
    print(classification_report(y_multi, preds_2c, zero_division=0))
    print(f"Overall Phase 2-C Accuracy: {accuracy_score(y_multi, preds_2c)*100:.2f}%")


    # ---- PHASE 4-2C (SIMPLE 2-STAGE) ----
    print("\nCalculating Phase 4-2C (Simple 2-Stage) Class-Wise Accuracy...")
    fr_label = next((l for l in np.unique(y_multi) if str(l).strip().upper() in ['F', 'FR', 'FUNCTIONAL']), 'FR')
    y_binary = np.array([fr_label if label == fr_label else 'NFR' for label in y_multi])
    
    def obj_42c(trial):
        # Stage 1
        w_b = get_base_weights(y_binary)
        w_b[fr_label] *= trial.suggest_float('w_FR', 0.5, 2.0)
        w_b['NFR'] *= trial.suggest_float('w_NFR', 0.5, 2.0)
        c_s1 = trial.suggest_float('C_1', 0.1, 10.0, log=True)
        s1 = LinearSVC(C=c_s1, class_weight=w_b, random_state=42)
        
        # Stage 2 (Only on NFRs)
        nfr_indices = (y_multi != fr_label)
        y_nfr = y_multi[nfr_indices]
        w_m = get_base_weights(y_nfr)
        for cls in np.unique(y_nfr):
            w_m[cls] *= trial.suggest_float(f'w_{cls}_s2', 0.5, 2.0)
        c_s2 = trial.suggest_float('C_2', 0.1, 10.0, log=True)
        s2 = LinearSVC(C=c_s2, class_weight=w_m, random_state=42)
        
        accs = []
        for train_idx, val_idx in skf.split(X, y_multi):
            X_train, X_val = X[train_idx], X[val_idx]
            yb_train = y_binary[train_idx]
            ym_train = y_multi[train_idx]
            ym_val = y_multi[val_idx]
            
            s1.fit(X_train, yb_train)
            preds_s1 = s1.predict(X_val)
            
            nfr_train_idx = (ym_train != fr_label)
            if np.sum(nfr_train_idx) > 0:
                s2.fit(X_train[nfr_train_idx], ym_train[nfr_train_idx])
                preds_s2 = s2.predict(X_val)
            else:
                preds_s2 = np.array(['Unknown'] * len(X_val))
                
            final_preds = []
            for i in range(len(preds_s1)):
                if preds_s1[i] == fr_label:
                    final_preds.append(fr_label)
                else:
                    final_preds.append(preds_s2[i])
            accs.append(accuracy_score(ym_val, final_preds))
        return np.mean(accs)

    study_42c = optuna.create_study(direction="maximize")
    study_42c.optimize(obj_42c, n_trials=10)
    
    # Re-run best 4-2C model to get exact class report
    # ... For speed, we can just run the objective logic again with best params and collect final_preds
    # But let's just do it cleanly:
    bp = study_42c.best_params
    w_b = get_base_weights(y_binary)
    w_b[fr_label] *= bp['w_FR']
    w_b['NFR'] *= bp['w_NFR']
    best_s1 = LinearSVC(C=bp['C_1'], class_weight=w_b, random_state=42)
    
    y_nfr = y_multi[y_multi != fr_label]
    w_m = get_base_weights(y_nfr)
    for cls in np.unique(y_nfr):
        w_m[cls] *= bp[f'w_{cls}_s2']
    best_s2 = LinearSVC(C=bp['C_2'], class_weight=w_m, random_state=42)
    
    all_final_preds = np.empty_like(y_multi, dtype=object)
    for train_idx, val_idx in skf.split(X, y_multi):
        X_train, X_val = X[train_idx], X[val_idx]
        yb_train = y_binary[train_idx]
        ym_train = y_multi[train_idx]
        
        best_s1.fit(X_train, yb_train)
        preds_s1 = best_s1.predict(X_val)
        
        nfr_train_idx = (ym_train != fr_label)
        best_s2.fit(X_train[nfr_train_idx], ym_train[nfr_train_idx])
        preds_s2 = best_s2.predict(X_val)
        
        for i, val_i in enumerate(val_idx):
            if preds_s1[i] == fr_label:
                all_final_preds[val_i] = fr_label
            else:
                all_final_preds[val_i] = preds_s2[i]
                
    print("\n### Phase 4-2C (Simple 2-Stage) Classification Report ###")
    print(classification_report(y_multi, all_final_preds, zero_division=0))
    print(f"Overall Phase 4-2C Accuracy: {accuracy_score(y_multi, all_final_preds)*100:.2f}%")

if __name__ == "__main__":
    run_class_wise_analysis()
