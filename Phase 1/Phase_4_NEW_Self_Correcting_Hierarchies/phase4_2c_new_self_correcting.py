import os
import pandas as pd
import numpy as np
import warnings
import optuna
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

def suggest_model(trial, algo_name, y_train, is_stage1=False):
    classes = np.unique(y_train)
    base_w = compute_class_weight('balanced', classes=classes, y=y_train)
    w_dict = dict(zip(classes, base_w))
    
    if algo_name != 'NB':
        for cls in classes:
            # Optuna tuning
            w_dict[cls] *= trial.suggest_float(f'w_{cls}', 0.5, 2.0)
            
        # THE ENGINEERING: Phase 4-2C-NEW PARANOID PENALTY
        if is_stage1 and 'NFR' in w_dict:
            w_dict['NFR'] *= 5.0 # Force extreme paranoia to push False Positives

    if algo_name == 'LinearSVC':
        c_val = trial.suggest_float('C', 0.1, 10.0, log=True)
        return LinearSVC(C=c_val, class_weight=w_dict, random_state=42)
    elif algo_name == 'RF':
        n_est = trial.suggest_int('n_est', 50, 150)
        max_d = trial.suggest_int('max_depth', 5, 20)
        return RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight=w_dict, random_state=42)
    elif algo_name == 'LR':
        c_val = trial.suggest_float('C', 0.1, 10.0, log=True)
        return LogisticRegression(C=c_val, class_weight=w_dict, random_state=42, max_iter=500)
    elif algo_name == 'NB':
        var_smooth = trial.suggest_float('var_smooth', 1e-10, 1e-8, log=True)
        return GaussianNB(var_smoothing=var_smooth)
    elif algo_name == 'XGBoost':
        n_est = trial.suggest_int('n_est', 50, 150)
        max_d = trial.suggest_int('max_depth', 3, 10)
        return XGBClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, eval_metric='logloss'), w_dict
    elif algo_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 9)
        return KNeighborsClassifier(n_neighbors=n_neighbors)

def train_and_predict(model_res, algo, X_train, y_train, X_test):
    if algo == 'XGBoost':
        model, w_dict = model_res
        y_map = {label: idx for idx, label in enumerate(np.unique(y_train))}
        y_rev = {idx: label for label, idx in y_map.items()}
        y_train_mapped = np.array([y_map[l] for l in y_train])
        sample_weight = np.array([w_dict[l] for l in y_train])
        model.fit(X_train, y_train_mapped, sample_weight=sample_weight)
        preds_mapped = model.predict(X_test)
        return np.array([y_rev[p] for p in preds_mapped])
    else:
        model = model_res
        model.fit(X_train, y_train)
        return model.predict(X_test)

def objective_42c_new(trial, algo, X, y_binary, y_multi):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    
    for train_idx, val_idx in skf.split(X, y_multi):
        X_train, X_val = X[train_idx], X[val_idx]
        yb_train, yb_val = y_binary[train_idx], y_binary[val_idx]
        ym_train, ym_val = y_multi[train_idx], y_multi[val_idx]
        
        # STAGE 1: Paranoid Binary Classification
        s1_model = suggest_model(trial, algo, yb_train, is_stage1=True)
        preds_s1 = train_and_predict(s1_model, algo, X_train, yb_train, X_val)
        
        # STAGE 2: Multi-Class with CATCH-BIN
        s2_model = suggest_model(trial, algo, ym_train, is_stage1=False)
        preds_s2_all = train_and_predict(s2_model, algo, X_train, ym_train, X_val)
        
        # Find the actual label used for Functional in this dataset
        fr_label = next((l for l in np.unique(ym_train) if str(l).strip().upper() in ['F', 'FR', 'FUNCTIONAL']), 'FR')
        
        # HIERARCHY LOGIC
        final_preds = []
        for i in range(len(preds_s1)):
            if preds_s1[i] == 'FR':
                final_preds.append(fr_label)
            else:
                final_preds.append(preds_s2_all[i])
                
        accs.append(accuracy_score(ym_val, final_preds))
        
    return np.mean(accs)

def run_phase42c_new(algo, embedding, dataset_name, X, y_multi):
    y_binary = np.array(['FR' if str(label).strip().upper() in ['F', 'FR', 'FUNCTIONAL'] else 'NFR' for label in y_multi])
    
    study = optuna.create_study(direction="maximize")
    # Reduced to 10 trials to prevent 10-hour lockups, while maintaining Optuna AI integrity
    study.optimize(lambda trial: objective_42c_new(trial, algo, X, y_binary, y_multi), n_trials=10)
    
    best_acc = study.best_value
    print(f"[{dataset_name}] {algo} + {embedding} -> Phase 4-2C-NEW Acc: {best_acc:.4f}")
    return {
        'Dataset': dataset_name,
        'Algorithm': algo,
        'Vectorization': embedding,
        'Phase 4-2C-NEW (Adaptive Self-Correcting)': best_acc
    }

class Phase42C_NEW_Generator:
    def __init__(self):
        self.algorithms = ['LinearSVC', 'RF', 'LR', 'NB', 'XGBoost', 'KNN']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
        }

    def run(self):
        results = []
        max_workers = os.cpu_count()
        print(f"--- RUNNING PHASE 4-2C-NEW (SELF-CORRECTING OPTUNA) AT MAX CAPACITY ({max_workers} CORES) ---")
        
        for dataset_name, path in self.dataset_paths.items():
            df = pd.read_csv(path, encoding='latin1')
            label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
            text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            df[label_col] = df[label_col].astype(str).str.strip()
            df['text'] = df[text_col].astype(str).fillna("")
            y_multi = df[label_col].values
            
            for embedding in self.embeddings:
                if embedding == 'TF-IDF':
                    vec = TfidfVectorizer(max_features=5000)
                    X = vec.fit_transform(df['text'].tolist()).toarray()
                else:
                    X_emb, _, _ = get_deep_embeddings(df['text'].tolist(), df['text'].tolist(), embedding)
                    X = X_emb.toarray() if hasattr(X_emb, "toarray") else X_emb.mean(axis=1) if len(X_emb.shape) == 3 else X_emb
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_phase42c_new, algo, embedding, dataset_name, X, y_multi) for algo in self.algorithms]
                    for f in futures:
                        results.append(f.result())
                        
        df_res = pd.DataFrame(results)
        df_res.to_csv('FINAL_Phase4_2C_NEW_SelfCorrecting_Results.csv', index=False)
        print("Success! Phase 4-2C-NEW Generated.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    Phase42C_NEW_Generator().run()
