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

def suggest_model(trial, algo_name, stage_prefix, y_train):
    classes = np.unique(y_train)
    base_w = compute_class_weight('balanced', classes=classes, y=y_train)
    w_dict = dict(zip(classes, base_w))
    
    # Optuna tuning the class weight dynamically (The "Adaptive" part of 2-C)
    if algo_name != 'NB': 
        # Randomly boost minority class weight
        minority_class = min(w_dict, key=lambda k: list(y_train).count(k))
        w_multiplier = trial.suggest_float(f'{stage_prefix}_w_mult', 1.0, 5.0)
        w_dict[minority_class] *= w_multiplier

    if algo_name == 'LinearSVC':
        c_val = trial.suggest_float(f'{stage_prefix}_C', 0.1, 10.0, log=True)
        return LinearSVC(C=c_val, class_weight=w_dict, random_state=42)
    elif algo_name == 'RF':
        n_est = trial.suggest_int(f'{stage_prefix}_n_est', 50, 150)
        max_d = trial.suggest_int(f'{stage_prefix}_max_depth', 5, 20)
        return RandomForestClassifier(n_estimators=n_est, max_depth=max_d, class_weight=w_dict, random_state=42)
    elif algo_name == 'LR':
        c_val = trial.suggest_float(f'{stage_prefix}_C', 0.1, 10.0, log=True)
        return LogisticRegression(C=c_val, class_weight=w_dict, random_state=42, max_iter=500)
    elif algo_name == 'NB':
        var_smooth = trial.suggest_float(f'{stage_prefix}_var_smooth', 1e-10, 1e-8, log=True)
        return GaussianNB(var_smoothing=var_smooth)
    elif algo_name == 'XGBoost':
        n_est = trial.suggest_int(f'{stage_prefix}_n_est', 50, 150)
        max_d = trial.suggest_int(f'{stage_prefix}_max_depth', 3, 10)
        return XGBClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, eval_metric='logloss'), w_dict
    elif algo_name == 'KNN':
        n_neighbors = trial.suggest_int(f'{stage_prefix}_n_neighbors', 3, 9)
        return KNeighborsClassifier(n_neighbors=n_neighbors)

def objective(trial, algo, X, y1, y2, fr_label):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    
    for train_idx, val_idx in skf.split(X, y2):
        X_train, X_val = X[train_idx], X[val_idx]
        y1_train, y1_val = y1[train_idx], y1[val_idx]
        y2_train, y2_val = y2[train_idx], y2[val_idx]
        
        # --- STAGE 1 ---
        model_s1_res = suggest_model(trial, algo, 's1', y1_train)
        if algo == 'XGBoost':
            model_s1, w_dict_s1 = model_s1_res
            sample_weight = np.array([w_dict_s1[label] for label in y1_train])
            model_s1.fit(X_train, np.where(y1_train == 'NFR', 1, 0), sample_weight=sample_weight)
            preds_s1 = np.where(model_s1.predict(X_val) == 1, 'NFR', 'FR')
        else:
            model_s1 = model_s1_res
            model_s1.fit(X_train, y1_train)
            preds_s1 = model_s1.predict(X_val)
            
        # --- STAGE 2 (SIMPLE, NO CATCH BIN) ---
        nfr_mask = (y1_train == 'NFR')
        if nfr_mask.sum() > 0:
            X_train_s2 = X_train[nfr_mask]
            y2_train_s2 = y2_train[nfr_mask]
            
            model_s2_res = suggest_model(trial, algo, 's2', y2_train_s2)
            if algo == 'XGBoost':
                model_s2, w_dict_s2 = model_s2_res
                y2_map = {label: idx for idx, label in enumerate(np.unique(y2_train_s2))}
                y2_rev = {idx: label for label, idx in y2_map.items()}
                sw2 = np.array([w_dict_s2[label] for label in y2_train_s2])
                model_s2.fit(X_train_s2, np.array([y2_map[l] for l in y2_train_s2]), sample_weight=sw2)
            else:
                model_s2 = model_s2_res
                model_s2.fit(X_train_s2, y2_train_s2)
        else:
            model_s2 = None
            
        # --- COMBINE PREDICTIONS ---
        final_preds = []
        for i, p1 in enumerate(preds_s1):
            if p1 == 'FR':
                final_preds.append(fr_label)
            else:
                if model_s2 is not None:
                    if algo == 'XGBoost':
                        pred = model_s2.predict(X_val[i].reshape(1, -1))[0]
                        final_preds.append(y2_rev[pred])
                    else:
                        final_preds.append(model_s2.predict(X_val[i].reshape(1, -1))[0])
                else:
                    final_preds.append('Unknown_NFR')
                    
        accs.append(accuracy_score(y2_val, final_preds))
        
    return np.mean(accs)

def run_optuna_combo(algo, embedding, dataset_name, X, y1, y2, fr_label):
    study = optuna.create_study(direction="maximize")
    # Reduced trials to 15 to ensure it finishes in a reasonable time while still tuning
    study.optimize(lambda trial: objective(trial, algo, X, y1, y2, fr_label), n_trials=15)
    best_acc = study.best_value
    print(f"[{dataset_name}] {algo} + {embedding} -> Best Adaptive Acc: {best_acc:.4f}")
    return {
        'Dataset': dataset_name,
        'Algorithm': algo,
        'Vectorization': embedding,
        'Phase 4-2C (Adaptive 2-Stage)': best_acc
    }

class Phase4_2C_Generator:
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
        print(f"--- RUNNING AT ABSOLUTE MAX CAPACITY ({max_workers} CORES) ---")
        
        for dataset_name, path in self.dataset_paths.items():
            df = pd.read_csv(path, encoding='latin1')
            label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
            text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            df[label_col] = df[label_col].astype(str).str.strip()
            df['text'] = df[text_col].astype(str).fillna("")
            fr_labels = ['F', 'FR', 'Functional', 'functional', 'f']
            df['Stage1_Label'] = df[label_col].apply(lambda x: 'FR' if x in fr_labels else 'NFR')
            df['Stage2_Label'] = df[label_col]
            fr_label = df[df['Stage1_Label'] == 'FR']['Stage2_Label'].iloc[0] if len(df[df['Stage1_Label'] == 'FR']) > 0 else 'F'
            
            for embedding in self.embeddings:
                if embedding == 'TF-IDF':
                    vec = TfidfVectorizer(max_features=5000)
                    X = vec.fit_transform(df['text'].tolist()).toarray()
                else:
                    X_emb, _, _ = get_deep_embeddings(df['text'].tolist(), df['text'].tolist(), embedding)
                    X = X_emb.toarray() if hasattr(X_emb, "toarray") else X_emb.mean(axis=1) if len(X_emb.shape) == 3 else X_emb
                
                y1 = df['Stage1_Label'].values
                y2 = df['Stage2_Label'].values
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_optuna_combo, algo, embedding, dataset_name, X, y1, y2, fr_label) for algo in self.algorithms]
                    for f in futures:
                        results.append(f.result())
                        
        df_res = pd.DataFrame(results)
        df_res.to_csv('FINAL_Phase4_2C_Simple_Optuna_Results.csv', index=False)
        print("Success! Phase 4-2C Adaptive 2-Stage generated.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    Phase4_2C_Generator().run()
