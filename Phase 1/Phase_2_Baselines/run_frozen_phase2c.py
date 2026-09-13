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
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

def suggest_1stage_model(trial, algo_name, y_train):
    classes = np.unique(y_train)
    base_w = compute_class_weight('balanced', classes=classes, y=y_train)
    w_dict = dict(zip(classes, base_w))
    
    if algo_name != 'NB': 
        for cls in classes:
            w_dict[cls] *= trial.suggest_float(f'w_{cls}', 0.5, 2.0)

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

def objective_1stage(trial, algo, X, y):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model_res = suggest_1stage_model(trial, algo, y_train)
        
        if algo == 'XGBoost':
            model, w_dict = model_res
            y_map = {label: idx for idx, label in enumerate(np.unique(y_train))}
            y_rev = {idx: label for label, idx in y_map.items()}
            y_train_mapped = np.array([y_map[l] for l in y_train])
            sample_weight = np.array([w_dict[l] for l in y_train])
            model.fit(X_train, y_train_mapped, sample_weight=sample_weight)
            preds_mapped = model.predict(X_val)
            preds = [y_rev[p] for p in preds_mapped]
        else:
            model = model_res
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
        accs.append(accuracy_score(y_val, preds))
        
    return np.mean(accs)

def run_phase2c(algo, embedding, dataset_name, X, y):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective_1stage(trial, algo, X, y), n_trials=15)
    best_acc = study.best_value
    print(f"[{dataset_name}] {algo} + {embedding} -> Phase 2-C Acc: {best_acc:.4f}")
    return {
        'Dataset': dataset_name,
        'Algorithm': algo,
        'Vectorization': embedding,
        'Phase 2-C': best_acc
    }

class FrozenData_Generator:
    def __init__(self):
        self.algorithms = ['LinearSVC', 'RF', 'LR', 'NB', 'XGBoost', 'KNN']
        # ONLY RUN THE MISSING EMBEDDINGS
        self.embeddings = ['GloVe', 'Word2Vec'] 
        self.dataset_paths = {
            'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
        }

    def run(self):
        results = []
        max_workers = os.cpu_count()
        print(f"--- RUNNING MISSING DATA AT MAX CAPACITY ({max_workers} CORES) ---")
        
        for dataset_name, path in self.dataset_paths.items():
            df = pd.read_csv(path, encoding='latin1')
            label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
            text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            df[label_col] = df[label_col].astype(str).str.strip()
            df['text'] = df[text_col].astype(str).fillna("")
            y = df[label_col].values
            
            for embedding in self.embeddings:
                X_emb, _, _ = get_deep_embeddings(df['text'].tolist(), df['text'].tolist(), embedding)
                X = X_emb.toarray() if hasattr(X_emb, "toarray") else X_emb.mean(axis=1) if len(X_emb.shape) == 3 else X_emb
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_phase2c, algo, embedding, dataset_name, X, y) for algo in self.algorithms]
                    for f in futures:
                        results.append(f.result())
                        
        df_res = pd.DataFrame(results)
        df_res.to_csv('FINAL_Phase2_C_MissingData.csv', index=False)
        print("Success! Missing Phase 2-C Data Generated.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    FrozenData_Generator().run()
