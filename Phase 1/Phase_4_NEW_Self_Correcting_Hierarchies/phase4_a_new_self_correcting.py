import os
import pandas as pd
import numpy as np
import warnings
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
from sklearn.utils import resample

warnings.filterwarnings('ignore')

import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

def get_model(algo_name, w_dict):
    if algo_name == 'LinearSVC':
        return LinearSVC(class_weight=w_dict, random_state=42)
    elif algo_name == 'RF':
        return RandomForestClassifier(class_weight=w_dict, random_state=42)
    elif algo_name == 'LR':
        return LogisticRegression(class_weight=w_dict, random_state=42, max_iter=1000)
    elif algo_name == 'NB':
        return GaussianNB() 
    elif algo_name == 'XGBoost':
        return XGBClassifier(random_state=42, eval_metric='logloss')
    elif algo_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5)

def run_single_algo(algo, embedding, dataset_name, X, y1, y2, multiplier, fr_label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    
    for train_idx, test_idx in skf.split(X, y2): 
        X_train, X_test = X[train_idx], X[test_idx]
        y1_train, y1_test = y1[train_idx], y1[test_idx]
        y2_train, y2_test = y2[train_idx], y2[test_idx]
        
        # --- STAGE 1: High-Penalty FR vs NFR ---
        classes_s1 = np.unique(y1_train)
        w_arr_s1 = compute_class_weight('balanced', classes=classes_s1, y=y1_train)
        w1 = dict(zip(classes_s1, w_arr_s1))
        
        if 'NFR' in w1:
            w1['NFR'] *= multiplier
            
        model_stage1 = get_model(algo, w1)
        
        if algo == 'XGBoost':
            sample_weight = np.array([w1[label] for label in y1_train])
            y1_train_num = np.where(y1_train == 'NFR', 1, 0)
            model_stage1.fit(X_train, y1_train_num, sample_weight=sample_weight)
            preds_num = model_stage1.predict(X_test)
            stage1_preds = np.where(preds_num == 1, 'NFR', 'FR')
        elif algo in ['NB', 'KNN']:
            model_stage1.fit(X_train, y1_train)
            stage1_preds = model_stage1.predict(X_test)
        else:
            model_stage1.fit(X_train, y1_train)
            stage1_preds = model_stage1.predict(X_test)
        
        # --- STAGE 2: SELF-CORRECTING LAYER ---
        nfr_mask = (y1_train == 'NFR')
        fr_mask = (y1_train == 'FR')
        
        if nfr_mask.sum() > 0:
            # 1. Get all True NFRs
            X_train_nfr = X_train[nfr_mask]
            y2_train_nfr = y2_train[nfr_mask]
            
            # 2. Get an equal sample of FRs to act as the "Catch Bin"
            X_train_fr = X_train[fr_mask]
            y2_train_fr = y2_train[fr_mask]
            
            n_samples = min(nfr_mask.sum(), fr_mask.sum())
            if n_samples > 0:
                X_train_fr_sampled, y2_train_fr_sampled = resample(X_train_fr, y2_train_fr, n_samples=n_samples, random_state=42)
                X_stage2 = np.vstack((X_train_nfr, X_train_fr_sampled))
                y_stage2 = np.concatenate((y2_train_nfr, y2_train_fr_sampled))
            else:
                X_stage2 = X_train_nfr
                y_stage2 = y2_train_nfr
            
            # 3. Train Stage 2 on this perfectly balanced self-correcting dataset
            classes_s2 = np.unique(y_stage2)
            w_arr_s2 = compute_class_weight('balanced', classes=classes_s2, y=y_stage2)
            w2 = dict(zip(classes_s2, w_arr_s2))
            
            model_stage2 = get_model(algo, w2)
            
            if algo == 'XGBoost':
                y2_map = {label: idx for idx, label in enumerate(classes_s2)}
                y2_rev = {idx: label for label, idx in y2_map.items()}
                y2_train_num = np.array([y2_map[l] for l in y_stage2])
                sw2 = np.array([w2[label] for label in y_stage2])
                model_stage2.fit(X_stage2, y2_train_num, sample_weight=sw2)
            elif algo in ['NB', 'KNN']:
                model_stage2.fit(X_stage2, y_stage2)
            else:
                model_stage2.fit(X_stage2, y_stage2)
        else:
            model_stage2 = None
        
        # --- HIERARCHICAL PREDICTION ---
        final_preds = []
        for i, p1 in enumerate(stage1_preds):
            if p1 == 'FR':
                final_preds.append(fr_label) 
            else:
                if model_stage2 is not None:
                    if algo == 'XGBoost':
                        p2_num = model_stage2.predict(X_test[i].reshape(1, -1))[0]
                        final_preds.append(y2_rev[p2_num])
                    else:
                        final_preds.append(model_stage2.predict(X_test[i].reshape(1, -1))[0])
                else:
                    final_preds.append('Unknown_NFR')
                    
        # Any 'FR' predictions out of Stage 2 need to be mapped properly if not already done
        mapped_preds = [fr_label if p == 'FR' else p for p in final_preds]
        fold_accs.append(accuracy_score(y2_test, mapped_preds))
        
    avg_acc = np.mean(fold_accs)
    print(f"[{dataset_name}] {algo} + {embedding} -> Accuracy: {avg_acc:.4f}")
    return {'Phase': 'Phase 4-A-New (Self-Correcting)', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': avg_acc}


class Phase4_A_New_SelfCorrecting:
    def __init__(self):
        self.algorithms = ['LinearSVC', 'RF', 'LR', 'NB', 'XGBoost', 'KNN']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
        }
        self.multiplier = 5.0

    def load_data(self, path):
        df = pd.read_csv(path, encoding='latin1')
        label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
        text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
            
        df[label_col] = df[label_col].astype(str).str.strip()
        df['text'] = df[text_col].astype(str).fillna("")
        
        fr_labels = ['F', 'FR', 'Functional', 'functional', 'f']
        df['Stage1_Label'] = df[label_col].apply(lambda x: 'FR' if x in fr_labels else 'NFR')
        df['Stage2_Label'] = df[label_col]
        return df

    def get_real_embeddings(self, texts, embed_type):
        if embed_type == 'TF-IDF':
            vec = TfidfVectorizer(max_features=5000)
            return vec.fit_transform(texts).toarray()
        else:
            X_emb, _, _ = get_deep_embeddings(texts, texts, embed_type)
            if hasattr(X_emb, "toarray"):
                X_emb = X_emb.toarray()
            elif len(X_emb.shape) == 3:
                X_emb = X_emb.mean(axis=1) 
            return X_emb

    def run(self):
        print("=" * 60)
        print("PHASE 4-A-NEW (SELF-CORRECTING HIERARCHY)")
        print("=" * 60)
        
        results = []
        max_workers = min(os.cpu_count() or 4, 6) 
        
        for dataset_name, path in self.dataset_paths.items():
            print(f"\n--- Loading Dataset: {dataset_name} ---")
            df = self.load_data(path)
            fr_label = df[df['Stage1_Label'] == 'FR']['Stage2_Label'].iloc[0] if len(df[df['Stage1_Label'] == 'FR']) > 0 else 'F'
            
            for embedding in self.embeddings:
                print(f"  -> Generating {embedding} Vectors...")
                X = self.get_real_embeddings(df['text'].tolist(), embedding)
                y1 = df['Stage1_Label'].values
                y2 = df['Stage2_Label'].values
                
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(run_single_algo, algo, embedding, dataset_name, X, y1, y2, self.multiplier, fr_label) for algo in self.algorithms]
                    for f in futures:
                        results.append(f.result())
                        
        res_df = pd.DataFrame(results)
        res_df.to_csv('FINAL_Phase4_A_New_SelfCorrecting_Results.csv', index=False)
        print("\n[SUCCESS] Phase 4-A-New complete. Results saved to CSV.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    Phase4_A_New_SelfCorrecting().run()
