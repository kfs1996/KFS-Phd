import os
import pandas as pd
import numpy as np
import warnings
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

# Helper to load genuine embeddings from Phase 2
import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

class Phase4_A_New_Cascading:
    def __init__(self):
        self.algorithms = ['LinearSVC', 'RF', 'LR', 'NB', 'XGBoost', 'KNN']
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
        }
        
        # This is the "Paranoid Guard" multiplier. 
        # We multiply the mathematical penalty of missing an NFR by 5x!
        self.stage1_nfr_penalty_multiplier = 5.0 

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

    def get_model(self, algo_name, w_dict):
        if algo_name == 'LinearSVC':
            return LinearSVC(class_weight=w_dict, random_state=42)
        elif algo_name == 'RF':
            return RandomForestClassifier(class_weight=w_dict, random_state=42)
        elif algo_name == 'LR':
            return LogisticRegression(class_weight=w_dict, random_state=42, max_iter=1000)
        elif algo_name == 'NB':
            return GaussianNB() # NB doesn't support class_weight natively in sklearn
        elif algo_name == 'XGBoost':
            return XGBClassifier(random_state=42, eval_metric='logloss')
        elif algo_name == 'KNN':
            return KNeighborsClassifier(n_neighbors=5)

    def run(self):
        print("=" * 60)
        print("PHASE 4-A-NEW: HIGH-PENALTY CASCADING HIERARCHY")
        print(f"Stage 1 NFR Penalty Multiplier: {self.stage1_nfr_penalty_multiplier}x")
        print("=" * 60)
        
        results = []
        for dataset_name, path in self.dataset_paths.items():
            print(f"\n--- Processing Dataset: {dataset_name} ---")
            df = self.load_data(path)
            
            for embedding in self.embeddings:
                print(f"  -> Generating {embedding} Vectors...")
                X = self.get_real_embeddings(df['text'].tolist(), embedding)
                y1 = df['Stage1_Label'].values
                y2 = df['Stage2_Label'].values
                
                for algo in self.algorithms:
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
                        
                        # Apply Cascading Penalty to NFR class!
                        if 'NFR' in w1:
                            w1['NFR'] *= self.stage1_nfr_penalty_multiplier
                            
                        # Format weight dict for XGBoost if needed (not natively dict supported, needs sample_weight)
                        # We'll just pass dict to sklearn algorithms.
                        model_stage1 = self.get_model(algo, w1)
                        
                        if algo == 'XGBoost':
                            # XGBoost needs label encoding and sample_weights
                            sample_weight = np.array([w1[label] for label in y1_train])
                            y1_train_num = np.where(y1_train == 'NFR', 1, 0)
                            model_stage1.fit(X_train, y1_train_num, sample_weight=sample_weight)
                            preds_num = model_stage1.predict(X_test)
                            stage1_preds = np.where(preds_num == 1, 'NFR', 'FR')
                        elif algo in ['NB', 'KNN']:
                            # NB and KNN don't take class_weight
                            model_stage1.fit(X_train, y1_train)
                            stage1_preds = model_stage1.predict(X_test)
                        else:
                            model_stage1.fit(X_train, y1_train)
                            stage1_preds = model_stage1.predict(X_test)
                        
                        # --- STAGE 2: Multi-class NFRs ---
                        nfr_mask = (y1_train == 'NFR')
                        if nfr_mask.sum() > 0:
                            X_train_nfr = X_train[nfr_mask]
                            y2_train_nfr = y2_train[nfr_mask]
                            
                            classes_s2 = np.unique(y2_train_nfr)
                            w_arr_s2 = compute_class_weight('balanced', classes=classes_s2, y=y2_train_nfr)
                            w2 = dict(zip(classes_s2, w_arr_s2))
                            
                            model_stage2 = self.get_model(algo, w2)
                            
                            if algo == 'XGBoost':
                                y2_map = {label: idx for idx, label in enumerate(classes_s2)}
                                y2_rev = {idx: label for label, idx in y2_map.items()}
                                y2_train_num = np.array([y2_map[l] for l in y2_train_nfr])
                                sw2 = np.array([w2[label] for label in y2_train_nfr])
                                model_stage2.fit(X_train_nfr, y2_train_num, sample_weight=sw2)
                            elif algo in ['NB', 'KNN']:
                                model_stage2.fit(X_train_nfr, y2_train_nfr)
                            else:
                                model_stage2.fit(X_train_nfr, y2_train_nfr)
                        else:
                            model_stage2 = None
                        
                        # --- HIERARCHICAL PREDICTION ---
                        final_preds = []
                        for i, p1 in enumerate(stage1_preds):
                            if p1 == 'FR':
                                final_preds.append('FR') 
                            else:
                                if model_stage2 is not None:
                                    if algo == 'XGBoost':
                                        p2_num = model_stage2.predict(X_test[i].reshape(1, -1))[0]
                                        final_preds.append(y2_rev[p2_num])
                                    else:
                                        final_preds.append(model_stage2.predict(X_test[i].reshape(1, -1))[0])
                                else:
                                    final_preds.append('Unknown_NFR')
                                    
                        fr_label = df[df['Stage1_Label'] == 'FR']['Stage2_Label'].iloc[0] if len(df[df['Stage1_Label'] == 'FR']) > 0 else 'F'
                        mapped_preds = [fr_label if p == 'FR' else p for p in final_preds]
                        
                        fold_accs.append(accuracy_score(y2_test, mapped_preds))
                        
                    avg_acc = np.mean(fold_accs)
                    results.append({'Phase': 'Phase 4-A-New', 'Dataset': dataset_name, 'Algorithm': algo, 'Vectorization': embedding, 'Average': avg_acc})
        
        res_df = pd.DataFrame(results)
        res_df.to_csv('FINAL_Phase4_A_New_Results.csv', index=False)
        print("\n[SUCCESS] Phase 4-A-New complete. Results saved to CSV.")

if __name__ == "__main__":
    Phase4_A_New_Cascading().run()
