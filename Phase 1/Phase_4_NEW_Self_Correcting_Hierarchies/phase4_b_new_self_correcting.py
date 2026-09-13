import os
import collections
import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

warnings.filterwarnings('ignore')

import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

# --- Custom CSL Implementations (Replicating Phase 2-B & 4-B) ---
def tempered_weights(y, alpha=0.5):
    counts = collections.Counter(y)
    max_count = max(counts.values())
    return {cls: (max_count / count)**alpha for cls, count in counts.items()}

class MetaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, cost_matrix=None, n_estimators=10):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(random_state=42)
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        bag = BaggingClassifier(estimator=self.base_estimator, n_estimators=self.n_estimators, random_state=42)
        bag.fit(X, y)
        probs = bag.predict_proba(X)
        classes = bag.classes_
        
        y_relabeled = np.zeros(len(y), dtype=y.dtype)
        for i in range(len(y)):
            expected_costs = [probs[i, j] * self.cost_matrix.get(c, 1.0) for j, c in enumerate(classes)]
            y_relabeled[i] = classes[np.argmax(expected_costs)]
            
        self.final_estimator_ = clone(self.base_estimator)
        self.final_estimator_.fit(X, y_relabeled)
        self.classes_ = self.final_estimator_.classes_
        return self

    def predict(self, X):
        return self.final_estimator_.predict(X)

class AdaCost(BaseEstimator, ClassifierMixin):
    def __init__(self, cost_matrix=None, n_estimators=50):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_estimators = n_estimators
        
    def fit(self, X, y):
        self.model_ = AdaBoostClassifier(n_estimators=self.n_estimators, random_state=42)
        sample_weight = np.array([self.cost_matrix.get(label, 1.0) for label in y])
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

class CSKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, cost_matrix=None, n_neighbors=5):
        self.cost_matrix = cost_matrix if cost_matrix is not None else {}
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.y_train_ = np.array(y)
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.knn_.fit(X, y)
        return self
        
    def predict(self, X):
        distances, indices = self.knn_.kneighbors(X)
        y_pred = []
        for i in range(len(X)):
            votes = {c: 0.0 for c in self.classes_}
            for j in range(self.n_neighbors):
                neighbor_class = self.y_train_[indices[i, j]]
                dist = distances[i, j]
                weight = 1.0 / (dist + 1e-5)
                votes[neighbor_class] += weight * self.cost_matrix.get(neighbor_class, 1.0)
            y_pred.append(max(votes, key=votes.get))
        return np.array(y_pred)

def get_native_csl_model(algo_name, cost_matrix):
    if algo_name == 'CS-SVM (Linear)': return LinearSVC(class_weight=cost_matrix, random_state=42)
    elif algo_name == 'CS-SVM (RBF)': return SVC(kernel='rbf', class_weight=cost_matrix, random_state=42)
    elif algo_name == 'CS-DT': return DecisionTreeClassifier(class_weight=cost_matrix, random_state=42)
    elif algo_name == 'CS-LR': return LogisticRegression(class_weight=cost_matrix, random_state=42, max_iter=1000)
    elif algo_name == 'CS-RF': return RandomForestClassifier(class_weight=cost_matrix, random_state=42)
    elif algo_name == 'MetaCost': return MetaCost(cost_matrix=cost_matrix)
    elif algo_name == 'AdaCost': return AdaCost(cost_matrix=cost_matrix)
    elif algo_name == 'CS-KNN': return CSKNN(cost_matrix=cost_matrix)

def run_4b_new(algo, embedding, dataset_name, X, y1, y2, fr_label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = []
    
    for train_idx, test_idx in skf.split(X, y2):
        X_train, X_test = X[train_idx], X[test_idx]
        y1_train, y2_train = y1[train_idx], y2[train_idx]
        y1_test, y2_test = y1[test_idx], y2[test_idx]
        
        # --- MECHANISM 1: STAGE 1 PARANOID 5x PENALTY (Directly in Cost Matrix) ---
        cost_matrix_s1 = tempered_weights(y1_train, alpha=0.5)
        if 'NFR' in cost_matrix_s1:
            cost_matrix_s1['NFR'] *= 5.0 
            
        model_stage1 = get_native_csl_model(algo, cost_matrix_s1)
        model_stage1.fit(X_train, y1_train)
        stage1_preds = model_stage1.predict(X_test)
        
        # --- MECHANISM 2: STAGE 2 CATCH-BIN (RESAMPLING FRs) ---
        nfr_mask = (y1_train == 'NFR')
        fr_mask = (y1_train == 'FR')
        
        if nfr_mask.sum() > 0:
            X_nfr, y2_nfr = X_train[nfr_mask], y2_train[nfr_mask]
            X_fr, y2_fr = X_train[fr_mask], y2_train[fr_mask]
            
            n_samples = min(nfr_mask.sum(), fr_mask.sum())
            if n_samples > 0:
                X_fr_samp, y2_fr_samp = resample(X_fr, y2_fr, n_samples=n_samples, random_state=42)
                X_s2 = np.vstack((X_nfr, X_fr_samp))
                y_s2 = np.concatenate((y2_nfr, y2_fr_samp))
            else:
                X_s2, y_s2 = X_nfr, y2_nfr
                
            # Compute new cost matrix for Stage 2
            cost_matrix_s2 = tempered_weights(y_s2, alpha=0.5)
            model_stage2 = get_native_csl_model(algo, cost_matrix_s2)
            model_stage2.fit(X_s2, y_s2)
        else:
            model_stage2 = None
            
        final_preds = []
        for i, p1 in enumerate(stage1_preds):
            if p1 == 'FR':
                final_preds.append(fr_label)
            else:
                if model_stage2 is not None:
                    final_preds.append(model_stage2.predict(X_test[i].reshape(1, -1))[0])
                else:
                    final_preds.append('Unknown_NFR')
                    
        acc.append(accuracy_score(y2_test, final_preds))
        
    print(f"[{dataset_name}] {algo} + {embedding} -> Accuracy: {np.mean(acc):.4f}")
    return {
        'Dataset': dataset_name,
        'Algorithm': algo,
        'Vectorization': embedding,
        'Phase 4-B-New': np.mean(acc)
    }

class Phase4_B_New_Generator:
    def __init__(self):
        # ALL 8 NATIVE CSL ALGORITHMS INCLUDED EXACTLY AS REQUESTED
        self.algorithms = ['CS-SVM (Linear)', 'CS-SVM (RBF)', 'CS-DT', 'CS-LR', 'CS-RF', 'MetaCost', 'AdaCost', 'CS-KNN'] 
        # ALL 6 EMBEDDINGS INCLUDED EXACTLY AS REQUESTED
        self.embeddings = ['TF-IDF', 'SBERT', 'BERT', 'MPNet', 'GloVe', 'Word2Vec']
        
        self.dataset_paths = {
            'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
            'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
        }

    def run(self):
        results = []
        max_workers = min(os.cpu_count() or 4, 6) 
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
                    futures = [executor.submit(run_4b_new, algo, embedding, dataset_name, X, y1, y2, fr_label) for algo in self.algorithms]
                    for f in futures:
                        results.append(f.result())
                        
        df_res = pd.DataFrame(results)
        df_res.to_csv('FINAL_Phase4_B_New_SelfCorrecting_Results.csv', index=False)
        print("Success! Phase 4-B-New generated.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    Phase4_B_New_Generator().run()
