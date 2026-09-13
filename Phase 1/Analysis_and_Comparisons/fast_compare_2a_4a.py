import pandas as pd
import numpy as np
import warnings
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

import sys
p2_path = r"D:\phd presentations\phase 1\PHASE 2\code_package"
if p2_path not in sys.path:
    sys.path.append(p2_path)
from models.deep_embeddings import get_deep_embeddings

def get_model(algo_name, w_dict):
    if algo_name == 'LinearSVC':
        return LinearSVC(class_weight=w_dict, random_state=42)
    elif algo_name == 'XGBoost':
        return XGBClassifier(random_state=42, eval_metric='logloss')

def main():
    algorithms = ['LinearSVC', 'XGBoost']
    embeddings = ['TF-IDF', 'MPNet']
    dataset_paths = {
        'PROMISE': r"D:\phd presentations\datasets\Promise.csv",
        'FNFC': r"D:\phd presentations\datasets\FNFC.csv"
    }

    results = []
    
    # Load 4-A-New results to match against
    df_new = pd.read_csv('FINAL_Phase4_A_New_SelfCorrecting_Results.csv')
    
    for dataset_name, path in dataset_paths.items():
        df = pd.read_csv(path, encoding='latin1')
        label_col = next((col for col in df.columns if col.strip().lower() in ['class', 'label', 'type', 'requirement_class']), df.columns[-1])
        text_col = 'Requirement' if 'Requirement' in df.columns else df.columns[0]
        df[label_col] = df[label_col].astype(str).str.strip()
        df['text'] = df[text_col].astype(str).fillna("")
        fr_labels = ['F', 'FR', 'Functional', 'functional', 'f']
        df['Stage1_Label'] = df[label_col].apply(lambda x: 'FR' if x in fr_labels else 'NFR')
        df['Stage2_Label'] = df[label_col]
        fr_label = df[df['Stage1_Label'] == 'FR']['Stage2_Label'].iloc[0] if len(df[df['Stage1_Label'] == 'FR']) > 0 else 'F'
        
        for embedding in embeddings:
            if embedding == 'TF-IDF':
                vec = TfidfVectorizer(max_features=5000)
                X = vec.fit_transform(df['text'].tolist()).toarray()
            else:
                X_emb, _, _ = get_deep_embeddings(df['text'].tolist(), df['text'].tolist(), embedding)
                X = X_emb.toarray() if hasattr(X_emb, "toarray") else X_emb.mean(axis=1) if len(X_emb.shape) == 3 else X_emb
            
            y1 = df['Stage1_Label'].values
            y2 = df['Stage2_Label'].values
            
            for algo in algorithms:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                acc_2a, acc_4a = [], []
                
                for train_idx, test_idx in skf.split(X, y2): 
                    X_train, X_test = X[train_idx], X[test_idx]
                    y1_train, y1_test = y1[train_idx], y1[test_idx]
                    y2_train, y2_test = y2[train_idx], y2[test_idx]
                    
                    # PHASE 2-A
                    classes_2a = np.unique(y2_train)
                    w_arr_2a = compute_class_weight('balanced', classes=classes_2a, y=y2_train)
                    w2a = dict(zip(classes_2a, w_arr_2a))
                    model_2a = get_model(algo, w2a)
                    if algo == 'XGBoost':
                        y2_map = {label: idx for idx, label in enumerate(classes_2a)}
                        y2_rev = {idx: label for label, idx in y2_map.items()}
                        model_2a.fit(X_train, np.array([y2_map[l] for l in y2_train]), sample_weight=np.array([w2a[label] for label in y2_train]))
                        preds_2a = [y2_rev[p] for p in model_2a.predict(X_test)]
                    else:
                        model_2a.fit(X_train, y2_train)
                        preds_2a = model_2a.predict(X_test)
                    acc_2a.append(accuracy_score(y2_test, preds_2a))
                    
                    # PHASE 4-A
                    classes_s1 = np.unique(y1_train)
                    w_arr_s1 = compute_class_weight('balanced', classes=classes_s1, y=y1_train)
                    w1 = dict(zip(classes_s1, w_arr_s1))
                    model_stage1 = get_model(algo, w1)
                    if algo == 'XGBoost':
                        model_stage1.fit(X_train, np.where(y1_train == 'NFR', 1, 0), sample_weight=np.array([w1[label] for label in y1_train]))
                        stage1_preds = np.where(model_stage1.predict(X_test) == 1, 'NFR', 'FR')
                    else:
                        model_stage1.fit(X_train, y1_train)
                        stage1_preds = model_stage1.predict(X_test)
                        
                    nfr_mask = (y1_train == 'NFR')
                    if nfr_mask.sum() > 0:
                        X_train_nfr, y2_train_nfr = X_train[nfr_mask], y2_train[nfr_mask]
                        classes_s2 = np.unique(y2_train_nfr)
                        w2 = dict(zip(classes_s2, compute_class_weight('balanced', classes=classes_s2, y=y2_train_nfr)))
                        model_stage2 = get_model(algo, w2)
                        if algo == 'XGBoost':
                            y2_map_s2 = {label: idx for idx, label in enumerate(classes_s2)}
                            y2_rev_s2 = {idx: label for label, idx in y2_map_s2.items()}
                            model_stage2.fit(X_train_nfr, np.array([y2_map_s2[l] for l in y2_train_nfr]), sample_weight=np.array([w2[label] for label in y2_train_nfr]))
                        else:
                            model_stage2.fit(X_train_nfr, y2_train_nfr)
                    else:
                        model_stage2 = None
                        
                    final_preds_4a = []
                    for i, p1 in enumerate(stage1_preds):
                        if p1 == 'FR':
                            final_preds_4a.append(fr_label)
                        else:
                            if model_stage2 is not None:
                                final_preds_4a.append(y2_rev_s2[model_stage2.predict(X_test[i].reshape(1, -1))[0]] if algo == 'XGBoost' else model_stage2.predict(X_test[i].reshape(1, -1))[0])
                            else:
                                final_preds_4a.append('Unknown_NFR')
                    acc_4a.append(accuracy_score(y2_test, final_preds_4a))
                
                # Get matching 4-A-New
                val_new = df_new[(df_new['Dataset'] == dataset_name) & (df_new['Algorithm'] == algo) & (df_new['Vectorization'] == embedding)]['Average'].values[0]
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': algo,
                    'Vectorization': embedding,
                    'Phase 2-A (Flat)': np.mean(acc_2a),
                    'Phase 4-A (2-Stage)': np.mean(acc_4a),
                    'Phase 4-A-New': val_new
                })

    df_res = pd.DataFrame(results)
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
