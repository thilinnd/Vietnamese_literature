# -*- coding: utf-8 -*-
"""
Script huấn luyện NER Toàn diện:
1. So sánh đa mô hình (CRF, LR, RF, Bi-LSTM, PhoBERT).
2. So sánh dữ liệu (Original vs Augmented).
3. Learning Curve (CRF, LR, RF).
Hỗ trợ đo RAM, Thời gian và Lưu Model.
"""

import os
import json
import time
import tracemalloc  # Khôi phục đo RAM
import warnings
import joblib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

# Deep Learning Imports
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
    
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from torch.optim import AdamW
    from tqdm import tqdm
except ImportError:
    print("Warning: Thiếu thư viện Deep Learning.")

warnings.filterwarnings("ignore")

# --- CẤU HÌNH ---
SEED = 42
MAX_LEN = 128
SAVE_DIR = 'saved_models'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

TAG2IDX = {}
IDX2TAG = {}

# =============================================================================
# 1. DATA & FEATURE ENGINEERING
# =============================================================================

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_tag_mappings(all_sents):
    tags = list(set([t[1] for sent in all_sents for t in sent]))
    if "O" not in tags: tags.append("O")
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(), 
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(), 
        'word.has_underscore': '_' in word,
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle()})
    else: features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle()})
    else: features['EOS'] = True
    return features

def prepare_classical_data(sents):
    X = [[word2features(s, i) for i in range(len(s))] for s in sents]
    y = [[label for token, label in s] for s in sents]
    return X, y

def prepare_dl_data(sents, word2idx, tag2idx, max_len):
    X = [[word2idx.get(w[0], word2idx.get("UNK", 0)) for w in s] for s in sents]
    X = pad_sequences(X, maxlen=max_len, padding="post", value=word2idx.get("PAD", 0))
    y = [[tag2idx[w[1]] for w in s] for s in sents]
    y = pad_sequences(y, maxlen=max_len, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]
    return np.array(X), np.array(y)

# --- KHÔI PHỤC: Hàm đo hiệu năng đầy đủ (RAM + Time) ---
def measure_performance(model_name, train_func, *args, **kwargs):
    print(f"\n--- ⏱️ Bắt đầu đo lường: {model_name} ---")
    
    # 1. Theo dõi RAM
    tracemalloc.start()
    
    # 2. Bấm giờ
    start_time = time.time()
    
    try:
        model, f1_score = train_func(*args, **kwargs)
    except Exception as e:
        print(f" Lỗi khi chạy {model_name}: {e}")
        tracemalloc.stop()
        return None
    
    end_time = time.time()
    
    # 3. Lấy thông số RAM
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024)
    
    print(f" Hoàn tất {model_name}:")
    print(f"   - F1-Score: {f1_score:.4f}")
    print(f"   - Thời gian: {execution_time:.2f}s")
    print(f"   - RAM đỉnh (Peak): {peak_memory_mb:.2f} MB")
    
    return {
        'Model': model_name,
        'F1-Score': f1_score,
        'Time (s)': execution_time,
        'Memory (MB)': peak_memory_mb
    }

# =============================================================================
# 2. MODEL TRAINING FUNCTIONS
# =============================================================================

def train_crf(X_train, y_train, X_test, y_test, save_path=None):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    
    if save_path:
        joblib.dump(crf, save_path)
    
    report = flat_classification_report(y_test, y_pred, digits=4, output_dict=True)
    return crf, report['weighted avg']['f1-score']

def train_sklearn(model_type, X_train, y_train, X_test, y_test):
    X_train_flat = [item for sublist in X_train for item in sublist]
    y_train_flat = [item for sublist in y_train for item in sublist]
    X_test_flat = [item for sublist in X_test for item in sublist]
    y_test_flat = [item for sublist in y_test for item in sublist]
    
    v = DictVectorizer(sparse=False)
    X_train_vec = v.fit_transform(X_train_flat)
    X_test_vec = v.transform(X_test_flat)
    
    if model_type == 'LR':
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
    elif model_type == 'RF':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED)
    
    clf.fit(X_train_vec, y_train_flat)
    y_pred = clf.predict(X_test_vec)
    
    # Save (optional)
    if model_type == 'LR': joblib.dump(clf, os.path.join(SAVE_DIR, 'lr_model.pkl'))
    
    report = classification_report(y_test_flat, y_pred, digits=4, output_dict=True)
    return clf, report['weighted avg']['f1-score']

def train_bilstm(train_sents, test_sents, word2idx, tag2idx):
    global IDX2TAG
    X_train, y_train = prepare_dl_data(train_sents, word2idx, tag2idx, MAX_LEN)
    X_test, y_test = prepare_dl_data(test_sents, word2idx, tag2idx, MAX_LEN)
    
    model = Sequential([
        Embedding(input_dim=len(word2idx), output_dim=50),
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.3),
        TimeDistributed(Dense(len(tag2idx), activation="softmax"))
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    
    model.save(os.path.join(SAVE_DIR, 'bilstm_model.h5'))
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    flat_pred, flat_true = [], []
    for i in range(len(test_sents)):
        length = len(test_sents[i])
        flat_pred.extend([IDX2TAG[idx] for idx in y_pred[i][:length]])
        flat_true.extend([IDX2TAG[idx] for idx in y_true[i][:length]])
        
    report = classification_report(flat_true, flat_pred, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# --- PHOBERT ---
class PhoBertDataset(Dataset):
    def __init__(self, sents, tokenizer, tag2idx, max_len=128):
        self.sents = sents
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self): return len(self.sents)

    def __getitem__(self, idx):
        sent = self.sents[idx]
        words = [t[0] for t in sent]
        labels = [self.tag2idx[t[1]] for t in sent]
        
        tokenized = self.tokenizer(words, is_split_into_words=True, padding='max_length', 
                                   truncation=True, max_length=self.max_len, return_tensors="pt")
        word_ids = tokenized.word_ids(batch_index=0)
        label_ids = []
        prev_idx = None
        for w_idx in word_ids:
            if w_idx is None or w_idx == prev_idx:
                label_ids.append(-100)
            else:
                label_ids.append(labels[w_idx])
            prev_idx = w_idx
        return {k: v.squeeze(0) for k, v in tokenized.items()}, torch.tensor(label_ids, dtype=torch.long)

def train_phobert(train_sents, test_sents, tag2idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PhoBERT device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base-v2", num_labels=len(tag2idx))
    model.to(device)
    
    train_loader = DataLoader(PhoBertDataset(train_sents, tokenizer, tag2idx), batch_size=16, shuffle=True)
    test_loader = DataLoader(PhoBertDataset(test_sents, tokenizer, tag2idx), batch_size=16)
    
    optim = AdamW(model.parameters(), lr=2e-5)
    
    print("Training PhoBERT (3 epochs)...")
    for epoch in range(3):
        model.train()
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            outputs = model(**batch_inputs, labels=batch_labels)
            outputs.loss.backward()
            optim.step()
            optim.zero_grad()
    
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'phobert_model.pth'))
    
    model.eval()
    true_labels, pred_labels = [], []
    idx2tag_local = {v: k for k, v in tag2idx.items()}
    
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            preds = torch.argmax(model(**batch_inputs).logits, dim=2)
            
            for i in range(len(batch_labels)):
                mask = batch_labels[i] != -100
                true_labels.extend([idx2tag_local[t.item()] for t in batch_labels[i][mask]])
                pred_labels.extend([idx2tag_local[p.item()] for p in preds[i][mask]])
                
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- A. SETUP PATHS ---
    path_aug = 'train_bio_augmented.json'
    path_org = 'train_bio.json' # File gốc để so sánh Kịch bản 2
    
    if not os.path.exists(path_aug):
        path_aug = os.path.join('..', 'Data', 'train_bio_augmented.json')
        path_org = os.path.join('..', 'Data', 'train_bio.json')

    print(f">>> Đọc dữ liệu từ: {path_aug}")
    data_aug = load_data(path_aug)
    if not data_aug: print("Lỗi data!"); exit()

    train_sents, test_sents = train_test_split(data_aug, test_size=0.2, random_state=SEED)
    
    TAG2IDX, IDX2TAG = get_tag_mappings(data_aug)
    vocab = list(set([t[0] for s in data_aug for t in s])) + ["UNK", "PAD"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    X_train_cls, y_train_cls = prepare_classical_data(train_sents)
    X_test_cls, y_test_cls = prepare_classical_data(test_sents)
    
    logs = []
    
    # --- B. KỊCH BẢN 1: SO SÁNH 5 MODELS ---
    print("\n" + "="*50)
    print("PHẦN 1: SO SÁNH CÁC MODEL (AUGMENTED DATA)")
    print("="*50)
    
    # 1. CRF
    res = measure_performance("CRF", train_crf, X_train_cls, y_train_cls, X_test_cls, y_test_cls, os.path.join(SAVE_DIR, 'crf.pkl'))
    if res: logs.append(res); f1_crf_aug = res['F1-Score']

    # 2. Logistic Regression
    res = measure_performance("Logistic Regression", train_sklearn, 'LR', X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res: logs.append(res)
    
    # 3. Random Forest
    res = measure_performance("Random Forest", train_sklearn, 'RF', X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res: logs.append(res)
    
    # 4. Bi-LSTM
    res = measure_performance("Bi-LSTM", train_bilstm, train_sents, test_sents, word2idx, TAG2IDX)
    if res: logs.append(res)
    
    # 5. PhoBERT
    res = measure_performance("PhoBERT", train_phobert, train_sents, test_sents, TAG2IDX)
    if res: logs.append(res)
    
    # KHÔI PHỤC: Vẽ biểu đồ so sánh Model (Bar Chart)
    df_results = pd.DataFrame(logs)
    print("\n>>> KẾT QUẢ TỔNG HỢP:")
    print(df_results)
    
    if not df_results.empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # F1 Score
        axes[0].bar(df_results['Model'], df_results['F1-Score'], color='teal')
        axes[0].set_title('F1-Score')
        axes[0].set_ylim(0, 1.0)
        
        # Time
        axes[1].bar(df_results['Model'], df_results['Time (s)'], color='salmon')
        axes[1].set_title('Training Time (s)')
        
        # Memory
        axes[2].bar(df_results['Model'], df_results['Memory (MB)'], color='purple')
        axes[2].set_title('Peak Memory (MB)')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("✅ Đã lưu biểu đồ 'model_comparison.png'")

    # --- C. KỊCH BẢN 2: AUGMENTED vs ORIGINAL (CRF) ---
    print("\n" + "="*50)
    print("PHẦN 2: SO SÁNH AUGMENTED vs ORIGINAL DATA (CRF)")
    print("="*50)
    
    data_org = load_data(path_org)
    if data_org and 'f1_crf_aug' in locals():
        train_org, test_org = train_test_split(data_org, test_size=0.2, random_state=SEED)
        X_tr_org, y_tr_org = prepare_classical_data(train_org)
        X_te_org, y_te_org = prepare_classical_data(test_org)
        
        print("-> Training CRF on Original Data...")
        _, f1_org = train_crf(X_tr_org, y_tr_org, X_te_org, y_te_org)
        
        print(f"CRF Original F1:  {f1_org:.4f}")
        print(f"CRF Augmented F1: {f1_crf_aug:.4f}")
        
        # Vẽ biểu đồ so sánh Data
        plt.figure(figsize=(6, 4))
        plt.bar(['Original', 'Augmented'], [f1_org, f1_crf_aug], color=['gray', 'cyan'])
        plt.title("Impact of Augmented Data (CRF)")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.0)
        plt.savefig('data_impact.png')
        print("✅ Đã lưu biểu đồ 'data_impact.png'")
    else:
        print("⚠️ Không tìm thấy dữ liệu gốc hoặc chưa train CRF Augmented.")

    # --- D. KỊCH BẢN 3: LEARNING CURVE (CRF, LR, RF) ---
    print("\n" + "="*50)
    print("PHẦN 3: LEARNING CURVE (3 MODELS ML)")
    print("="*50)
    
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_results = {'CRF': [], 'LR': [], 'RF': []}
    
    # Flatten tập test một lần dùng chung cho LR/RF
    X_test_flat_full = [item for sublist in X_test_cls for item in sublist]
    y_test_flat_full = [item for sublist in y_test_cls for item in sublist]

    for r in ratios:
        size = int(len(train_sents) * r)
        curr_train = train_sents[:size]
        print(f"\n--- Training {int(r*100)}% Data ({size} câu) ---")
        
        # Prep Data Subset
        X_sub, y_sub = prepare_classical_data(curr_train)
        
        # 1. CRF
        try:
            crf_tmp = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50)
            crf_tmp.fit(X_sub, y_sub)
            y_p = crf_tmp.predict(X_test_cls)
            s = flat_classification_report(y_test_cls, y_p, output_dict=True)['weighted avg']['f1-score']
            lc_results['CRF'].append(s)
        except: lc_results['CRF'].append(0)

        # 2. LR & RF (Cần vectorize lại)
        X_sub_flat = [item for sublist in X_sub for item in sublist]
        y_sub_flat = [item for sublist in y_sub for item in sublist]
        v_tmp = DictVectorizer(sparse=False)
        X_sub_vec = v_tmp.fit_transform(X_sub_flat)
        X_test_vec_tmp = v_tmp.transform(X_test_flat_full)

        try:
            lr = LogisticRegression(max_iter=100, n_jobs=-1)
            lr.fit(X_sub_vec, y_sub_flat)
            yp = lr.predict(X_test_vec_tmp)
            s = classification_report(y_test_flat_full, yp, output_dict=True)['weighted avg']['f1-score']
            lc_results['LR'].append(s)
        except: lc_results['LR'].append(0)

        try:
            rf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
            rf.fit(X_sub_vec, y_sub_flat)
            yp = rf.predict(X_test_vec_tmp)
            s = classification_report(y_test_flat_full, yp, output_dict=True)['weighted avg']['f1-score']
            lc_results['RF'].append(s)
        except: lc_results['RF'].append(0)
        
        print(f"Result: CRF={lc_results['CRF'][-1]:.3f}, LR={lc_results['LR'][-1]:.3f}, RF={lc_results['RF'][-1]:.3f}")

    # Vẽ biểu đồ Learning Curve
    plt.figure(figsize=(10, 6))
    x_axis = [x*100 for x in ratios]
    plt.plot(x_axis, lc_results['CRF'], marker='o', label='CRF')
    plt.plot(x_axis, lc_results['LR'], marker='s', label='Logistic Regression')
    plt.plot(x_axis, lc_results['RF'], marker='^', label='Random Forest')
    
    plt.title("Learning Curve: Data Size vs F1-Score")
    plt.xlabel("% Training Data")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_3models.png')
    print("✅ Đã lưu 'learning_curve_3models.png'")