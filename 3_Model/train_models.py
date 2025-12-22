"""
Script huấn luyện NER: CRF, LR, RF, Bi-LSTM, PhoBERT.
Hỗ trợ: Learning Curve (ML), Lưu Model để Voting.
"""

import os
import json
import time
import tracemalloc
import warnings
import joblib # Để lưu model ML
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
    # TensorFlow (cho Bi-LSTM)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
    
    # PyTorch & Transformers (cho PhoBERT)
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from torch.optim import AdamW
    from tqdm import tqdm # Thanh tiến trình
except ImportError:
    print("Warning: Thiếu thư viện Deep Learning (TensorFlow/PyTorch/Transformers).")

# Tắt cảnh báo
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
# 1. DATA PREPARATION UTILS
# =============================================================================

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_tag_mappings(all_sents):
    tags = list(set([t[1] for sent in all_sents for t in sent]))
    # Đảm bảo padding 'O' hoặc tag đặc biệt
    if "O" not in tags: tags.append("O")
    
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag

# --- Classical ML Feature Engineering ---
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

# --- Bi-LSTM Data Prep ---
def prepare_dl_data(sents, word2idx, tag2idx, max_len):
    X = [[word2idx.get(w[0], word2idx.get("UNK", 0)) for w in s] for s in sents]
    X = pad_sequences(X, maxlen=max_len, padding="post", value=word2idx.get("PAD", 0))
    
    y = [[tag2idx[w[1]] for w in s] for s in sents]
    y = pad_sequences(y, maxlen=max_len, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]
    return np.array(X), np.array(y)

# --- Wrapper đo hiệu năng ---
def measure_performance(model_name, train_func, *args, **kwargs):
    print(f"\n--- ⏱️ Training: {model_name} ---")
    start = time.time()
    try:
        model, f1 = train_func(*args, **kwargs)
    except Exception as e:
        print(f"Lỗi {model_name}: {e}")
        return None
    end = time.time()
    
    print(f"✅ Xong {model_name}: F1={f1:.4f}, Time={end-start:.2f}s")
    return {'Model': model_name, 'F1-Score': f1, 'Time (s)': end-start}

# =============================================================================
# 2. MODEL TRAINING FUNCTIONS
# =============================================================================

def train_crf(X_train, y_train, X_test, y_test):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    
    # Save model
    joblib.dump(crf, os.path.join(SAVE_DIR, 'crf_model.pkl'))
    
    report = flat_classification_report(y_test, y_pred, digits=4, output_dict=True)
    return crf, report['weighted avg']['f1-score']

def train_sklearn(model_type, X_train, y_train, X_test, y_test):
    # Flatten
    X_train_flat = [item for sublist in X_train for item in sublist]
    y_train_flat = [item for sublist in y_train for item in sublist]
    X_test_flat = [item for sublist in X_test for item in sublist]
    y_test_flat = [item for sublist in y_test for item in sublist]
    
    # Vectorize
    v = DictVectorizer(sparse=False)
    X_train_vec = v.fit_transform(X_train_flat)
    X_test_vec = v.transform(X_test_flat)
    
    if model_type == 'LR':
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
    elif model_type == 'RF':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED)
    
    clf.fit(X_train_vec, y_train_flat)
    y_pred = clf.predict(X_test_vec)
    
    # Save model & vectorizer
    joblib.dump(clf, os.path.join(SAVE_DIR, f'{model_type}_model.pkl'))
    joblib.dump(v, os.path.join(SAVE_DIR, 'vectorizer.pkl'))
    
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
    # Tắt verbose để log gọn hơn
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0) 
    
    # Save model
    model.save(os.path.join(SAVE_DIR, 'bilstm_model.h5'))

    # Eval
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    flat_pred, flat_true = [], []
    for i in range(len(test_sents)):
        length = len(test_sents[i])
        flat_pred.extend([IDX2TAG[idx] for idx in y_pred[i][:length]])
        flat_true.extend([IDX2TAG[idx] for idx in y_true[i][:length]])
        
    report = classification_report(flat_true, flat_pred, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# --- PHOBERT SETUP ---
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
    
    train_ds = PhoBertDataset(train_sents, tokenizer, tag2idx)
    test_ds = PhoBertDataset(test_sents, tokenizer, tag2idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16)
    
    optim = AdamW(model.parameters(), lr=2e-5)
    
    # Train 3 epochs để có kết quả tốt hơn
    print("Training PhoBERT epochs...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            
            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Save Model (Quan trọng cho Voting sau này)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'phobert_model.pth'))
    print(f"Đã lưu PhoBERT model vào {SAVE_DIR}/phobert_model.pth")

    # Evaluate
    model.eval()
    true_labels, pred_labels = [], []
    idx2tag_local = {v: k for k, v in tag2idx.items()}
    
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device) # (BS, MaxLen)
            
            outputs = model(**batch_inputs)
            preds = torch.argmax(outputs.logits, dim=2) # (BS, MaxLen)
            
            for i in range(len(batch_labels)):
                # Lọc bỏ -100
                mask = batch_labels[i] != -100
                t_lbls = batch_labels[i][mask]
                p_lbls = preds[i][mask]
                
                true_labels.extend([idx2tag_local[t.item()] for t in t_lbls])
                pred_labels.extend([idx2tag_local[p.item()] for p in p_lbls])
                
    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- A. LOAD DATA ---
    path_aug = 'train_bio_augmented.json' 
    # Check nếu chạy local thì tìm trong ../Data
    if not os.path.exists(path_aug):
        path_aug = os.path.join('..', 'Data', 'train_bio_augmented.json')

    print(f">>> Loading data from: {path_aug}")
    data = load_data(path_aug)
    if not data:
        print("Không tìm thấy dữ liệu!"); exit()

    train_sents, test_sents = train_test_split(data, test_size=0.2, random_state=SEED)
    
    # Global Setup
    TAG2IDX, IDX2TAG = get_tag_mappings(data)
    vocab = list(set([t[0] for s in data for t in s])) + ["UNK", "PAD"]
    word2idx = {w: i for i, w in enumerate(vocab)}
    
    # Prep Classical Data (Full)
    X_train_cls, y_train_cls = prepare_classical_data(train_sents)
    X_test_cls, y_test_cls = prepare_classical_data(test_sents)

    logs = []

    # --- B. FULL TRAINING (Để lấy model cho Voting sau này) ---
    print("\n" + "="*50)
    print("PHẦN 1: TRAINING FULL MODEL & SAVE")
    print("="*50)

    # 1. CRF
    res = measure_performance("CRF", train_crf, X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res: logs.append(res)
    
    # 2. Logistic Regression
    res = measure_performance("Logistic Regression", train_sklearn, 'LR', X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res: logs.append(res)
    
    # 3. Random Forest
    res = measure_performance("Random Forest", train_sklearn, 'RF', X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res: logs.append(res)

    # 4. Bi-LSTM
    res = measure_performance("Bi-LSTM", train_bilstm, train_sents, test_sents, word2idx, TAG2IDX)
    if res: logs.append(res)

    # 5. PhoBERT (Đã kích hoạt)
    res = measure_performance("PhoBERT", train_phobert, train_sents, test_sents, TAG2IDX)
    if res: logs.append(res)

    # In bảng kết quả tổng hợp
    print("\n>>> KẾT QUẢ TỔNG HỢP (FULL DATA):")
    print(pd.DataFrame(logs))

    # --- C. LEARNING CURVE (3 Models: CRF, LR, RF) ---
    print("\n" + "="*50)
    print("PHẦN 2: LEARNING CURVE (3 MODELS ML)")
    print("="*50)
    
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_results = {'CRF': [], 'LR': [], 'RF': []}
    
    # Tập test cố định cho learning curve (đã flatten cho LR/RF)
    X_test_flat_full = [item for sublist in X_test_cls for item in sublist]
    y_test_flat_full = [item for sublist in y_test_cls for item in sublist]

    for r in ratios:
        size = int(len(train_sents) * r)
        curr_train = train_sents[:size]
        print(f"\n--- Ratio {int(r*100)}% ({size} sents) ---")
        
        # Prep Data Subset
        X_sub, y_sub = prepare_classical_data(curr_train)
        
        # 1. CRF LC
        try:
            crf_tmp = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50)
            crf_tmp.fit(X_sub, y_sub)
            y_p = crf_tmp.predict(X_test_cls)
            s = flat_classification_report(y_test_cls, y_p, output_dict=True)['weighted avg']['f1-score']
            lc_results['CRF'].append(s)
        except: lc_results['CRF'].append(0)

        # Vectorize cho LR/RF (Fit trên subset mới đúng luật)
        X_sub_flat = [item for sublist in X_sub for item in sublist]
        y_sub_flat = [item for sublist in y_sub for item in sublist]
        v_tmp = DictVectorizer(sparse=False)
        X_sub_vec = v_tmp.fit_transform(X_sub_flat)
        X_test_vec_tmp = v_tmp.transform(X_test_flat_full)

        # 2. LR LC
        try:
            lr = LogisticRegression(max_iter=100, n_jobs=-1)
            lr.fit(X_sub_vec, y_sub_flat)
            yp = lr.predict(X_test_vec_tmp)
            s = classification_report(y_test_flat_full, yp, output_dict=True)['weighted avg']['f1-score']
            lc_results['LR'].append(s)
        except: lc_results['LR'].append(0)

        # 3. RF LC
        try:
            rf = RandomForestClassifier(n_estimators=30, n_jobs=-1)
            rf.fit(X_sub_vec, y_sub_flat)
            yp = rf.predict(X_test_vec_tmp)
            s = classification_report(y_test_flat_full, yp, output_dict=True)['weighted avg']['f1-score']
            lc_results['RF'].append(s)
        except: lc_results['RF'].append(0)
        
        print(f"Result: CRF={lc_results['CRF'][-1]:.3f}, LR={lc_results['LR'][-1]:.3f}, RF={lc_results['RF'][-1]:.3f}")

    # Vẽ và lưu biểu đồ
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
    print("\n✅ Đã lưu biểu đồ Learning Curve vào 'learning_curve_3models.png'")
    print(f"✅ Đã lưu toàn bộ models vào thư mục '{SAVE_DIR}' để dùng cho Voting.")