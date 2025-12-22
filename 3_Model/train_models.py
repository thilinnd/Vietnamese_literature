"""
Script huấn luyện và so sánh các mô hình NER (CRF, Machine Learning, Bi-LSTM)
Được chuyển đổi từ Jupyter Notebook.
"""

import os
import json
import time
import tracemalloc
import warnings
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

# Deep Learning Imports (TensorFlow/Keras)
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
except ImportError:
    print("Warning: TensorFlow không được cài đặt hoặc lỗi import.")

# PhoBERT / PyTorch Imports
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from torch.optim import AdamW
except ImportError:
    print("Warning: PyTorch hoặc Transformers không được cài đặt.")

# Bỏ qua các warning không cần thiết
warnings.filterwarnings("ignore")

# --- CẤU HÌNH TOÀN CỤC ---
SEED = 42
MAX_LEN = 128  # Độ dài câu tối đa cho DL models
SAVE_DIR = 'saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# Biến toàn cục map nhãn (sẽ cập nhật khi load data)
TAG2IDX = {}
IDX2TAG = {}

# =============================================================================
# PHẦN 1: DATA UTILS & FEATURE ENGINEERING
# =============================================================================

def load_data(file_path):
    """Load dữ liệu từ file JSON."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {file_path}")
        return []

def get_tag_mappings(all_sents):
    """Tạo mapping từ điển cho labels."""
    tags = list(set([t[1] for sent in all_sents for t in sent]))
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}
    return tag2idx, idx2tag

# --- Feature Engineering cho Classical ML (CRF, LR, RF) ---
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
    else: 
        features['BOS'] = True
    
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle()})
    else: 
        features['EOS'] = True
    return features

def prepare_classical_data(sents):
    X = [[word2features(s, i) for i in range(len(s))] for s in sents]
    y = [[label for token, label in s] for s in sents]
    return X, y

# --- Feature Engineering cho Bi-LSTM ---
def prepare_dl_data(sents, word2idx, tag2idx, max_len):
    X = [[word2idx.get(w[0], word2idx.get("UNK", 0)) for w in s] for s in sents]
    X = pad_sequences(X, maxlen=max_len, padding="post", value=word2idx.get("PAD", 0))
    
    y = [[tag2idx[w[1]] for w in s] for s in sents]
    y = pad_sequences(y, maxlen=max_len, padding="post", value=tag2idx["O"])
    y = [to_categorical(i, num_classes=len(tag2idx)) for i in y]
    return np.array(X), np.array(y)

def measure_performance(model_name, train_func, *args, **kwargs):
    """
    Hàm wrapper để đo thời gian và bộ nhớ của một hàm huấn luyện.
    """
    print(f"\n--- ⏱️ Bắt đầu đo lường: {model_name} ---")
    
    # 1. Bắt đầu theo dõi RAM
    tracemalloc.start()
    
    # 2. Bắt đầu bấm giờ
    start_time = time.time()
    
    # 3. Chạy hàm huấn luyện thực tế
    try:
        model, f1_score = train_func(*args, **kwargs)
    except Exception as e:
        print(f" Lỗi khi chạy {model_name}: {e}")
        tracemalloc.stop()
        return None
    
    # 4. Kết thúc bấm giờ
    end_time = time.time()
    
    # 5. Lấy thông số RAM (current, peak)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time = end_time - start_time
    peak_memory_mb = peak / (1024 * 1024) # Đổi từ Byte sang MB
    
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
# PHẦN 2: CÁC HÀM TRAIN MODEL
# =============================================================================

def train_crf(X_train, y_train, X_test, y_test):
    print("... Đang training CRF ...")
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
    crf.fit(X_train, y_train)
    
    y_pred = crf.predict(X_test)
    
    print("\n" + "="*40)
    print(">>> DETAILED REPORT: CRF")
    print("="*40)
    print(flat_classification_report(y_test, y_pred, digits=4))
    
    report = flat_classification_report(y_test, y_pred, digits=4, output_dict=True)
    return crf, report['weighted avg']['f1-score']

def train_classical_sklearn(model_name, X_train, y_train, X_test, y_test):
    print(f"... Đang training {model_name} ...")
    
    # Flatten & Vectorize
    X_train_flat = [item for sublist in X_train for item in sublist]
    y_train_flat = [item for sublist in y_train for item in sublist]
    X_test_flat = [item for sublist in X_test for item in sublist]
    y_test_flat = [item for sublist in y_test for item in sublist]
    
    v = DictVectorizer(sparse=False)
    X_train_vec = v.fit_transform(X_train_flat)
    X_test_vec = v.transform(X_test_flat)
    
    if model_name == 'LogisticRegression':
        clf = LogisticRegression(max_iter=500, n_jobs=-1)
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED)
    else:
        raise ValueError("Model not supported")
        
    clf.fit(X_train_vec, y_train_flat)
    y_pred = clf.predict(X_test_vec)
    
    print("\n" + "="*40)
    print(f">>> DETAILED REPORT: {model_name}")
    print("="*40)
    print(classification_report(y_test_flat, y_pred, digits=4))
    
    report = classification_report(y_test_flat, y_pred, digits=4, output_dict=True)
    return clf, report['weighted avg']['f1-score']

def train_bilstm(train_sents, test_sents, word2idx, tag2idx):
    print("... Đang training Bi-LSTM ...")
    # Lưu ý: Cần import global hoặc truyền vào IDX2TAG để giải mã
    global IDX2TAG 
    
    X_train, y_train = prepare_dl_data(train_sents, word2idx, tag2idx, MAX_LEN)
    X_test, y_test = prepare_dl_data(test_sents, word2idx, tag2idx, MAX_LEN)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

    model = Sequential([
        Embedding(input_dim=len(word2idx), output_dim=50), # Keras mới tự hiểu input_length
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.3),
        TimeDistributed(Dense(len(tag2idx), activation="softmax"))
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=32, epochs=30, verbose=1)

    # Evaluate
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    flat_pred, flat_true = [], []
    for i in range(len(test_sents)):
        true_len = len(test_sents[i])
        # Cắt padding
        flat_pred.extend([IDX2TAG[idx] for idx in y_pred[i][:true_len]])
        flat_true.extend([IDX2TAG[idx] for idx in y_true[i][:true_len]])
    
    print("\n" + "="*40)
    print(">>> DETAILED REPORT: Bi-LSTM")
    print("="*40)
    print(classification_report(flat_true, flat_pred, digits=4))

    report = classification_report(flat_true, flat_pred, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# --- PHẦN CHO PHOBERT (Dự phòng) ---
class NERDataset(Dataset):
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

        # Tokenize & Align labels
        tokenized_inputs = self.tokenizer(words, is_split_into_words=True, 
                                          padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt")
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids': tokenized_inputs['input_ids'][0],
            'attention_mask': tokenized_inputs['attention_mask'][0],
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

def train_phobert(train_sents, test_sents, tag2idx):
    print("... Training PhoBERT (vinai/phobert-base) ...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base-v2", num_labels=len(tag2idx))
    model.to(device)
    
    train_dataset = NERDataset(train_sents, tokenizer, tag2idx)
    test_dataset = NERDataset(test_sents, tokenizer, tag2idx)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train Loop (1 epoch demo)
    model.train()
    for epoch in range(1):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Eval
    model.eval()
    true_labels, pred_labels = [], []
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=2)
            
            for i in range(len(labels)):
                true_ids = labels[i][labels[i] != -100]
                pred_ids = preds[i][labels[i] != -100]
                true_labels.extend([idx2tag[t.item()] for t in true_ids])
                pred_labels.extend([idx2tag[p.item()] for p in pred_ids])

    report = classification_report(true_labels, pred_labels, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# =============================================================================
# PHẦN 3: MAIN EXECUTION FLOW
# =============================================================================

if __name__ == "__main__":
    # --- 1. Setup Dữ liệu ---
    # Giả sử bạn đang chạy trong thư mục chứa code và data nằm ở thư mục cha/Data
    # Bạn cần điều chỉnh đường dẫn này cho phù hợp với môi trường của mình (Local/Kaggle)
    
    # Ví dụ đường dẫn trên Kaggle thường là: /kaggle/input/tên-dataset/train_bio_augmented.json
    # Dưới đây là logic tự động tìm đường dẫn tương đối như trong notebook cũ
    current_folder = os.getcwd()
    # Nếu chạy local:
    data_folder_path = os.path.abspath(os.path.join(current_folder, '..', 'Data'))
    path_aug = os.path.join(data_folder_path, 'train_bio_augmented.json')
    path_org = os.path.join(data_folder_path, 'train_bio.json')

    # Nếu file không tồn tại, thử tìm ở thư mục hiện tại (cho Kaggle nếu upload vào working)
    if not os.path.exists(path_aug):
        path_aug = 'train_bio_augmented.json'
        path_org = 'train_bio.json'

    print(f"Đang đọc dữ liệu từ: {path_aug}")
    data_aug = load_data(path_aug)
    
    if not data_aug:
        print("Không tìm thấy dữ liệu. Vui lòng kiểm tra đường dẫn.")
        exit()

    # Split Data
    train_sents_aug, test_sents_aug = train_test_split(data_aug, test_size=0.2, random_state=SEED)
    
    # Setup Vocab & Tags Global
    words = list(set([t[0] for sent in data_aug for t in sent])) + ["UNK", "PAD"]
    TAG2IDX, IDX2TAG = get_tag_mappings(data_aug)
    word2idx = {w: i for i, w in enumerate(words)}
    
    # --- KỊCH BẢN 1: SO SÁNH CÁC MODEL (Augmented Data) ---
    print("\n=== KỊCH BẢN 1: SO SÁNH HIỆU SUẤT CÁC MODEL ===")
    
    performance_log = []
    
    # Prepare Data for Classical ML
    X_train_cls, y_train_cls = prepare_classical_data(train_sents_aug)
    X_test_cls, y_test_cls = prepare_classical_data(test_sents_aug)

    # 1. CRF
    res_crf = measure_performance("CRF", train_crf, X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_crf: 
        performance_log.append(res_crf)
        f1_crf = res_crf['F1-Score'] # Lưu lại để so sánh sau

    # 2. Logistic Regression
    res_lr = measure_performance("Logistic Regression", train_classical_sklearn, "LogisticRegression", X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_lr: performance_log.append(res_lr)

    # 3. Random Forest
    res_rf = measure_performance("Random Forest", train_classical_sklearn, "RandomForest", X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_rf: performance_log.append(res_rf)

    # 4. Bi-LSTM
    res_lstm = measure_performance("Bi-LSTM", train_bilstm, train_sents_aug, test_sents_aug, word2idx, TAG2IDX)
    if res_lstm: performance_log.append(res_lstm)
    
    # 5. PhoBERT (Optional - Bỏ comment nếu muốn chạy và đã cài đủ thư viện)
    # res_bert = measure_performance("PhoBERT", train_phobert, train_sents_aug, test_sents_aug, TAG2IDX)
    # if res_bert: performance_log.append(res_bert)

    # Tổng hợp kết quả
    df_results = pd.DataFrame(performance_log)
    print("\n>>> BẢNG TỔNG HỢP SO SÁNH:")
    print(df_results)
    
    # Lưu biểu đồ so sánh
    if not df_results.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(df_results['Model'], df_results['F1-Score'], color='teal')
        plt.title('So sánh F1-Score các mô hình')
        plt.ylim(0, 1.0)
        plt.savefig('model_comparison.png') # Lưu ảnh thay vì show()
        print("Đã lưu biểu đồ vào 'model_comparison.png'")

    # --- KỊCH BẢN 2: SO SÁNH AUGMENTED vs ORIGINAL (CRF) ---
    print("\n=== KỊCH BẢN 2: AUGMENTED vs ORIGINAL (CRF) ===")
    data_org = load_data(path_org)
    
    if data_org and 'f1_crf' in locals():
        train_sents_org, test_sents_org = train_test_split(data_org, test_size=0.2, random_state=SEED)
        X_train_org, y_train_org = prepare_classical_data(train_sents_org)
        X_test_org, y_test_org = prepare_classical_data(test_sents_org)
        
        print("-> Training CRF on Original Data...")
        _, f1_org = train_crf(X_train_org, y_train_org, X_test_org, y_test_org)
        
        print(f"CRF Original F1:  {f1_org:.4f}")
        print(f"CRF Augmented F1: {f1_crf:.4f}")
    else:
        print("Bỏ qua kịch bản 2 do thiếu dữ liệu gốc hoặc chưa chạy CRF.")

    # --- KỊCH BẢN 3: LEARNING CURVE (Ảnh hưởng kích thước dữ liệu) ---
    print("\n=== KỊCH BẢN 3: LEARNING CURVE ===")
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    lc_results = {'CRF': [], 'Logistic Regression': [], 'Random Forest': []}
    
    # Tập test cố định (Full)
    X_test_flat_full = [item for sublist in X_test_cls for item in sublist]
    y_test_flat_full = [item for sublist in y_test_cls for item in sublist]
    
    for r in ratios:
        subset_size = int(len(train_sents_aug) * r)
        current_sents = train_sents_aug[:subset_size]
        print(f"\n--- Training với {int(r*100)}% dữ liệu ({subset_size} câu) ---")
        
        # Prep Data
        X_sub, y_sub = prepare_classical_data(current_sents)
        
        # 1. CRF
        try:
            crf_tmp = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=50)
            crf_tmp.fit(X_sub, y_sub)
            y_pred_crf = crf_tmp.predict(X_test_cls)
            s_crf = flat_classification_report(y_test_cls, y_pred_crf, output_dict=True)['weighted avg']['f1-score']
            lc_results['CRF'].append(s_crf)
        except: lc_results['CRF'].append(0)
        
        # Prep Data cho LR/RF (cần vectorize lại theo vocab của tập con)
        X_sub_flat = [item for sublist in X_sub for item in sublist]
        y_sub_flat = [item for sublist in y_sub for item in sublist]
        
        v_tmp = DictVectorizer(sparse=False)
        X_sub_vec = v_tmp.fit_transform(X_sub_flat)
        X_test_vec_tmp = v_tmp.transform(X_test_flat_full)
        
        # 2. LR
        try:
            lr_tmp = LogisticRegression(max_iter=100, n_jobs=-1)
            lr_tmp.fit(X_sub_vec, y_sub_flat)
            y_pred_lr = lr_tmp.predict(X_test_vec_tmp)
            s_lr = classification_report(y_test_flat_full, y_pred_lr, output_dict=True)['weighted avg']['f1-score']
            lc_results['Logistic Regression'].append(s_lr)
        except: lc_results['Logistic Regression'].append(0)

        # 3. RF
        try:
            rf_tmp = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=SEED)
            rf_tmp.fit(X_sub_vec, y_sub_flat)
            y_pred_rf = rf_tmp.predict(X_test_vec_tmp)
            s_rf = classification_report(y_test_flat_full, y_pred_rf, output_dict=True)['weighted avg']['f1-score']
            lc_results['Random Forest'].append(s_rf)
        except: lc_results['Random Forest'].append(0)
        
        print(f"Result: CRF={lc_results['CRF'][-1]:.3f}, LR={lc_results['Logistic Regression'][-1]:.3f}")

    # Vẽ biểu đồ Learning Curve
    plt.figure(figsize=(10, 6))
    x_axis = [r*100 for r in ratios]
    for name, scores in lc_results.items():
        plt.plot(x_axis, scores, marker='o', label=name)
    plt.title("Learning Curve: Data Size vs F1-Score")
    plt.xlabel("% Training Data")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Đã lưu biểu đồ vào 'learning_curve.png'")