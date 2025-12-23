"""
Script huấn luyện và so sánh các mô hình NER (CRF, Machine Learning, Bi-LSTM)
Được chuyển đổi từ Jupyter Notebook.
Đã sửa lỗi logic biến và bổ sung SVM đầy đủ.
"""

import os
import json
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Imports
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import confusion_matrix

# Deep Learning Imports (TensorFlow/Keras)
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
    from tensorflow.keras.backend import clear_session
except ImportError:
    print("Warning: TensorFlow không được cài đặt hoặc lỗi import.")

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

# --- Feature Engineering cho Classical ML (CRF, LR, RF, SVM) ---
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
    """Hàm wrapper để đo thời gian và bộ nhớ."""
    print(f"\n--- ⏱️ Bắt đầu đo lường: {model_name} ---")
    tracemalloc.start()
    start_time = time.time()
    
    try:
        model, f1_score = train_func(*args, **kwargs)
    except Exception as e:
        print(f" Lỗi khi chạy {model_name}: {e}")
        tracemalloc.stop()
        return None
    
    end_time = time.time()
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

def plot_ner_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix", filename=None):
    """Vẽ và lưu Confusion Matrix cho bài toán NER."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Chuẩn hóa
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"  [Info] Đã lưu biểu đồ: {filename}")
    # plt.show() # Comment lại nếu chạy batch để tránh pop-up liên tục
    plt.close()

# =============================================================================
# PHẦN 2: CÁC HÀM TRAIN MODEL
# =============================================================================

def train_crf(X_train, y_train, X_test, y_test):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=500)
    crf.fit(X_train, y_train)
    
    y_pred = crf.predict(X_test)
    
    flat_true = [item for sublist in y_test for item in sublist]
    flat_pred = [item for sublist in y_pred for item in sublist]
    labels = sorted(list(set(flat_true + flat_pred)))
    
    plot_ner_confusion_matrix(flat_true, flat_pred, labels, title="CRF Confusion Matrix", filename="cm_crf.png")
    
    print("\n" + "="*40 + "\n>>> DETAILED REPORT: CRF\n" + "="*40)
    print(flat_classification_report(y_test, y_pred, digits=4))
    
    report = flat_classification_report(y_test, y_pred, digits=4, output_dict=True)
    return crf, report['weighted avg']['f1-score']

def train_classical_sklearn(model_name, X_train, y_train, X_test, y_test):
    print(f"... Đang training {model_name} ...")
    
    X_train_flat = [item for sublist in X_train for item in sublist]
    y_train_flat = [item for sublist in y_train for item in sublist]
    X_test_flat = [item for sublist in X_test for item in sublist]
    y_test_flat = [item for sublist in y_test for item in sublist]
    
    v = DictVectorizer(sparse=False)
    X_train_vec = v.fit_transform(X_train_flat)
    X_test_vec = v.transform(X_test_flat)
    
    if model_name == 'LogisticRegression':
        clf = LogisticRegression(max_iter=500, n_jobs=-1, random_state=SEED)
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=SEED)
    elif model_name == 'SVM':
        print("   -> Sử dụng LinearSVC...")
        clf = LinearSVC(random_state=SEED, max_iter=500, dual=False) 
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ.")
        
    clf.fit(X_train_vec, y_train_flat)
    y_pred = clf.predict(X_test_vec)
    
    labels = sorted(list(set(y_test_flat + list(y_pred))))
    try:
        plot_ner_confusion_matrix(y_test_flat, y_pred, labels, 
                                  title=f"{model_name} Confusion Matrix", 
                                  filename=f"cm_{model_name.replace(' ', '_')}.png")
    except NameError: pass

    print("\n" + "="*40 + f"\n>>> DETAILED REPORT: {model_name}\n" + "="*40)
    print(classification_report(y_test_flat, y_pred, digits=4))
    
    report = classification_report(y_test_flat, y_pred, digits=4, output_dict=True)
    return clf, report['weighted avg']['f1-score']


def train_bilstm(train_sents, test_sents, word2idx, tag2idx):
    global IDX2TAG 
    
    X_train, y_train = prepare_dl_data(train_sents, word2idx, tag2idx, MAX_LEN)
    X_test, y_test = prepare_dl_data(test_sents, word2idx, tag2idx, MAX_LEN)
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

    model = Sequential([
        Embedding(input_dim=len(word2idx), output_dim=50), 
        Bidirectional(LSTM(units=64, return_sequences=True)),
        Dropout(0.3),
        TimeDistributed(Dense(len(tag2idx), activation="softmax"))
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=32, epochs=50, verbose=1)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    flat_pred, flat_true = [], []
    for i in range(len(test_sents)):
        true_len = len(test_sents[i])
        flat_pred.extend([IDX2TAG[idx] for idx in y_pred[i][:true_len]])
        flat_true.extend([IDX2TAG[idx] for idx in y_true[i][:true_len]])
    
    labels = sorted([t for t in tag2idx.keys() if t not in ["PAD", "UNK"]])
    plot_ner_confusion_matrix(flat_true, flat_pred, labels, title="Bi-LSTM Confusion Matrix", filename="cm_bilstm.png")

    print("\n" + "="*40 + "\n>>> DETAILED REPORT: Bi-LSTM\n" + "="*40)
    print(classification_report(flat_true, flat_pred, digits=4))

    report = classification_report(flat_true, flat_pred, digits=4, output_dict=True)
    return model, report['weighted avg']['f1-score']

# =============================================================================
# PHẦN 3: MAIN EXECUTION FLOW
# =============================================================================

if __name__ == "__main__":
    # --- 1. Setup Dữ liệu ---
    current_folder = os.getcwd()
    data_folder_path = os.path.abspath(os.path.join(current_folder, '..', 'Data'))
    path_aug = os.path.join(data_folder_path, 'train_bio_augmented.json')
    path_org = os.path.join(data_folder_path, 'train_bio.json')

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
        f1_crf_aug = res_crf['F1-Score'] # [ĐÃ SỬA] Lưu vào biến để dùng sau

    # 2. Logistic Regression
    res_lr = measure_performance("Logistic Regression", train_classical_sklearn, "LogisticRegression", X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_lr: 
        performance_log.append(res_lr)
        f1_lr_aug = res_lr['F1-Score'] # [ĐÃ SỬA]

    # 3. Random Forest
    res_rf = measure_performance("Random Forest", train_classical_sklearn, "RandomForest", X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_rf: 
        performance_log.append(res_rf)
        f1_rf_aug = res_rf['F1-Score'] # [ĐÃ SỬA]

    # 4. SVM
    res_svm = measure_performance("SVM", train_classical_sklearn, "SVM", X_train_cls, y_train_cls, X_test_cls, y_test_cls)
    if res_svm: 
        performance_log.append(res_svm)
        f1_svm_aug = res_svm['F1-Score'] # [ĐÃ SỬA]

    # 5. Bi-LSTM
    res_lstm = measure_performance("Bi-LSTM", train_bilstm, train_sents_aug, test_sents_aug, word2idx, TAG2IDX)
    if res_lstm: 
        performance_log.append(res_lstm)
        f1_lstm_aug = res_lstm['F1-Score'] # [ĐÃ SỬA]

    # Tổng hợp kết quả
    df_results = pd.DataFrame(performance_log)
    print("\n>>> BẢNG TỔNG HỢP SO SÁNH:")
    print(df_results)
    
    if not df_results.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(df_results['Model'], df_results['F1-Score'], color='teal')
        plt.title('So sánh F1-Score các mô hình')
        plt.ylim(0, 1.0)
        plt.savefig('model_comparison.png')
        plt.close()

    # --- KỊCH BẢN 2: SO SÁNH AUGMENTED vs ORIGINAL (ALL MODELS) ---
    print("\n" + "="*50)
    print("PHẦN 2: SO SÁNH AUGMENTED vs ORIGINAL DATA")
    print("="*50)
    
    data_org = load_data(path_org)
    
    # [ĐÃ SỬA] Kiểm tra đủ biến f1_..._aug
    if data_org and 'f1_crf_aug' in locals():
        train_sents_org, test_sents_org = train_test_split(data_org, test_size=0.2, random_state=SEED)
        
        # Classical Data Prep
        X_tr_org, y_tr_org = prepare_classical_data(train_sents_org)
        X_te_org, y_te_org = prepare_classical_data(test_sents_org)
        
        # --- A. CRF (Original) ---
        print("\n-> [1/5] Training CRF on Original Data...")
        _, f1_crf_org = train_crf(X_tr_org, y_tr_org, X_te_org, y_te_org)
        
        # --- B. Logistic Regression (Original) ---
        print("\n-> [2/5] Training Logistic Regression on Original Data...")
        _, f1_lr_org = train_classical_sklearn('LogisticRegression', X_tr_org, y_tr_org, X_te_org, y_te_org)
        
        # --- C. Random Forest (Original) ---
        print("\n-> [3/5] Training Random Forest on Original Data...")
        _, f1_rf_org = train_classical_sklearn('RandomForest', X_tr_org, y_tr_org, X_te_org, y_te_org)
        
        # --- D. SVM (Original) [ĐÃ SỬA: Thêm phần này] ---
        print("\n-> [4/5] Training SVM on Original Data...")
        _, f1_svm_org = train_classical_sklearn('SVM', X_tr_org, y_tr_org, X_te_org, y_te_org)
        
        # --- E. Bi-LSTM (Original) ---
        print("\n-> [5/5] Training Bi-LSTM on Original Data...")
        _, f1_lstm_org = train_bilstm(train_sents_org, test_sents_org, word2idx, TAG2IDX)

        # --- TỔNG HỢP SO SÁNH [ĐÃ SỬA: Thêm SVM vào Dataframe] ---
        comparison_data = {
            'Model': ['CRF', 'Logistic Reg', 'Random Forest', 'SVM', 'Bi-LSTM'],
            'Original F1': [f1_crf_org, f1_lr_org, f1_rf_org, f1_svm_org, f1_lstm_org],
            'Augmented F1': [f1_crf_aug, f1_lr_aug, f1_rf_aug, f1_svm_aug, f1_lstm_aug]
        }
        
        df_comp = pd.DataFrame(comparison_data)
        df_comp['Improvement'] = df_comp['Augmented F1'] - df_comp['Original F1']
        
        print("\n>>> BẢNG SO SÁNH CHI TIẾT:")
        print(df_comp)
        
        df_comp.plot(x='Model', y=['Original F1', 'Augmented F1'], kind='bar', figsize=(10, 6), color=['gray', 'teal'])
        plt.title("Impact of Data Augmentation on Models")
        plt.ylabel("F1 Score")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('augmentation_impact_all_models.png')
        plt.close()
        print("✅ Đã lưu biểu đồ so sánh vào 'augmentation_impact_all_models.png'")
        
    else:
        print("⚠️ Bỏ qua Kịch bản 2 do thiếu dữ liệu gốc hoặc chưa chạy xong Kịch bản 1.")

    # --- KỊCH BẢN 3: LEARNING CURVE (Ảnh hưởng kích thước dữ liệu) ---
    print("\n=== KỊCH BẢN 3: LEARNING CURVE ===")

    # A. Cho Classical ML
    X_test_cls, y_test_cls = prepare_classical_data(test_sents_aug)
    X_test_flat_full = [item for sublist in X_test_cls for item in sublist]
    y_test_flat_full = [item for sublist in y_test_cls for item in sublist]
    
    # B. Cho Deep Learning
    X_test_dl, y_test_dl = prepare_dl_data(test_sents_aug, word2idx, TAG2IDX, MAX_LEN)

    # Cấu hình vòng lặp
    ratios = [0.33, 0.66, 1.0]
    # [ĐÃ SỬA] Thêm key 'SVM' vào đây
    lc_results = {'CRF': [], 'LR': [], 'RF': [], 'SVM': [], 'Bi-LSTM': []} 
    
    for r in ratios:
        subset_size = int(len(train_sents_aug) * r)
        current_sents = train_sents_aug[:subset_size]
        print(f"\n--- Training với {int(r*100)}% dữ liệu ({subset_size} câu) ---")
        
        # --- A. Prep Data cho Classical ---
        X_sub_cls, y_sub_cls = prepare_classical_data(current_sents)
        
        # 1. CRF
        try:
            crf_tmp = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=500)
            crf_tmp.fit(X_sub_cls, y_sub_cls)
            y_pred_crf = crf_tmp.predict(X_test_cls)
            s_crf = flat_classification_report(y_test_cls, y_pred_crf, output_dict=True)['weighted avg']['f1-score']
            lc_results['CRF'].append(s_crf)
        except Exception as e: lc_results['CRF'].append(0)
        
        # Prep Data Vectorized cho ML thường
        X_sub_flat = [item for sublist in X_sub_cls for item in sublist]
        y_sub_flat = [item for sublist in y_sub_cls for item in sublist]
        
        v_tmp = DictVectorizer(sparse=False)
        X_sub_vec = v_tmp.fit_transform(X_sub_flat)
        X_test_vec_tmp = v_tmp.transform(X_test_flat_full) 
        
        # 2. LR
        try:
            lr_tmp = LogisticRegression(max_iter=500, n_jobs=-1)
            lr_tmp.fit(X_sub_vec, y_sub_flat)
            y_pred_lr = lr_tmp.predict(X_test_vec_tmp)
            s_lr = classification_report(y_test_flat_full, y_pred_lr, output_dict=True)['weighted avg']['f1-score']
            lc_results['LR'].append(s_lr)
        except: lc_results['LR'].append(0)

        # 3. RF
        try:
            rf_tmp = RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=SEED)
            rf_tmp.fit(X_sub_vec, y_sub_flat)
            y_pred_rf = rf_tmp.predict(X_test_vec_tmp)
            s_rf = classification_report(y_test_flat_full, y_pred_rf, output_dict=True)['weighted avg']['f1-score']
            lc_results['RF'].append(s_rf)
        except: lc_results['RF'].append(0)

        try:
            svm_tmp = LinearSVC(random_state=SEED, max_iter=500, dual=False)
            svm_tmp.fit(X_sub_vec, y_sub_flat)
            y_pred_svm = svm_tmp.predict(X_test_vec_tmp)
            s_svm = classification_report(y_test_flat_full, y_pred_svm, output_dict=True)['weighted avg']['f1-score']
            lc_results['SVM'].append(s_svm)
        except Exception as e: 
            print(f"Lỗi SVM: {e}")
            lc_results['SVM'].append(0)
        
        # 5. Bi-LSTM
        try: 
            import tensorflow.keras.backend as K
            K.clear_session()
            X_sub_dl, y_sub_dl = prepare_dl_data(current_sents, word2idx, TAG2IDX, MAX_LEN)
            X_tr, X_val, y_tr, y_val = train_test_split(X_sub_dl, y_sub_dl, test_size=0.1, random_state=SEED)
            
            model_tmp = Sequential([
                Embedding(input_dim=len(word2idx), output_dim=50), 
                Bidirectional(LSTM(units=64, return_sequences=True)),
                Dropout(0.3),
                TimeDistributed(Dense(len(TAG2IDX), activation="softmax"))
            ])
            model_tmp.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            model_tmp.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=32, epochs=50, verbose=1)
            
            y_pred_probs_tmp = model_tmp.predict(X_test_dl, verbose=0)
            y_pred_tmp = np.argmax(y_pred_probs_tmp, axis=-1)
            y_true_tmp = np.argmax(y_test_dl, axis=-1)
            
            flat_pred_tmp, flat_true_tmp = [], []
            for i in range(len(test_sents_aug)):
                true_len = len(test_sents_aug[i])
                flat_pred_tmp.extend([IDX2TAG[idx] for idx in y_pred_tmp[i][:true_len]])
                flat_true_tmp.extend([IDX2TAG[idx] for idx in y_true_tmp[i][:true_len]])
            
            s_lstm = classification_report(flat_true_tmp, flat_pred_tmp, output_dict=True)['weighted avg']['f1-score']
            lc_results['Bi-LSTM'].append(s_lstm)
        except Exception as e: 
            lc_results['Bi-LSTM'].append(0)

        print(f"Result: CRF={lc_results['CRF'][-1]:.3f}, SVM={lc_results['SVM'][-1]:.3f}, RF={lc_results['RF'][-1]:.3f},  LSTM={lc_results['Bi-LSTM'][-1]:.3f}")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    x_axis = [r*100 for r in ratios]
    for name, scores in lc_results.items():
        if scores: 
            plt.plot(x_axis, scores, marker='o', label=name)
            
    plt.title("Learning Curve: Data Size vs F1-Score")
    plt.xlabel("% Training Data")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curve_all_models.png')
    plt.close()
    print("✅ Hoàn tất toàn bộ kịch bản.")