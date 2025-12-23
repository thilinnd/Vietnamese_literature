import streamlit as st
import pickle
import re
import os
import pandas as pd
from pyvi import ViTokenizer

st.set_page_config(
    page_title="Bài toán NER - Văn học Việt Nam",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap');

    /* 1. Nút bấm (Style cũ của bạn) */
    div.stButton > button:first-child {
        background-color: #795548;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        width: 100%;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #5d4037;
        color: white;
    }

    /* 2. Khung nền mờ (Dùng cho Bảng chú thích & Kết quả) */
    .translucent-box {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        backdrop-filter: blur(8px);
    }

    /* 3. Style cho bảng */
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
    }
    table.custom-table th {
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #795548;
        color: #5d4037;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 0.85rem;
    }
    table.custom-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
        vertical-align: middle;
    }

    /* 4. Tiêu đề section (Dùng cho khung kết quả) */
    .section-title {
        color: #795548;
        font-family: 'Source Serif Pro', serif;
        font-weight: bold;
        font-size: 22px;
        margin-bottom: 15px;
        border-left: 5px solid #795548;
        padding-left: 12px;
    }
    
    /* 5. Header to */
    .header-style {
        background: linear-gradient(to right, #795548, #4e342e);
        color: white; 
        padding: 25px; 
        border-radius: 15px; 
        text-align: center; 
        font-family: 'Source Serif Pro', serif; 
        font-size: 50px; 
        font-weight: bold; 
        margin-bottom: 30px;
        white-space: nowrap; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        width: 100%;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://i.pinimg.com/1200x/08/fb/3b/08fb3ba21d7b06a1a4d345a055ab817a.jpg")

tag_colors = {
    "CHAR": "#d32f2f", "PER": "#7b1fa2", "WORK": "#303f9f",
    "LOC": "#00796b", "TIME/DATE": "#f57f17", "ORG": "#3e2723"
}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, '..', 'saved_models')

@st.cache_resource
def load_crf_model():
    try:
        path = os.path.join(SAVE_DIR, 'crf_model.pkl')
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        return None

def preprocess_text_for_prediction(text):
    tokenized_text = ViTokenizer.tokenize(text)
    raw_tokens = tokenized_text.split()
    regex_pattern = r'[^\w\s\d_ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂÂÊÔƠƯưăâêôơư]'
    cleaned_tokens = [re.sub(regex_pattern, '', token) for token in raw_tokens if re.sub(regex_pattern, '', token)]
    return cleaned_tokens

def get_features_for_prediction(sent):
    sent_features = []
    for i in range(len(sent)):
        word = sent[i]
        features = {
            'bias': 1.0, 'word.lower()': word.lower(), 'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(), 'word.isdigit()': word.isdigit(), 'word.has_underscore': '_' in word,
        }
        if i > 0:
            word1 = sent[i-1]
            features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle()})
        else: features['BOS'] = True
        if i < len(sent)-1:
            word1 = sent[i+1]
            features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle()})
        else: features['EOS'] = True
        sent_features.append(features)
    return sent_features

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None


st.markdown('<div class="header-style">HỆ THỐNG NHẬN DIỆN THỰC THỂ VĂN HỌC VIỆT NAM</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([5, 6], gap="large")

crf_model = load_crf_model()

with col_left:
    labels_info = {
        "Nhãn": ["CHAR", "PER", "WORK", "LOC", "TIME/DATE", "ORG"],
        "Ý nghĩa": ["Nhân vật trong tác phẩm", "Nhà văn, nhà thơ, nhân vật lịch sử,... có thật", "Tên tác phẩm", "Địa danh", "Thời gian, niên đại", "Tổ chức, triều đại, nhà xuất bản"]
    }
    html_legend_rows = ""
    for label, meaning in zip(labels_info["Nhãn"], labels_info["Ý nghĩa"]):
        color = tag_colors.get(label, "black")
        html_legend_rows += f"<tr><td style='color: {color}; font-weight: bold;'>{label}</td><td>{meaning}</td></tr>"

    st.markdown(f"""
        <div class="translucent-box">
            <div class="section-title">Bảng chú thích nhãn thực thể</div>
            <table class="custom-table">
                <thead><tr><th style="width:30%">Nhãn</th><th>Ý nghĩa</th></tr></thead>
                <tbody>{html_legend_rows}</tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("") 

    st.markdown("""
        <div style="
            background-color: #795548; 
            color: white; 
            padding: 8px 20px; 
            border-radius: 5px 5px 0 0; 
            font-weight: bold; 
            display: inline-block;
            font-family: sans-serif;
            margin-top: 10px;
        ">
            NHẬP NỘI DUNG VĂN BẢN
        </div>
    """, unsafe_allow_html=True)
    
    input_text = st.text_area("", height=200, placeholder="Ví dụ: Tác phẩm Lão Hạc của Nam Cao...", label_visibility="collapsed")
    
    if st.button("PHÂN TÍCH"):
        if not crf_model:
            st.error("Không tìm thấy Model.")
        elif not input_text.strip():
            st.warning("Vui lòng nhập nội dung.")
        else:
            with st.spinner("Đang xử lý..."):
                tokens = preprocess_text_for_prediction(input_text)
                if tokens:
                    features = get_features_for_prediction(tokens)
                    try:
                        predicted_tags = crf_model.predict_single(features)
                        st.session_state.prediction_result = (tokens, predicted_tags)
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                else:
                    st.error("Không có từ hợp lệ.")

with col_right:
    if st.session_state.prediction_result:
        tokens, predicted_tags = st.session_state.prediction_result
        
        html_visual = '<div style="line-height: 2.5;">'
        for token, tag in zip(tokens, predicted_tags):
            display_token = token.replace('_', ' ')
            if tag != 'O':
                color = tag_colors.get(tag, "#333")
                html_visual += f'<span style="background-color: {color}; color: white; padding: 5px 10px; border-radius: 6px; margin: 0 4px; display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.15); font-weight: bold;">{display_token} <small style="font-size: 0.7em; opacity: 0.9; margin-left: 3px;">{tag}</small></span>'
            else:
                html_visual += f'<span style="margin: 0 2px; color: #333; font-size: 1.05em;">{display_token}</span>'
        html_visual += '</div>'

        st.markdown(f"""
            <div class="translucent-box">
                <div class="section-title">Kết quả phân tích</div>
                {html_visual}
            </div>
        """, unsafe_allow_html=True)

        html_table_rows_result = ""
        for idx, (token, tag) in enumerate(zip(tokens, predicted_tags)):
            if tag != 'O':
                row_html = f'<tr style="background-color: rgba(255,255,255,0.6);"><td style="color:#666;">{idx+1}</td><td style="font-weight:bold; color:#333;">{token}</td><td style="font-weight:bold; color:#b8860b;">{tag}</td></tr>'
            else:
                row_html = f'<tr><td style="color:#999;">{idx+1}</td><td style="color:#555;">{token}</td><td style="color:#ccc;">{tag}</td></tr>'
            
            html_table_rows_result += row_html

        st.markdown(f"""
            <div class="translucent-box">
                <div class="section-title">Bảng chi tiết</div>
                <div style="max-height: 500px; overflow-y: auto;">
                    <table class="custom-table">
                        <thead>
                            <tr>
                                <th style="width: 10%">STT</th>
                                <th style="width: 50%">Từ</th>
                                <th style="width: 40%">Loại thực thể</th>
                            </tr>
                        </thead>
                        <tbody>
                            {html_table_rows_result}
                        </tbody>
                    </table>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="translucent-box" style="text-align: center; padding: 50px 20px; opacity: 0.7;">
                <p>Kết quả phân tích</p>
            </div>
        """, unsafe_allow_html=True)