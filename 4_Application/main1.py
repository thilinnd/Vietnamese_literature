import streamlit as st
import pickle
import re
import os
import pandas as pd
from pyvi import ViTokenizer

st.set_page_config(
    page_title="Bài toán NER - Nhận diện thực thể trong văn học Việt Nam",
    layout="wide"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&display=swap');

    /* 1. Nút bấm */
    div.stButton > button:first-child {
        background-color: #795548;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        width: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: #5d4037;
        color: white;
    }

    /* 2. Class chung cho các khung nền mờ (Translucent Box) */
    .translucent-box {
        background-color: rgba(255, 255, 255, 0.85); /* Độ trong suốt */
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        backdrop-filter: blur(5px); /* Hiệu ứng làm mờ ảnh nền phía sau */
    }

    /* 3. Style cho bảng HTML (Dùng chung cho Legend và Kết quả) */
    table.custom-table {
        width: 100%;
        border-collapse: collapse;
        font-family: sans-serif;
        font-size: 15px;
    }
    table.custom-table th {
        text-align: left;
        padding: 12px;
        border-bottom: 2px solid #795548; /* Đường kẻ màu nâu */
        color: #5d4037;
        font-weight: bold;
    }
    table.custom-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
        vertical-align: middle;
    }
    table.custom-table tr:last-child td {
        border-bottom: none;
    }

    /* 4. Tiêu đề section nhỏ */
    .section-title {
        color: #795548;
        font-family: 'Source Serif Pro', serif;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #795548;
        padding-left: 10px;
    }
    
    /* 5. Header to */
    .header-style {
        background-color: #795548; 
        color: white; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center; 
        font-family: 'Source Serif Pro', serif; 
        font-size: 50px; 
        font-weight: bold; 
        margin-bottom: 30px;
        white-space: nowrap; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        width: 100%;
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
    "CHAR": "#d32f2f",      
    "PER": "#7b1fa2",       
    "WORK": "#303f9f",      
    "LOC": "#00796b",       
    "TIME/DATE": "#f57f17", 
    "ORG": "#3e2723"        
}

SAVE_DIR = 'saved_models'

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
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word.has_underscore': '_' in word,
        }
        if i > 0:
            word1 = sent[i-1]
            features.update({'-1:word.lower()': word1.lower(), '-1:word.istitle()': word1.istitle()})
        else:
            features['BOS'] = True
        if i < len(sent)-1:
            word1 = sent[i+1]
            features.update({'+1:word.lower()': word1.lower(), '+1:word.istitle()': word1.istitle()})
        else:
            features['EOS'] = True
        sent_features.append(features)
    return sent_features


st.markdown('<div class="header-style">HỆ THỐNG NHẬN DIỆN THỰC THỂ VĂN HỌC VIỆT NAM</div>', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    labels_info = {
        "Nhãn": ["CHAR", "PER", "WORK", "LOC", "TIME/DATE", "ORG"],
        "Ý nghĩa": [
            "Nhân vật trong tác phẩm",
            "Nhà văn, nhà thơ, nhân vật lịch sử,... có thật",
            "Tên tác phẩm",
            "Địa danh",
            "Thời gian, niên đại",
            "Tổ chức, triều đại, nhà xuất bản"
        ]
    }
    
    html_legend_rows = ""
    for label, meaning in zip(labels_info["Nhãn"], labels_info["Ý nghĩa"]):
        color = tag_colors.get(label, "black")
        html_legend_rows += f"<tr><td style='color: {color}; font-weight: bold;'>{label}</td><td>{meaning}</td></tr>"

    st.markdown(f"""
        <div class="translucent-box">
            <div class="section-title">Bảng chú thích thực thể</div>
            <table class="custom-table">
                <thead><tr><th style="width:30%">Nhãn</th><th>Ý nghĩa</th></tr></thead>
                <tbody>{html_legend_rows}</tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    crf_model = load_crf_model()

    st.markdown("""
        <div style="background-color: #795548; color: white; padding: 8px 15px; border-radius: 5px 5px 0 0; font-weight: bold; display: inline-block;">
            NHẬP NỘI DUNG VĂN BẢN
        </div>
    """, unsafe_allow_html=True)
    
    input_text = st.text_area("", height=150, placeholder="Ví dụ: Tác phẩm Truyện Kiều của Nguyễn Du...", label_visibility="collapsed")

    if st.button("PHÂN TÍCH", type="primary"):
        if not crf_model:
            st.error("Chưa tìm thấy model. Vui lòng kiểm tra lại file model.")
        elif not input_text.strip():
            st.warning("Vui lòng nhập nội dung văn bản.")
        else:
            with st.spinner("Đang xử lý..."):
                tokens = preprocess_text_for_prediction(input_text)
                if not tokens:
                    st.error("Không tìm thấy từ hợp lệ.")
                else:
                    features = get_features_for_prediction(tokens)
                    try:
                        predicted_tags = crf_model.predict_single(features)
                        
                        html_visual = '<div style="line-height: 2.5;">'
                        for token, tag in zip(tokens, predicted_tags):
                            display_token = token.replace('_', ' ')
                            if tag != 'O':
                                color = tag_colors.get(tag, "#333")
                                html_visual += f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 6px; margin: 0 4px; display: inline-block; box-shadow: 0 2px 4px rgba(0,0,0,0.1); font-weight: bold;">{display_token} <small style="font-size: 0.7em; opacity: 0.9; margin-left: 2px;">{tag}</small></span>'
                            else:
                                html_visual += f'<span style="margin: 0 2px; color: #333;">{display_token}</span>'
                        html_visual += '</div>'

                        st.markdown(f"""
                            <div class="translucent-box" style="margin-top: 20px;">
                                <div class="section-title">Kết quả phân tích văn bản</div>
                                {html_visual}
                            </div>
                        """, unsafe_allow_html=True)

                        html_table_rows_result = ""
                        for idx, (token, tag) in enumerate(zip(tokens, predicted_tags)):
                            if tag != 'O':
                                color = tag_colors.get(tag, "#333")
                                row_html = f'<tr style="background-color: rgba(255,255,255,0.5);"><td style="color:#666;">{idx}</td><td style="font-weight:bold; color:#333;">{token}</td><td style="font-weight:bold; color:#b8860b;">{tag}</td></tr>'
                            else:
                                row_html = f'<tr><td style="color:#999;">{idx}</td><td style="color:#555;">{token}</td><td style="color:#bbb;">{tag}</td></tr>'
                            
                            html_table_rows_result += row_html

                        st.markdown(f"""
                            <div class="translucent-box">
                                <div class="section-title">Bảng chi tiết</div>
                                <div style="max-height: 400px; overflow-y: auto;">
                                    <table class="custom-table">
                                        <thead>
                                            <tr>
                                                <th style="width: 10%">#</th>
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

                    except Exception as e:
                        st.error(f"Đã xảy ra lỗi khi dự đoán: {e}")