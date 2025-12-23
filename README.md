# 🇻🇳 Vietnamese Literature Named Entity Recognition (NER)

Dự án xây dựng **hệ thống Nhận diện Thực thể Tên (NER)** chuyên biệt cho **miền văn học Việt Nam hiện đại**.
Mô hình được huấn luyện trên dữ liệu gồm **59 tác phẩm** và **51 tác giả**, đồng thời **so sánh hiệu quả giữa các mô hình Machine Learning cổ điển và Deep Learning**.

---

## Cấu trúc dự án

```text
├── 1_Crawling/          # Scripts thu thập dữ liệu từ Wikipedia
├── 2_Preprocess/        # Làm sạch dữ liệu, chuẩn hóa & chuyển BIO
├── 3_Model/             # Huấn luyện và đánh giá các mô hình NER
├── 4_Application/       # Giao diện demo (GUI) sử dụng model tốt nhất
├── Data/                # (KHÔNG có sẵn trên GitHub) – tải từ Google Drive
├── saved_models/        # (KHÔNG có sẵn trên GitHub) – tải từ Google Drive
├── requirements.txt     # Danh sách thư viện cần thiết
└── README.md            # Tài liệu hướng dẫn
```

> **Lưu ý quan trọng**
>
> * Thư mục **`Data/`** và **`saved_models/`** **không chứa dữ liệu trên GitHub**
> * Người dùng cần **tải thủ công từ Google Drive** (link bên dưới)

---

## Cài đặt môi trường

```bash
pip install -r requirements.txt
```

Khuyến nghị sử dụng **Python ≥ 3.12**.

---

## 🔄 Workflow sử dụng hệ thống

### **Bước 1: Thu thập dữ liệu (Data Crawling)**

Dữ liệu được thu thập tự động từ **Wikipedia**, tập trung vào các bài viết về:

* 59 tác phẩm văn học Việt Nam hiện đại
* 51 tác giả tiêu biểu

**Input**

* Danh sách URL về tác phẩm và tác giả được lưu trong Data/link_href.csv

**Thực hiện**

* Chạy các script trong thư mục `1_Crawling/`
* Văn bản được trích xuất và tách câu (sentence segmentation)

**Output**

* Các file văn bản thô đã được tách câu được lưu trong Data/final_dataset.json

---

### **Bước 2: Tiền xử lý & Gán nhãn (Preprocessing & Labeling)**

Đây là bước tạo **ground truth** cho bài toán NER.

**2.1. Làm sạch dữ liệu**

* Chạy script trong `2_Preprocess/`
* Loại bỏ ký tự đặc biệt, chuẩn hóa unicode, định dạng văn bản

**2.2. Gán nhãn thủ công**

* Dữ liệu sạch được đưa lên **Label Studio**
* Gán nhãn các thực thể (nhân vật, tác phẩm, tác giả, …)
* Dữ liệu được tải về từ **Label Studio** được lưu trong Data/train_final.json

**2.3. Chuyển đổi định dạng**

* Chuyển file JSON từ Label Studio sang định dạng **BIO (Begin – Inside – Outside)**
* Các câu **không chứa thực thể sẽ bị loại bỏ** để giảm nhiễu

**Output**

* File JSON (Data/train_bio.json) tổng hợp ở định dạng BIO, sẵn sàng cho huấn luyện mô hình

---

## Tải dữ liệu & model có sẵn (Khuyến nghị)

Người dùng **có thể bỏ qua toàn bộ bước chuẩn bị dữ liệu** bằng cách tải trực tiếp dữ liệu đã xử lý:

🔗 **Google Drive**
[https://drive.google.com/drive/folders/1LLXzent3J1pMUhYszDa6cWiXVdpOktkx](https://drive.google.com/drive/folders/1LLXzent3J1pMUhYszDa6cWiXVdpOktkx)
[https://drive.google.com/drive/folders/1FfrHfUeSdFUTIBU8DVzZFyBh4QO7VJzy](https://drive.google.com/drive/folders/1FfrHfUeSdFUTIBU8DVzZFyBh4QO7VJzy)

Sau khi tải:

* Giải nén và đặt đúng cấu trúc:

  * `Data/`
  * `saved_models/`

---

## Huấn luyện & đánh giá mô hình

* Thực hiện trong thư mục `3_Model/`

---

## Ứng dụng demo

* Thư mục `4_Application/`
* Giao diện demo sử dụng **mô hình có hiệu năng tốt nhất**
* Cho phép nhập văn bản và hiển thị kết quả NER trực quan
