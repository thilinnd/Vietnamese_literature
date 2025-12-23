# 🇻🇳 Vietnamese Literature Named Entity Recognition (NER)

Dự án xây dựng **hệ thống Nhận diện Thực thể Tên (NER)** chuyên biệt cho **Văn học Việt Nam hiện đại**.
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

Quá trình huấn luyện và đánh giá được thiết kế theo **nhiều tình huống thực nghiệm** nhằm phân tích toàn diện hiệu quả của các mô hình NER.

### **3.3.1. Tình huống 1: So sánh các mô hình (Model Comparison)**

Mục tiêu: So sánh hiệu năng giữa các mô hình Machine Learning và Deep Learning.

**Tiêu chí đánh giá**

* **F1-score** (chỉ số chính)
* **Thời gian huấn luyện**
* **Mức sử dụng bộ nhớ (RAM / GPU)**

Kết quả giúp lựa chọn mô hình tối ưu giữa độ chính xác và chi phí tính toán.


### **3.3.2. Tình huống 2: Ảnh hưởng của tăng cường dữ liệu (Data Augmentation)**

Mục tiêu: Đánh giá mức cải thiện hiệu quả mô hình khi áp dụng kỹ thuật tăng cường dữ liệu.

So sánh:

* Mô hình huấn luyện **trước khi tăng cường dữ liệu**
* Mô hình huấn luyện **sau khi tăng cường dữ liệu**

Chỉ số đánh giá chính: **F1-score**.


### **3.3.3. Tình huống 3: Phân tích lỗi (Error Analysis)**

Mục tiêu: Hiểu rõ các dạng lỗi phổ biến của mô hình NER.

**Định hướng phân tích**

* Phân tích **Ma trận nhầm lẫn (Confusion Matrix)**
* Xác định các cặp nhãn dễ bị nhầm lẫn

Kết quả giúp đề xuất hướng cải thiện mô hình và dữ liệu.


### **3.3.4. Ảnh hưởng của kích thước dữ liệu huấn luyện**

Mục tiêu: Phân tích mối quan hệ giữa kích thước tập huấn luyện và độ chính xác mô hình.

Thực nghiệm huấn luyện với các tỷ lệ dữ liệu:

* **33%** tập dữ liệu
* **66%** tập dữ liệu
* **100%** tập dữ liệu

Quan sát sự thay đổi của **F1-score** để đánh giá mức độ phụ thuộc của mô hình vào quy mô dữ liệu.

---

## Ứng dụng demo

* Thư mục `4_Application/`

* Nhóm nghiên cứu xây dựng giao diện web mang tên “Hệ thống nhận diện thực thể văn học Việt Nam” nhằm cho phép người dùng tương tác trực tiếp và kiểm chứng kết quả của mô hình NER đã huấn luyện trên dữ liệu văn bản văn học Việt Nam.

* Dựa trên kết quả đánh giá thực nghiệm (ma trận nhầm lẫn, bảng thống kê và biểu đồ hiệu năng), mô hình Conditional Random Field (CRF) đạt độ chính xác cao và ổn định nhất, do đó được lựa chọn làm mô hình lõi của hệ thống.

* Về kiến trúc, hệ thống được phát triển chủ yếu bằng Python; trong đó Streamlit được sử dụng để xây dựng giao diện web tương tác, kết hợp với HTML nhằm tăng tính trực quan và thân thiện với người dùng. Cách tiếp cận này giúp hệ thống dễ triển khai, phù hợp cho mục đích trình diễn và nghiên cứu.