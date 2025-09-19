# Machine Learning Assignment - DNA05 Team

## Thông tin môn học

- Môn học: Học máy (Machine Learning) 
- Mã môn học: CO3117.
- Học kỳ: 251, năm học 2025 - 2026.
- Giảng viên hướng dẫn: TS. Lê Thành Sách.

## Thành viên nhóm

Nhóm: DNA05

|MSSV| Họ và tên| Email|
|----|----------|------|
|2310510| Phạm Khánh Duy| duy.pham84210@hcmut.edu.vn|
|2310167| Tăng Hồng Ái| ai.tangmeo200922@hcmut.edu.vn|
|2312506| Nguyễn Trần Yến Nhi| nhi.nguyen140809@hcmut.edu.vn|

## Mục tiêu của bài tập lớn

Bài tập lớn được thiết kế với các mục tiêu cụ thể sau:

- Hiểu và áp dụng được quy trình **pipeline học máy truyền thống**, bao gồm: tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện và đánh giá mô hình.
- Rèn luyện kỹ năng triển khai mô hình học máy trên các loại dữ liệu khác nhau: bảng, văn bản, và ảnh.
- Phát triển khả năng phân tích, so sánh, và đánh giá hiệu quả của các mô hình học máy thông qua các chỉ số đo lường (Accuracy, Precision, Recall, F1-score).
- Rèn luyện kỹ năng lập trình, thử nghiệm, và tổ chức báo cáo khoa học.

## Đường link và cấu trúc thư mục của dự án

GitHub: [https://github.com/nhinguyen140809/ml-asm-dna05](https://github.com/nhinguyen140809/ml-asm-dna05)

```
ml-asm-dna05/
|-- notebooks/      # Notebook Google Colab
|   |-- DNA05_BTL1.ipynb                 
|-- reports/        # Báo cáo
|   |-- DNA05_BTL1_Report.pdf
|-- features/       # File đặc trưng được trích xuất
|-- modules/        # Các module hỗ trợ
|-- README.md       # Tài liệu hướng dẫn
```

## Usage

Để sử dụng repository và chạy các notebook:
### Local
1. **Clone repository về máy:**

```bash
git clone https://github.com/nhinguyen140809/ml-asm-dna05.git
cd ml-asm-dna05
```

2. **(Tuỳ chọn) Tạo môi trường ảo để quản lý thư viện:**

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows
```

3. Cập nhật pip (nếu cần)

```bash
python -m pip install --upgrade pip
```

1. **Cài đặt các dependencies cần thiết:**

```bash
pip install -r requirements.txt
```

4. **Launch notebook:**

```bash
jupyter notebook
```

1. **Mở và chạy notebook cụ thể**

Ví dụ, `notebook/DNA05-BTL1.ipynb`.

### Google Colab

- Click vào link Colab notebook: [DNA05-BTL1](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)

- Chạy tất cả các cell để thực hiện pipeline trực tuyến mà không cần cài đặt thêm trên máy. Có thể dùng **Run all** để chạy toàn bộ cell tự động.

## Bài tập lớn 1 - Dự đoán bệnh tim mạch (Heart Disease Prediction)

Bài tập lớn sử dụng bộ dữ liệu `heart_disease.csv` từ Kaggle, dạng bảng với ~10.000 mẫu, chứa các thông tin về sức khỏe và các chỉ số sinh học. Nhóm triển khai pipeline học máy đầy đủ, bao gồm:

1. Phân tích dữ liệu (EDA)  
2. Tiền xử lý dữ liệu (missing values, scaling, xử lý mất cân bằng)  
3. Trích xuất & giảm chiều dữ liệu bằng PCA  
4. Huấn luyện nhiều mô hình phân loại (Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Decision Tree)  
5. Đánh giá mô hình bằng các metric: Accuracy, Precision, Recall, F1-score  
6. Trực quan hóa kết quả và Confusion Matrix  

### Dataset

- [Heart Disease Dataset – Oktay Rdeki](https://www.kaggle.com/datasets/oktayrdeki/heart-disease)  

### Report và Notebook

- Report PDF: [DNA05-BTL1-Report](https://github.com/nhinguyen140809/ml-asm-dna05/blob/main/reports/DNA05_BTL1_Report.pdf)  
- Colab Notebook: [DNA05-BTL1](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)  

## Bài tập lớn 2 (Cập nhật sau)

## Bài tập lớn 3 (Cập nhật sau)

## Bài tập lớn Mở rộng (Cập nhật sau)

## Tài liệu tham khảo

- [Heart Disease Prediction with 83.8% Accuracy - Kaggle Notebook](https://www.kaggle.com/code/hossainhedayati/heart-disease-prediction-with-83-8-accuracy)
