# 🧬 Machine Learning Assignment - DNA05 Team

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Colab](https://img.shields.io/badge/Colab-Open-green.svg)](https://colab.research.google.com/)

---

## 📚 Thông tin môn học

* **Môn học:** Học máy (Machine Learning)
* **Mã môn học:** CO3117
* **Học kỳ:** 251, năm học 2025 - 2026
* **Giảng viên hướng dẫn:** TS. Lê Thành Sách

---

## 👥 Thành viên nhóm

**Nhóm: DNA05**

| MSSV    | Họ và tên           | Email                                                                 |
| ------- | ------------------- | --------------------------------------------------------------------- |
| 2310510 | Phạm Khánh Duy      | [duy.pham84210@hcmut.edu.vn](mailto:duy.pham84210@hcmut.edu.vn)       |
| 2310167 | Tăng Hồng Ái        | [ai.tangmeo200922@hcmut.edu.vn](mailto:ai.tangmeo200922@hcmut.edu.vn) |
| 2312506 | Nguyễn Trần Yến Nhi | [nhi.nguyen140809@hcmut.edu.vn](mailto:nhi.nguyen140809@hcmut.edu.vn) |

---

## 📖 Tổng quan về dự án

Trong khuôn khổ môn học **Học Máy – CO3117**, có 4 dự án:

* Trang chung cho toàn dự án: [Github Page](https://nhinguyen140809.github.io/ml-asm-dna05/index.html)

| Dự án       | Nội dung            | Trạng thái      | Trang Github                                                             | Colab                                                                                                  |
| ----------- | ------------------- | --------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| **BTL1**    | Tabular Data        | ✅ Hoàn thành    | [Tabular Data](https://nhinguyen140809.github.io/ml-asm-dna05/btl1.html) | [Open In Colab](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing) |
| **BTL2**    | Text Data           | ✅ Hoàn thành    | [Text Data](https://nhinguyen140809.github.io/ml-asm-dna05/btl2.html)    | [Open In Colab](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing) |
| **BTL3**    | Image Data          | 🕐 Cập nhật sau | 🕐 Cập nhật sau                                                          | 🕐 Cập nhật sau                                                                                        |
| **Mở rộng** | Hidden Markov Model | 🕐 Cập nhật sau | 🕐 Cập nhật sau                                                          | 🕐 Cập nhật sau                                                                                        |

---

## 🎯 Mục tiêu của bài tập lớn

Bài tập lớn nhằm giúp sinh viên:

* Hiểu và áp dụng **pipeline học máy truyền thống**: tiền xử lý, trích xuất đặc trưng, huấn luyện và đánh giá mô hình.
* Rèn luyện kỹ năng triển khai mô hình trên các loại dữ liệu: bảng, văn bản, ảnh.
* Phát triển khả năng **phân tích và so sánh hiệu quả** mô hình qua các chỉ số đo lường.
* Rèn luyện kỹ năng lập trình, thử nghiệm và tổ chức báo cáo khoa học.

---

## 🔗 Đường link và cấu trúc thư mục

* **GitHub Repository:** [ml-asm-dna05](https://github.com/nhinguyen140809/ml-asm-dna05)

```
ml-asm-dna05/
|-- notebooks/                                # Notebook Google Colab
|   |-- DNA05_BTL1.ipynb
|   |-- DNA05_BTL2.ipynb
|-- reports/                                  # Báo cáo PDF
|   |-- DNA05_BTL1_Report.pdf
|   |-- DNA05_BTL2_Report.pdf
|-- features/                                 # File đặc trưng
|   |-- BTL1
|   |-- BTL2
|   |-- BTL3
|   |-- BTL_MR
|-- modules/                                  # Các module hỗ trợ
|   |-- ml_pipeline.py
|   |-- EAssignment/
|       |-- hmm.py
|-- docs/
|   |-- meeting_evidence                      # Biên bản và hình ảnh cuộc họp
|      |-- BTL1
|      |-- BTL2
|      |-- BTL3
|      |-- BTL_MR
|-- README.md                                 # Tài liệu hướng dẫn
|-- requirements.txt                          # Thư viện cần cài
```

---

## 🚀 Usage

### 1️⃣ Local

1. **Clone repository:**

```bash
git clone https://github.com/nhinguyen140809/ml-asm-dna05.git
cd ml-asm-dna05
```

2. **(Tuỳ chọn) Tạo môi trường ảo:**

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường
source venv/bin/activate   # Linux / Mac
# venv\Scripts\activate    # Windows
```

3. **Cập nhật pip (nếu cần):**

```bash
python -m pip install --upgrade pip
```

4. **Cài đặt dependencies:**

```bash
pip install -r requirements.txt
```

5. **Launch notebook:**

```bash
jupyter notebook
```

6. **Mở notebook cụ thể** (ví dụ `DNA05_BTL1.ipynb`) và chạy.

### 2️⃣ Google Colab

* Mở trực tiếp các notebook:

  * [DNA05-BTL1](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)
  * [DNA05-BTL2](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing)

* Chạy tất cả cell (**Run all**) để thực hiện pipeline trực tuyến.

---

## 🏥 Bài tập lớn 1 - Dự đoán bệnh tim mạch (Heart Disease Prediction)

* **Dataset:** [Heart Disease Dataset – Oktay Rdeki](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) (~10.000 mẫu, dạng bảng)
* **Report PDF:** [DNA05-BTL1-Report](https://github.com/nhinguyen140809/ml-asm-dna05/blob/main/reports/DNA05_BTL1_Report.pdf)
* **Colab Notebook:** [DNA05-BTL1](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)

---

## 📚 Bài tập lớn 2 - Phân loại câu hỏi học tập (Student Questions Classification)

* **Dataset:** [Students Questions Dataset – Kaggle](https://www.kaggle.com/datasets/mrutyunjaybiswal/iitjee-neet-aims-students-questions-data) (121.679 mẫu)
* **Report PDF:** [DNA05-BTL2-Report](https://github.com/nhinguyen140809/ml-asm-dna05/blob/main/reports/DNA05_BTL2_Report.pdf)
* **Colab Notebook:** [DNA05-BTL2](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing)

---

## 🕐 Bài tập lớn 3 & Mở rộng

* Cập nhật sau.

---

## 📑 Tài liệu tham khảo

* [Heart Disease Prediction with 83.8% Accuracy - Kaggle Notebook](https://www.kaggle.com/code/hossainhedayati/heart-disease-prediction-with-83-8-accuracy)
