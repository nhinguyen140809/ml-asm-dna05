# 🧬 Bài Tập Lớn Học Máy - Nhóm DNA05

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  [![Colab](https://img.shields.io/badge/Colab-Open-green.svg)](https://colab.research.google.com/)

---

<div align="center">
  <h2>🎯 Dự án: Bài Tập Lớn Học Máy - Nhóm DNA05</h2>
  <p>Pipeline xử lý dữ liệu bảng, văn bản và ảnh</p>
</div>

---

## 📚 Thông tin môn học

* **Môn học:** Học máy (Machine Learning)
* **Mã môn học:** CO3117
* **Học kỳ:** 251, Năm học 2025 - 2026
* **Giảng viên hướng dẫn:** TS. Lê Thành Sách

---

## 👥 Thành viên nhóm

**Nhóm DNA05**

| MSSV    | Họ và tên           | Email                                                                 |
| ------- | ------------------- | --------------------------------------------------------------------- |
| 2310510 | Phạm Khánh Duy      | [duy.pham84210@hcmut.edu.vn](mailto:duy.pham84210@hcmut.edu.vn)       |
| 2310167 | Tăng Hồng Ái        | [ai.tangmeo200922@hcmut.edu.vn](mailto:ai.tangmeo200922@hcmut.edu.vn) |
| 2312506 | Nguyễn Trần Yến Nhi | [nhi.nguyen140809@hcmut.edu.vn](mailto:nhi.nguyen140809@hcmut.edu.vn) |

---

## 🌟 Tổng quan dự án

Khám phá **4 dự án lớn** trong môn học Học Máy với notebook trực tuyến và báo cáo:

* Trang tổng quan: [GitHub Pages](https://nhinguyen140809.github.io/ml-asm-dna05/index.html)

| Dự án       | Loại dữ liệu        | Trạng thái     | Trang GitHub                                                             | Notebook Colab                                                                                    |
| ----------- | ------------------- | -------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **BTL1**    | Dữ liệu bảng        | ✅ Hoàn thành   | [Tabular Data](https://nhinguyen140809.github.io/ml-asm-dna05/btl1.html) | [Mở Colab](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing) |
| **BTL2**    | Dữ liệu văn bản     | ✅ Hoàn thành   | [Text Data](https://nhinguyen140809.github.io/ml-asm-dna05/btl2.html)    | [Mở Colab](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing) |
| **BTL3**    | Dữ liệu ảnh         | 🕐 Sẽ cập nhật | 🕐 Sẽ cập nhật                                                           | 🕐 Sẽ cập nhật                                                                                    |
| **Mở rộng** | Hidden Markov Model | 🕐 Sẽ cập nhật | 🕐 Sẽ cập nhật                                                           | 🕐 Sẽ cập nhật                                                                                    |

---

## 🎯 Mục tiêu

* Áp dụng **pipeline học máy truyền thống**: tiền xử lý, trích xuất đặc trưng, huấn luyện, đánh giá.
* Thực hành xử lý **dữ liệu bảng, văn bản, ảnh**.
* Phân tích và so sánh hiệu quả mô hình qua các chỉ số đo lường.
* Rèn luyện kỹ năng lập trình, thử nghiệm và báo cáo khoa học.

---

## 🔗 Cấu trúc repository

* **GitHub Repository:** [ml-asm-dna05](https://github.com/nhinguyen140809/ml-asm-dna05)

```
ml-asm-dna05/
├─ notebooks/                      # Notebook Google Colab
│   ├─ DNA05_BTL1.ipynb
│   └─ DNA05_BTL2.ipynb
├─ reports/                        # Báo cáo PDF
│   ├─ DNA05_BTL1_Report.pdf
│   └─ DNA05_BTL2_Report.pdf
├─ features/                       # File đặc trưng
│   ├─ BTL1
│   ├─ BTL2
│   ├─ BTL3
│   └─ BTL_MR
├─ modules/                        # Module hỗ trợ
│   ├─ ml_pipeline.py
│   └─ EAssignment/hmm.py
├─ docs/                           # Biên bản và hình ảnh cuộc họp
│   └─ meeting_evidence/{BTL1,BTL2,BTL3,BTL_MR}
├─ README.md
└─ requirements.txt
```

---

## 🚀 Cách sử dụng

### 1️⃣ Cài đặt Local

```bash
# Clone repository
git clone https://github.com/nhinguyen140809/ml-asm-dna05.git
cd ml-asm-dna05

# (Tùy chọn) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux / Mac
# venv\Scripts\activate   # Windows

# Cập nhật pip
python -m pip install --upgrade pip

# Cài đặt dependencies
pip install -r requirements.txt

# Khởi chạy Jupyter Notebook
jupyter notebook
```

Mở notebook cụ thể (ví dụ `DNA05_BTL1.ipynb`) và chạy.

### 2️⃣ Google Colab

* Mở trực tiếp các notebook:

  * [BTL1](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)
  * [BTL2](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing)

Nhấn **Run all** để thực hiện pipeline trực tuyến mà không cần cài đặt.

---

## 🏥 BTL1 - Dự đoán bệnh tim mạch

* **Dataset:** [Heart Disease – Oktay Rdeki](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) (~10.000 mẫu, bảng dữ liệu)
* **Báo cáo PDF:** [BTL1 Report](https://github.com/nhinguyen140809/ml-asm-dna05/blob/main/reports/DNA05_BTL1_Report.pdf)
* **Notebook:** [BTL1 Colab](https://colab.research.google.com/drive/1Bz4B_MAlvOQ6Acb93SF8WxtnKEEAdTf7?usp=sharing)

## 📚 BTL2 - Phân loại câu hỏi học tập

* **Dataset:** [Students Questions – Kaggle](https://www.kaggle.com/datasets/mrutyunjaybiswal/iitjee-neet-aims-students-questions-data) (~121.679 mẫu)
* **Báo cáo PDF:** [BTL2 Report](https://github.com/nhinguyen140809/ml-asm-dna05/blob/main/reports/DNA05_BTL2_Report.pdf)
* **Notebook:** [BTL2 Colab](https://colab.research.google.com/drive/1V6W7FOQARQ1WcBAhl4ELyyc-sd0L5z54?usp=sharing)

## 🕐 BTL3 & Mở rộng

* Sẽ cập nhật sau...

---

## 📑 Tài liệu tham khảo

* [Heart Disease Prediction Kaggle Notebook](https://www.kaggle.com/code/hossainhedayati/heart-disease-prediction-with-83-8-accuracy)
