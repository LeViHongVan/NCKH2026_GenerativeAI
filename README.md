# 🔬 NCKH 2026: Xây Dựng 3 Mô Hình Generative AI Sinh Mẫu Mã Độc Nhân Tạo (Synthetic Malware Samples)

Dự án này hiện thực hóa toàn diện kiến trúc nghiên cứu và hệ thống thực nghiệm từ bài báo khoa học xuất bản trên **IEEE Access 2025**:
> **"Generating Synthetic Malware Samples Using Generative AI"**  
> *Tiffany Bao, Kylie Trousil, Quang Duy Tran, Fabio Di Troia, Younghee Park.*

Cùng các định hướng nghiên cứu từ công trình khảo sát của nhóm tác giả Học viện Kỹ thuật Mật mã (ACTVN) - TS. Bùi Thu Lâm et al. (JCSC 2025): *"A Survey on Generative Adversarial Networks for Malware Analysis"*.

---

## 📌 Tổng Quan Hệ Thống

Dự án giải quyết bài toán **mất cân bằng dữ liệu nghiêm trọng** trong phân loại mã độc bằng cách sinh các vector đặc trưng mã độc nhân tạo (**Synthetic Malware Embeddings**) chất lượng cao:
1. **Tiền xử lý**: File nhị phân/ASM $\rightarrow$ Trích xuất chuỗi Opcode (`mov`, `push`, `call`...) $\rightarrow$ Huấn luyện **Word2Vec** $\rightarrow$ Average Pooling ra vector **104 chiều** đại diện cho từng file $\rightarrow$ Chuẩn hóa về $[-1, 1]$.
2. **3 Mô hình GenAI**:
   - **Vanilla Dense GAN**: Mạng đối kháng cơ bản (Generator Dense 200-400-800-104 + BatchNorm + Tanh; Discriminator Dense 400-200-1 + LeakyReLU).
   - **Conv1D WGAN-GP**: Mạng Wasserstein GAN kết hợp Gradient Penalty ($\lambda=10$) và tích chập 1D để học đặc trưng chuỗi opcode.
   - **1D U-Net Diffusion Model**: Mô hình khuếch tán khử nhiễu (DDPM 1D) với **cải tiến rút ngắn quá trình thêm nhiễu đến $T/2 = 500$ bước** (thay vì $T=1000$) để bảo toàn cấu trúc tín hiệu 1D.
3. **Hệ thống đánh giá 6 tiêu chí khoa học**:
   - `Metric 1`: Binary Classification (Real vs Synthetic: Random Forest, SVM, MLP).
   - `Metric 2`: t-SNE Visualization (Chiếu không gian 104D về 2D, quan sát hình thái cụm manifold).
   - `Metric 3`: Synthetic-Only Training Evaluation (Train bộ phân loại chỉ trên Fake, Test trên Real).
   - `Metric 4`: Fidelity Score (Train bộ phân loại trên Real, Test trên Fake).
   - `Metric 5`: Threshold-Based Augmentation (Tăng cường dữ liệu có điều kiện threshold $t=0.95$ trên Random Forest).
   - `Metric 6`: Robustness / Cosine Similarity (Đo độ lệch trung vị cosine similarity so với baseline).

---

## 📂 Cấu Trúc Mã Nguồn

```text
NCKH2026_GenerativeAI/
│
├── data/                      # Dữ liệu
│   ├── processed/             # real_embeddings.npy, real_labels.npy
│   └── synthetic/             # Mẫu sinh từ GAN, WGAN-GP, Diffusion
│
├── src/
│   ├── models/                # 3 Mô hình sinh
│   │   ├── gan.py             # Vanilla Dense GAN
│   │   ├── wgan_gp.py         # Conv1D WGAN-GP (Lipschitz Constraint)
│   │   └── diffusion.py       # 1D U-Net Diffusion (T/2 schedule)
│   │
│   ├── preprocessing/         # Tiền xử lý dữ liệu
│   │   ├── extract_opcode.py  # Trích xuất opcode từ ASM / binary
│   │   └── build_embedding.py # Huấn luyện Word2Vec & Average Pooling 104D
│   │
│   ├── training/
│   │   └── train_models.py    # Vòng lặp huấn luyện từng mô hình
│   │
│   ├── evaluation/            # Hệ thống 6 tiêu chí đánh giá
│   │   ├── metrics.py         # Binary, Fidelity, Synthetic-Only, Augmentation, Cosine
│   │   └── visualize.py       # t-SNE 2D plot & Augmentation Delta chart
│   │
│   └── data_generator.py      # Sinh dữ liệu Benchmark mô phỏng 5 họ malware
│
├── notebooks/
│   └── 01_full_pipeline_demo.ipynb # Notebook chạy thực nghiệm trên Colab / Jupyter
│
├── results/                   # Lưu kết quả biểu đồ & file JSON tổng hợp
├── run_demo_pipeline.py       # SCRIPT CHẠY TOÀN BỘ PIPELINE ĐẦU-CUỐI
├── requirements.txt           # Danh sách thư viện Python
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Ngay

### Cách 1: Chạy trực tiếp trên máy cục bộ (Windows)

1. **Cài đặt Python 3.10 hoặc 3.11** (nếu máy chưa có):
   - Mở PowerShell và chạy:
     ```powershell
     winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements
     ```
   - Hoặc tải bộ cài từ: [https://www.python.org/downloads/](https://www.python.org/downloads/) (nhớ tích chọn **"Add Python to PATH"**).

2. **Cài đặt các thư viện cần thiết**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ngay Pipeline thực nghiệm đầu-cuối**:
   ```bash
   python run_demo_pipeline.py
   ```
   *Quá trình này sẽ tự động sinh dữ liệu giả lập chuẩn hóa 104D, huấn luyện cả 3 mô hình, sinh mẫu và xuất báo cáo + 2 biểu đồ phân tích vào thư mục `results/`.*

---

### Cách 2: Chạy trên Google Colab / Kaggle (Có GPU T4 miễn phí)

Nếu bạn muốn huấn luyện nhanh bằng GPU:
1. Tải toàn bộ thư mục code lên Google Drive hoặc GitHub.
2. Mở file [01_full_pipeline_demo.ipynb](file:///e:/NCKH2026_GenerativeAI/notebooks/01_full_pipeline_demo.ipynb) trên Google Colab.
3. Chọn Runtime $\rightarrow$ **Change runtime type** $\rightarrow$ Chọn **T4 GPU**.
4. Bấm **Run All** để xem trực tiếp các biểu đồ t-SNE và bảng điểm F1!

---

## 📊 Kết Quả Thực Nghiệm Dự Kiến Trong Thư Mục `results/`

Sau khi chạy `run_demo_pipeline.py`, các file kết quả sẽ được tạo:
1. `results/experiment_summary.json`: Điểm số chi tiết của cả 6 metric.
2. `results/tsne_comparison.png`: Biểu đồ t-SNE 2D thể hiện sự khác biệt:
   - **Diffusion**: Bám sát cấu trúc phân bố dạng vòng (ring) của mẫu thật.
   - **WGAN-GP & GAN**: Có xu hướng tụ về trung tâm (thể hiện hiện tượng mode averaging/collapse).
3. `results/augmentation_delta.png`: Biểu đồ cột thể hiện mức cải thiện F1-score sau khi tăng cường dữ liệu với threshold $t=0.95$.
