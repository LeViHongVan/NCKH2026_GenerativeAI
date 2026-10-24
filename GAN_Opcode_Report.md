# Báo Cáo: Ứng dụng GAN và Kỹ thuật Nhúng Opcode (Word2Vec)

Báo cáo này trình bày chi tiết về kiến trúc cơ bản của mạng GAN (Generative Adversarial Network), các nguồn dữ liệu phổ biến, và một quy trình (demo) để trích xuất opcode từ file thực thi (binary) bằng `objdump` sau đó nhúng (embedding) chúng thông qua thuật toán Word2Vec.

---

## 1. Kiến thức cơ bản về mạng GAN (Generative Adversarial Networks)

Mạng GAN là một lớp mô hình học máy được giới thiệu bởi Ian Goodfellow vào năm 2014, chuyên dùng để sinh ra dữ liệu mới có phân phối giống với dữ liệu gốc (dữ liệu huấn luyện). GAN hoạt động dựa trên lý thuyết trò chơi (Game Theory), với hai mạng neural cạnh tranh nhau.

### Hai thành phần cốt lõi:

*   **Generator (Bộ sinh - $G$)**: 
    *   **Nhiệm vụ**: Tạo ra dữ liệu "giả" (fake data) sao cho giống dữ liệu "thật" (real data) nhất có thể để đánh lừa Discriminator.
    *   **Đầu vào**: Một vector nhiễu ngẫu nhiên (latent vector / noise, thường ký hiệu là $z$).
    *   **Đầu ra**: Dữ liệu được sinh ra $G(z)$ (ví dụ: hình ảnh, chuỗi opcode, văn bản...).
*   **Discriminator (Bộ phân biệt - $D$)**:
    *   **Nhiệm vụ**: Phân loại xem dữ liệu đầu vào là "thật" (lấy từ tập dữ liệu huấn luyện) hay "giả" (do Generator tạo ra).
    *   **Đầu vào**: Dữ liệu $x$ (có thể là $x_{real}$ từ dataset hoặc $x_{fake} = G(z)$ từ Generator).
    *   **Đầu ra**: Xác suất $D(x)$ nằm trong khoảng $[0, 1]$, thể hiện niềm tin của Discriminator rằng đầu vào là dữ liệu thật (1 là thật, 0 là giả).

Quá trình huấn luyện diễn ra song song: $D$ ngày càng giỏi trong việc phát hiện đồ giả, còn $G$ ngày càng giỏi trong việc tạo ra đồ giả chân thực hơn.

### Hàm Loss Function (Hàm mất mát Minimax)

Hàm mục tiêu tiêu chuẩn của GAN là một bài toán tối ưu hóa Minimax (Tối thiểu hóa - Tối đa hóa):

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

**Ý nghĩa của hàm Loss:**
1.  **Đối với Discriminator ($D$)**: Muốn **tối đa hóa** hàm $V(D,G)$. 
    *   Nó muốn $D(x)$ (dự đoán dữ liệu thật) càng gần 1 càng tốt ($\log(1) = 0$).
    *   Nó muốn $D(G(z))$ (dự đoán dữ liệu giả) càng gần 0 càng tốt ($\log(1-0) = \log(1) = 0$).
2.  **Đối với Generator ($G$)**: Muốn **tối thiểu hóa** hàm $V(D,G)$.
    *   Nó không thể tác động vào phần dữ liệu thật, nên nó chỉ tập trung vào phần sau: tối thiểu hóa $\log(1 - D(G(z)))$. Điều này đồng nghĩa với việc nó muốn $D(G(z))$ tiến tới 1 (khiến $D$ tưởng dữ liệu giả là thật).

> [!NOTE] Các biến thể
> Trong thực tế, hàm loss gốc thường gặp vấn đề "Vanishing Gradient" (mất mát gradient). Các kiến trúc mới hơn như WGAN (Wasserstein GAN) sử dụng hàm khoảng cách Earth Mover để huấn luyện ổn định hơn.

---

## 2. Bộ dữ liệu (Dataset) lấy ở đâu?

Tùy thuộc vào mục tiêu bài toán (sinh ảnh, văn bản, hay opcode của malware), bạn có thể lấy dữ liệu từ các nguồn sau:

### Dành cho bài toán Mã Độc (Malware) / Phân tích Opcode
Nếu mục tiêu của bạn là sinh ra các biến thể mã độc mới (để test hệ thống phòng thủ) hoặc trích xuất opcode, các nguồn sau là tốt nhất:
1.  **MalwareBazaar (Abuse.ch)**: Nguồn chia sẻ mẫu mã độc cập nhật từng ngày, cho phép tải xuống miễn phí qua API. Rất phù hợp để trích xuất opcode từ malware hiện đại.
2.  **VirusShare (virusshare.com)**: Một trong những kho lưu trữ mã độc lớn nhất thế giới, cung cấp các file torrent chứa hàng GB/TB mẫu mã độc. (Cần đăng ký/yêu cầu quyền truy cập).
3.  **Kaggle - Microsoft Malware Classification Challenge (BIG 2015)**: 
    *   Link: `kaggle.com/c/malware-classification`
    *   Đặc biệt hữu ích vì Microsoft đã cung cấp sẵn các file `.asm` (chứa các lệnh assembly/opcode) và `.bytes` của mã độc, bạn không cần phải tự chạy `objdump` hay dịch ngược nữa.
4.  **VX Underground (vx-underground.org)**: Nguồn lưu trữ malware source code, paper, và samples khổng lồ cho mục đích nghiên cứu (cần cẩn thận khi sử dụng).

### Dành cho bài toán sinh ảnh/học GAN cơ bản
Nếu bạn mới bắt đầu học GAN, nên chạy thử trên các bộ dữ liệu ảnh (đã được làm sạch):
*   **MNIST / Fashion MNIST**: Tập dữ liệu chữ số/quần áo viết tay.
*   **CIFAR-10**: Tập dữ liệu 10 lớp hình ảnh.
*   **CelebA**: Tập dữ liệu khuôn mặt người thật (dùng nhiều cho Deepfake/GAN).
*   Có sẵn trong `keras.datasets` hoặc `torchvision.datasets`.

---

## 3. Demo: Sinh ra opcode bằng `objdump` và Word2Vec (Vec2word)

Bài toán: Trích xuất các chuỗi lệnh (Opcodes - ví dụ: `mov`, `push`, `call`) từ một file thực thi (Binary file) bằng công cụ `objdump`, sau đó dùng thuật toán `Word2Vec` (của thư viện Gensim) để biểu diễn các opcode thành các vector không gian (Embeddings) để đưa vào mạng Neural/GAN.

### Bước 1: Trích xuất Opcode bằng `objdump`

Giả sử bạn có một file thực thi (VD: `sample.exe` hoặc `sample.elf`). Mở terminal (Linux/WSL) và chạy:

```bash
# Lệnh objdump -d dùng để Disassemble file thực thi.
# Dùng awk để cắt lấy chính xác cột chứa opcode (ví dụ: mov, push, sub...)
objdump -d sample.exe | grep -Eo '^\s*[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s*[a-z]+' | awk '{print $NF}' > opcodes.txt
```

Hoặc dùng một Script Python ngắn để tự động hóa việc gọi `objdump` và lọc:

```python
import subprocess
import re

def extract_opcodes(binary_path):
    opcodes = []
    try:
        # Gọi objdump
        result = subprocess.run(['objdump', '-d', binary_path], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        lines = result.stdout.split('\n')
        
        # Regex để tìm dòng chứa assembly instruction
        # Ví dụ: " 4004a4:	55                   	push   rbp"
        asm_pattern = re.compile(r'\s*[0-9a-f]+:\s+([0-9a-f]{2}\s+)+\s*([a-z]+)')
        
        for line in lines:
            match = asm_pattern.search(line)
            if match:
                opcode = match.group(2) # Lấy chữ (push, mov, call...)
                opcodes.append(opcode)
    except Exception as e:
        print(f"Lỗi: {e}")
        
    return opcodes

# Demo chạy thử
my_opcodes = extract_opcodes("sample.exe")
print(f"Tổng số opcodes trích xuất: {len(my_opcodes)}")
print("10 opcodes đầu tiên:", my_opcodes[:10])
# Output: ['push', 'mov', 'sub', 'call', 'test', 'je', 'mov', 'add', 'pop', 'ret']
```

### Bước 2: Nhúng Opcode bằng Word2Vec (Biến chữ thành số)

Mạng GAN không hiểu chuỗi văn bản (`'mov'`, `'push'`). Ta dùng **Word2Vec** để nhúng (embed) chúng. Word2Vec sẽ coi toàn bộ file binary là một "đoạn văn", và mỗi opcode là một "từ".

```python
from gensim.models import Word2Vec

# Giả sử bạn có danh sách opcodes của NHIỀU file (để train hiệu quả)
# Mỗi sub-list là opcodes của 1 file exe
dataset_opcodes = [
    ['push', 'mov', 'sub', 'call', 'test', 'je', 'pop', 'ret'], # Mẫu 1
    ['xor', 'push', 'mov', 'call', 'add', 'ret'],               # Mẫu 2
    my_opcodes                                                  # Mẫu 3 từ phía trên
]

# 1. Khởi tạo và Train mô hình Word2Vec
print("Đang huấn luyện mô hình Word2Vec...")
model = Word2Vec(sentences=dataset_opcodes, 
                 vector_size=64, # Kích thước vector nhúng (VD: 64 chiều)
                 window=5,       # Cửa sổ ngữ cảnh (5 opcode lân cận)
                 min_count=1,    # Tần suất tối thiểu xuất hiện
                 workers=4)      # Số luồng CPU xử lý

# 2. Xem kết quả Vector của một Opcode (VD: 'mov')
vector_mov = model.wv['mov']
print(f"\nVector đại diện cho 'mov' (chiều dài {len(vector_mov)}):")
print(vector_mov[:5], "...") # In 5 giá trị đầu tiên

# 3. Tìm các opcode có ngữ cảnh giống 'mov' nhất
print("\nTop 3 opcode giống với 'mov' nhất trong ngữ cảnh code:")
similar_opcodes = model.wv.most_similar('mov', topn=3)
for op, score in similar_opcodes:
    print(f" - {op}: {score:.4f}")

# (Tùy chọn) Lưu mô hình để dùng lại cho GAN
# model.save("opcode_word2vec.model")
```

### Bước 3: Đưa vào mạng GAN (Tóm tắt ý tưởng)
Sau khi đã có Word2Vec, thay vì truyền các từ (`mov`, `push`), ta truyền **chuỗi các vector số thực** vào Generator và Discriminator (thường ở dạng mạng LSTM-GAN hoặc SeqGAN). Generator sẽ tạo ra một chuỗi vector (chưa có thật), sau đó ta dò ngược (cosine similarity) chuỗi vector này trong bộ từ điển Word2Vec để biến nó lại thành các opcode tạo ra *mã độc giả*.
