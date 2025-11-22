# Phân đoạn mạch bạch huyết (Lymphatic Vessel Segmentation)

Dự án này cung cấp một quy trình bán giám sát (semi-supervised) để phân đoạn các mạch bạch huyết trong video. Nó sử dụng mô hình **UNet++** với bộ mã hóa **ResNet34** và áp dụng phương pháp **Mean Teacher** để tận dụng dữ liệu chưa gán nhãn.

## Tính năng nổi bật

* **Kiến trúc UNet++:** Mô hình học sâu mạnh mẽ chuyên biệt cho phân đoạn hình ảnh y tế.
* **Mô hình nâng cao Stitch-ViT**: Một mô hình thay thế tích hợp Vision Transformer (ViT) với cơ chế ghép nối (stitching) để có khả năng trích xuất đặc trưng tốt hơn.
* **Học bán giám sát (Mean Teacher):** Tự động cải thiện độ chính xác bằng cách học từ dữ liệu video chưa được gán nhãn thông qua cơ chế giáo viên-học sinh (Teacher-Student).
* **Hàm mất mát nhận biết đường viền:** Kết hợp Dice Loss và Boundary Loss để phân đoạn chính xác các cạnh mạch máu nhỏ.
* **Quy trình 2 giai đoạn tinh gọn:**
    1.  **Stage 1:** Huấn luyện mô hình cơ sở (Baseline) trên dữ liệu có nhãn.
    2.  **Stage 2:** Huấn luyện mô hình cuối cùng (Final) sử dụng Mean Teacher.
* **Cấu hình linh hoạt:** Dễ dàng tùy chỉnh tham số qua các file JSON riêng biệt cho từng giai đoạn (`config_stage1.json`, `config_stage2.json`).
* **GUI Application:** Công cụ trực quan hỗ trợ đo đạc đường kính mạch và kiểm tra kết quả.

## Cấu trúc dự án

```text
.
├── app.py                   # Ứng dụng GUI
├── config.json              # Cấu hình chung (chọn type: Human hoặc Rat)
├── config_stage1.json       # Cấu hình cho Stage 1 (Baseline)
├── config_stage2.json       # Cấu hình cho Stage 2 (Final - Mean Teacher)
├── config_stage1_stitchvit.json # Cấu hình cho Stage 1 (Stitch-ViT)
├── config_stage2_stitchvit.json # Cấu hình cho Stage 2 (Stitch-ViT)
├── data/
│   ├── annotated/           # Chứa ảnh gốc và file JSON nhãn (Human/Rat)
│   ├── masks/               # Chứa ảnh mặt nạ nhị phân sau khi convert
│   └── video/               # Chứa video thô chưa gán nhãn
├── models/                  # Nơi lưu trọng số mô hình (weights)
├── logs/                    # Nhật ký huấn luyện
└── src/                     # Mã nguồn chính
```

## Cài đặt

### 1. Thiết lập môi trường

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Cấu hình Dữ liệu & Loại (Type)

Mở các file `config.json`, `config_stage1.json`, và `config_stage2.json`. Đặt trường `"type"` thành `"Human"` hoặc `"Rat"` tùy vào dữ liệu bạn đang làm việc.

**Ví dụ:**

```json
{
    "type": "Human",
    ...
}
```

### 3. Chuẩn bị dữ liệu

Dựa trên `type` bạn đã chọn (ví dụ: `Human`):

*   **Dữ liệu có nhãn:** Đặt ảnh và file annotation `.json` vào `data/annotated/Human/`.
*   **Dữ liệu chưa nhãn:** Đặt video vào `data/video/Human/`.

## Hướng dẫn sử dụng (Pipeline)

Hệ thống sẽ tự động nhận diện các thư mục dựa trên `type` trong file config.

### Chạy từng giai đoạn

**Giai đoạn 1: Huấn luyện Baseline**
Đây là bước bắt buộc để tạo trọng số khởi tạo cho mô hình Mean Teacher.

```bash
python -m src.main baseline
```

**Giai đoạn 2: Huấn luyện Final (Mean Teacher)**
Mô hình sẽ sử dụng trọng số từ Stage 1 và bắt đầu quy trình bán giám sát.

```bash
python -m src.main final
``` 

### Chạy toàn bộ quy trình (Khuyến nghị)

Lệnh này sẽ chạy tuần tự Stage 1 -> Stage 2 và tự động trực quan hóa kết quả sau khi xong.

```bash
python -m src.main all --visualize
```

### Chạy mô hình Stitch-ViT

Để sử dụng mô hình Stitch-ViT, chỉ định các file cấu hình của nó bằng flag `--config`.

**Stage 1 (Stitch-ViT):**
```bash
python -m src.main baseline --config config_stage1_stitchvit.json
```

**Stage 2 (Stitch-ViT):**
```bash
python -m src.main final --config config_stage2_stitchvit.json
```

**Toàn bộ quy trình (Stitch-ViT):**
```bash
python -m src.main all --config config_stage1_stitchvit.json --visualize
python -m src.main all --config config_stage2_stitchvit.json --visualize
```

### Các tùy chọn khác (Flags)

*   `--config <path>`: Sử dụng file config tùy chỉnh.
*   `--small-test`: Chạy thử nghiệm nhanh trên tập dữ liệu nhỏ (debug).
*   `--visualize`: Vẽ biểu đồ dự đoán sau khi train xong.
*   `--early-stop-patience <int>`: Ghi đè số epoch đợi dừng sớm (early stopping).

## Công cụ hỗ trợ (Tools)

### 1. Chuyển đổi nhãn JSON sang Mask (Binary Images)

Dùng để chuẩn bị dữ liệu training từ file annotation (ví dụ từ LabelMe).

```bash
python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
```

### 2. Trích xuất khung hình từ Video

Dùng để tạo dữ liệu unlabeled cho quá trình bán giám sát.

```bash
python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
```

### 3. Trực quan hóa & Đánh giá

**Xem kết quả dự đoán:**

```bash
python -m tools.scripts.visualize_predictions
```

**Vẽ biểu đồ Loss/Accuracy:**

```bash
python -m tools.scripts.plot_training_curves
```

**Tạo bảng tổng kết chỉ số (Evaluation Summary):**

```bash
python -m src.main visualize_eval
```

### 4. So sánh các mô hình

So sánh kết quả dự đoán của hai mô hình khác nhau.

```bash
python -m tools.scripts.compare_models --log-dir1 <đường_dẫn_tới_log_model1> --log-dir2 <đường_dẫn_tới_log_model2>
```

## GUI Application

Khởi chạy ứng dụng giao diện người dùng để xem và phân tích kết quả trực quan:

```bash
python app.py
```

## Chi tiết kỹ thuật

*   **Mô hình:** UNet++ (Backbone ResNet34).
*   **Chiến lược bán giám sát:** Mean Teacher. Mô hình Teacher là bản sao trọng số trung bình (Exponential Moving Average - EMA) của Student. Student học từ dữ liệu có nhãn và học tính nhất quán (consistency) từ Teacher trên dữ liệu chưa nhãn.
