# Phân Vùng Mạch Bạch Huyết với CTO-Net

Dự án này cải tiến và mở rộng một quy trình học có giám sát nền tảng bằng cách giới thiệu phương pháp **học bán giám sát (semi-supervised)** để phân vùng các mạch bạch huyết. Cốt lõi của dự án là việc triển khai thuật toán **Mean Teacher** để tận dụng một lượng lớn dữ liệu video không được gán nhãn, cùng với việc đánh giá các kiến trúc nâng cao như **CTO-Net** và **CTO Stitch-ViT**.

**Tài Nguyên Dự Án (Dataset, Models, Kết Quả):** [Google Drive Link](https://drive.google.com/drive/folders/1ORzUm1P5PK35O_L4YQ2L_IgIZZbkXi-0)

## Mục Lục
- [Bối Cảnh Dự Án](#bối-cảnh-dự-án)
- [Nhóm Dự Án và Phân Công Nhiệm Vụ](#nhóm-dự-án-và-phân-công-nhiệm-vụ)
- [Các Tính Năng Chính](#các-tính-năng-chính)
- [Các Mô Hình](#các-mô-hình)
- [Chi Tiết Bộ Dữ Liệu](#chi-tiết-bộ-dữ-liệu)
- [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
- [Cài Đặt](#cài-đặt)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
- [Công Cụ & Scripts](#công-cụ--scripts)
- [Ứng Dụng Giao Diện Đồ Họa (GUI)](#ứng-dụng-giao-diện-đồ-họa-gui)
- [Kết Quả Thực Nghiệm và Phân Tích](#kết-quả-thực-nghiệm-và-phân-tích)
- [Phân Tích Dự Án: Thành Công và Hạn Chế](#phân-tích-dự-án-thành-công-và-hạn-chế)
- [Những Gì Đã Học Được](#những-gì-đã-học-được)

## Nhóm Dự Án và Phân Công Nhiệm Vụ

Dự án này là nỗ lực hợp tác giữa hai thành viên. Các nhiệm vụ được phân chia để đảm bảo bao quát toàn diện tất cả các khía cạnh của dự án, từ phát triển mô hình, quản lý dữ liệu đến phân tích cuối cùng.

| Tên Thành Viên      | MSSV      | Trách Nhiệm Chính                                                                                                                                                                                          |
| -------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vũ Hải Tuấn**      | 2353280   | <ul><li>Chịu trách nhiệm chính về việc triển khai và huấn luyện các kiến trúc mô hình lõi: **CTO-Net** và mô hình thử nghiệm **CTO Stitch-ViT**.</li><li>Phát triển quy trình huấn luyện có giám sát (Giai đoạn 1).</li></ul> |
| **Lê Hoàng Chí Vĩ**  | 2353336   | <ul><li>Chịu trách nhiệm chính về việc triển khai và tinh chỉnh quy trình học bán giám sát (Giai đoạn 2) sử dụng thuật toán **Mean Teacher**.</li><li>Phát triển các script xử lý dữ liệu và trực quan hóa kết quả.</li></ul> |

**Đóng góp chung (Lê Hoàng Chí Vĩ & Vũ Hải Tuấn):**
*   Thiết kế và tinh chỉnh hàm loss tổng hợp (BCE, Dice, và Boundary Loss).
*   Phân tích kết quả thực nghiệm để xác định điểm mạnh và điểm yếu của mô hình.
*   Chuẩn bị báo cáo cuối kỳ, tài liệu và bài thuyết trình.
 
## Bối Cảnh Dự Án

Công trình này là sự tiếp nối trực tiếp của dự án **"Deep learning in Medical Researches: Lymphatic Vessel Segmentation"** ([Kho lưu trữ gốc](https://github.com/TUng1872004/Lymphatic-vessel)). Dự án gốc đã thiết lập một quy trình học hoàn toàn có giám sát và giới thiệu các mô hình nền tảng như UNet++ cho bài toán này.

Chúng tôi mở rộng công trình này bằng cách tích hợp một mô hình học bán giám sát để giảm sự phụ thuộc vào dữ liệu được gán nhãn thủ công. Ứng dụng GUI tương tác (`app.py`) ban đầu được phát triển bởi **Vũ Hoàng Tùng** trong khuôn khổ dự án nền tảng, và chúng tôi đã điều chỉnh nó cho quy trình mới của mình. Chúng tôi xin gửi lời cảm ơn chân thành vì sự đóng góp quan trọng này.

## Các Tính Năng Chính

*   **Kiến trúc chính (CTO-Net):** Một mô hình phân vùng hiệu suất cao với backbone **Res2Net-50** được thiết kế để trích xuất đặc trưng chi tiết.
*   **Học Bán Giám Sát (Semi-Supervised Learning):** Tận dụng dữ liệu video không nhãn thông qua phương pháp **Mean Teacher** để cải thiện độ chính xác và giảm công sức gán nhãn.
*   **Hàm Loss Nhận Diện Biên (Boundary-Aware Loss):** Kết hợp Dice Loss và Boundary Loss để phát hiện các cạnh của mạch máu một cách sắc nét và chính xác.
*   **Deep Supervision:** Sử dụng các đầu ra phụ trợ (auxiliary heads) ở các tỷ lệ khác nhau để cải thiện luồng gradient và khả năng học đặc trưng.
*   **Quy Trình Huấn Luyện 2 Giai Đoạn:**
    1.  **Giai đoạn 1:** Huấn luyện một mô hình cơ sở (baseline) trên một tập dữ liệu nhỏ có nhãn.
    2.  **Giai đoạn 2:** Tinh chỉnh mô hình bằng phương pháp Mean Teacher với cả dữ liệu có nhãn và một lượng lớn dữ liệu không nhãn.
*   **Cấu Hình Linh Hoạt:** Dễ dàng điều chỉnh các tham số cho mỗi giai đoạn thông qua các tệp JSON chuyên dụng.
*   **GUI Phân Tích:** Một công cụ tương tác để trực quan hóa các dự đoán và đo đường kính mạch máu.

## Các Mô Hình

Dự án này hỗ trợ ba mô hình:

1.  **CTO-Net (Mặc định):** Mô hình chính của dự án. Nó sử dụng backbone **Res2Net-50** và một kiến trúc tùy chỉnh được thiết kế để phân vùng mạch máu. Cấu hình cho mô hình này là `cto`.
2.  **CTO Stitch-ViT (Thử nghiệm):** Một phiên bản nâng cao của CTO-Net tích hợp các khối Vision Transformer (ViT) với cơ chế "stitching". Mô hình này nhằm mục đích nắm bắt các đặc trưng ngữ cảnh toàn cục tốt hơn. Cấu hình cho mô hình này là `cto_stitchvit`.
3.  **UNet++:** Một kiến trúc nổi tiếng cho phân vùng ảnh y tế, có sẵn dưới dạng một lựa chọn thay thế đã được sử dụng trong dự án gốc. Nó có thể được cấu hình bằng cách đặt `name` của mô hình thành `unetpp` trong tệp cấu hình JSON.
 
## Chi Tiết Bộ Dữ Liệu

Dự án này sử dụng hai bộ dữ liệu chính để phân vùng mạch bạch huyết. Tất cả dữ liệu được cung cấp bởi phòng thí nghiệm của nghiên cứu viên **Lê Quỳnh Trâm**, người cũng đã thực hiện toàn bộ việc gán nhãn thủ công cho dữ liệu có nhãn.

### Bộ Dữ Liệu Người (Human Dataset)
*   **Nguồn:** Phòng thí nghiệm của NCV. Lê Quỳnh Trâm.
*   **Nội dung:** Các video ghi lại mạch bạch huyết trong mô người.
*   **Thống kê:**
    *   **Dữ liệu có nhãn:** 33 ảnh với chú thích thủ công.
    *   **Dữ liệu không nhãn:** 3 video, từ đó các khung hình được trích xuất.
*   **Xử lý:** Dữ liệu có nhãn (chú thích `.json`) được chuyển đổi thành các mặt nạ nhị phân (binary mask). Đối với dữ liệu không nhãn, các khung hình được trích xuất từ video để sử dụng trong giai đoạn huấn luyện bán giám sát.

### Bộ Dữ Liệu Chuột (Rat Dataset)
*   **Nguồn:** Phòng thí nghiệm của NCV. Lê Quỳnh Trâm.
*   **Nội dung:** Các video ghi lại mạch bạch huyết trong mô chuột.
*   **Thống kê:**
    *   **Dữ liệu có nhãn:** 33 ảnh với chú thích thủ công.
    *   **Dữ liệu không nhãn:** 8 video, từ đó các khung hình được trích xuất.
*   **Xử lý:** Tương tự như bộ dữ liệu Người, các mặt nạ được tạo ra từ các chú thích và các khung hình được trích xuất từ video.


## Cấu Trúc Dự Án
```text
.
├── app.py                      # File chạy ứng dụng GUI
├── config_stage1.json          # Cấu hình cho Giai đoạn 1 (CTO-Net)
├── config_stage2.json          # Cấu hình cho Giai đoạn 2 (CTO-Net)
├── requirements.txt
├── data/
│   ├── annotated/              # Ảnh và chú thích JSON có nhãn
│   ├── masks/                  # Các mặt nạ nhị phân đã được tạo
│   ├── frames/                 # Các khung hình được trích xuất từ video không nhãn
│   └── video/                  # Các video thô không nhãn
├── logs/                       # Log huấn luyện, biểu đồ và ảnh dự đoán
├── models/                     # Các checkpoint của mô hình đã lưu (.pth)
└── src/                        # Mã nguồn
```

## Cài Đặt

1.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình loại dữ liệu:**
    Mở các tệp cấu hình (ví dụ: `config_stage1.json`). Đặt trường `"type"` thành `"Human"` hoặc `"Rat"` để chỉ định bộ dữ liệu.

    **Ví dụ (`config_stage1.json`):**
    ```json
    {
        "type": "Human",
        ...
    }
    ```

## Hướng Dẫn Sử Dụng

### Chuẩn Bị Dữ Liệu

1.  **Dữ liệu có nhãn:** Đặt các tệp ảnh và `.json` vào `data/annotated/<type>/`.
2.  **Chuyển đổi chú thích sang mặt nạ (mask):**
    ```bash
    python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
    ```
3.  **Dữ liệu không nhãn:** Đặt các video vào `data/video/<type>/`.
4.  **Trích xuất khung hình từ video:**
    ```bash
    python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
    ```

### Quy Trình Huấn Luyện

Chạy toàn bộ quy trình (Giai đoạn 1 và Giai đoạn 2) cho mô hình **CTO-Net** mặc định:
```bash
python -m src.main all --visualize
```
Lệnh này sử dụng `config_stage1.json` và `config_stage2.json` làm mặc định.

Để chạy với mô hình thử nghiệm **Stitch-ViT**, hãy chỉ định các tệp cấu hình của nó:
```bash
# Giai đoạn 1
python -m src.main baseline --config config_stage1_stitchvit.json --visualize
# Giai đoạn 2
python -m src.main final --config config_stage2_stitchvit.json --visualize
```

## Công Cụ & Scripts
Các script hữu ích được đặt trong `tools/scripts/`.

*   **So sánh các mô hình:**
    ```bash
    python -m tools.scripts.compare_models --log-dir1 <path_to_model1_logs> --log-dir2 <path_to_model2_logs>
    ```
*   **Vẽ biểu đồ huấn luyện:**
    ```bash
    python -m tools.scripts.plot_training_curves --log-dir <path_to_log_directory>
    ```
*   **Trực quan hóa dự đoán:**
    ```bash
    python -m tools.scripts.visualize_predictions --log-dir <path_to_log_directory>
    ```
*   **Tạo Bảng Đánh Giá:**
    ```bash
    python -m src.main visualize_eval --log-dir <path_to_log_directory>
    ```

## Ứng Dụng Giao Diện Đồ Họa (GUI)
Dự án bao gồm một giao diện đồ họa người dùng để dự đoán và phân tích có tương tác.
**Khởi chạy GUI:**
```bash
python app.py
```

## Kết Quả Thực Nghiệm và Phân Tích

Hiệu suất của mô hình được đánh giá bằng các chỉ số Dice Score, Intersection over Union (IoU), Precision và Recall. Quá trình huấn luyện bao gồm hai giai đoạn: huấn luyện cơ sở trên dữ liệu có nhãn và tinh chỉnh bán giám sát bằng phương pháp Mean Teacher.

### So Sánh Định Lượng

Để cung cấp một sự so sánh rõ ràng, bảng dưới đây tóm tắt hiệu suất của cả **CTO-Net** và mô hình thử nghiệm **CTO Stitch-ViT** trên cả hai bộ dữ liệu Người và Chuột. Các chỉ số được báo cáo cho Giai đoạn 2 (mô hình cuối cùng sau khi học bán giám sát).

| Model              | Dataset | Dice Score | IoU Score  | Boundary F1 |
| ------------------ | ------- | :--------: | :--------: | :---------: |
| **CTO-Net**        | Người   | **95.76%** | **91.88%** | **72.16%**  |
| (Res2Net-50)       | Chuột   |  91.07%    |  83.64%    |   75.78%    |
| **CTO Stitch-ViT** | Người   |  92.43%    |  86.09%    |   59.52%    |
| (Thử nghiệm)       | Chuột   | **91.99%** | **85.38%** | **85.14%**  |

**Phân tích kết quả định lượng:**
*   **Sự cải thiện từ học bán giám sát:** Nhìn chung, kết quả Giai đoạn 2 (không hiển thị trong bảng tóm tắt này nhưng có trong log) cho thấy sự cải thiện nhất quán so với Giai đoạn 1, xác thực hiệu quả của phương pháp Mean Teacher.
*   **CTO-Net trên bộ dữ liệu Người:** `CTO-Net` tiêu chuẩn cho thấy hiệu suất vượt trội trên bộ dữ liệu Người, vượt qua mô hình Stitch-ViT thử nghiệm trên tất cả các chỉ số chính. Điều này cho thấy kiến trúc của nó rất hiệu quả đối với cấu trúc mạch trong bộ dữ liệu này.
*   **Stitch-ViT trên bộ dữ liệu Chuột:** Kết quả củng cố giả thuyết ban đầu của chúng tôi. Mô hình `CTO Stitch-ViT` cho thấy sự gia tăng hiệu suất đáng chú ý trên bộ dữ liệu Chuột, đặc biệt là ở điểm Boundary F1. Điều này cho thấy khả năng nắm bắt ngữ cảnh toàn cục của nó đặc biệt thuận lợi cho các cấu trúc mạch phức tạp và đa dạng hơn có trong dữ liệu đó.

### Biểu Đồ Huấn Luyện
Các biểu đồ huấn luyện cho thấy sự tiến triển của điểm Dice và loss, cho thấy sự hội tụ thành công cho tất cả các mô hình. Các đường cong Giai đoạn 2 thể hiện sự ổn định hơn nữa khi các mô hình học hỏi từ hàng nghìn khung hình không được gán nhãn.

#### CTO-Net (Vanilla)
**Bộ dữ liệu Người:**
![CTO-Net Human Curves](Human/CTO-Net/training_curves.png)

**Bộ dữ liệu Chuột:**
![CTO-Net Rat Curves](Rat/CTO-Net/training_curves.png)

#### CTO Stitch-ViT (Thử nghiệm)
**Bộ dữ liệu Người:**
![Stitch-ViT Human Curves](Human/CTO-Stitch-ViT/training_curves.png)

**Bộ dữ liệu Chuột:**
![Stitch-ViT Rat Curves](Rat/CTO-Stitch-ViT/training_curves.png)

### Chất Lượng Dự Đoán
Kiểm tra trực quan cho thấy sự cải thiện đáng kể từ mô hình cơ sở đến mô hình cuối cùng. Mô hình cuối cùng tạo ra các phân vùng sạch hơn nhiều với ít dương tính giả (false positives) hơn.

#### CTO-Net (Vanilla)
**Bộ dữ liệu Người:**
![CTO-Net Human Predictions](Human/CTO-Net/final_predictions.png)

**Bộ dữ liệu Chuột:**
![CTO-Net Rat Predictions](Rat/CTO-Net/final_predictions.png)

#### CTO Stitch-ViT (Thử nghiệm)
**Bộ dữ liệu Người:**
![Stitch-ViT Human Predictions](Human/CTO-Stitch-ViT/final_predictions.png)

**Bộ dữ liệu Chuột:**
![Stitch-ViT Rat Predictions](Rat/CTO-Stitch-ViT/final_predictions.png)

**Phân tích trực quan:**
- **Cơ sở vs. Cuối cùng:** Trong mọi trường hợp, các mô hình Giai đoạn 2 cuối cùng tạo ra các phân vùng mượt mà và mạch lạc hơn đáng kể so với các mô hình cơ sở Giai đoạn 1 của chúng. Consistency loss từ phương pháp Mean Teacher giúp giảm nhiễu và lấp đầy các khoảng trống một cách hiệu quả.
- **So sánh mô hình:** Trên bộ dữ liệu Chuột, các dự đoán từ `CTO Stitch-ViT` có vẻ khả quan hơn và nắm bắt được các chi tiết tốt hơn dọc theo ranh giới mạch máu, phù hợp với điểm Boundary F1 cao hơn của nó. Trên bộ dữ liệu Người, `CTO-Net` thì lại cung cấp các mặt nạ sạch và chính xác hơn.

## Phân Tích Dự Án: Thành Công và Hạn Chế

### Thành Công
1.  **Học Bán Giám Sát Hiệu Quả:** Quy trình Mean Teacher đã tận dụng thành công một lượng lớn dữ liệu video không nhãn để cải thiện đáng kể chất lượng phân vùng, chuyển từ các dự đoán nhiễu ở mô hình cơ sở sang các mặt nạ cuối cùng sạch sẽ và liền mạch. Điều này chứng tỏ tính khả thi của phương pháp này trong việc giảm chi phí gán nhãn thủ công.
2.  **Phân Vùng Chất Lượng Cao:** Sự kết hợp giữa backbone Res2Net, kiến trúc CTO-Net tùy chỉnh và hàm loss nhận diện biên đã chứng tỏ hiệu quả trong việc tạo ra các phân vùng sắc nét và chính xác của các mạch bạch huyết.
3.  **Quy Trình End-to-End:** Dự án cung cấp một quy trình hoàn chỉnh, có thể sử dụng được, từ xử lý video thô, tạo mặt nạ, đến huấn luyện, đánh giá và phân tích tương tác với GUI.

### Hạn Chế
1.  **Mô hình Stitch-ViT còn trong giai đoạn thử nghiệm:** Mô hình `cto_stitchvit` vẫn đang trong giai đoạn thử nghiệm. Mặc dù nó cho thấy tiềm năng trong việc nắm bắt ngữ cảnh toàn cục, nó đòi hỏi phải tinh chỉnh và đánh giá sâu hơn để xác minh một cách chắc chắn về lợi ích của nó so với CTO-Net tiêu chuẩn.
2.  **Hiệu suất phụ thuộc vào bộ dữ liệu của Stitch-ViT:** Như đã lưu ý trong kết quả, lợi ích của `cto_stitchvit` dường như phụ thuộc vào bộ dữ liệu. Ưu điểm của nó đối với bộ dữ liệu Chuột là rõ ràng, nhưng nó không hoàn toàn vượt trội hơn CTO-Net tiêu chuẩn trên bộ dữ liệu Người, cho thấy không có giải pháp nào phù hợp cho tất cả.
3.  **Phụ thuộc vào Nhãn Giả (Pseudo-Labels):** Hiệu suất của giai đoạn bán giám sát phụ thuộc nhiều vào chất lượng của các nhãn giả được tạo ra bởi teacher model. Trong trường hợp có sự khác biệt lớn (domain shift) giữa dữ liệu có nhãn và không nhãn, điều này có thể làm giảm hiệu suất.
4.  **Chi Phí Tính Toán:** Quá trình huấn luyện hai giai đoạn, đặc biệt với số lượng lớn khung hình không nhãn, đòi hỏi nhiều tài nguyên tính toán và tốn thời gian.

## Những Gì Đã Học Được

Dự án này mang lại kinh nghiệm thực tế quý báu trong một số lĩnh vực chính của học sâu và thị giác máy tính:

1.  **Học Bán Giám Sát:** Có được sự hiểu biết sâu sắc và thực tế về phương pháp Mean Teacher, bao gồm việc triển khai consistency loss và cập nhật trọng số bằng Trung bình Động Hàm mũ (EMA).
2.  **Kiến Trúc Mô Hình Nâng Cao:** Học cách triển khai và tích hợp các kiến trúc phức tạp, bao gồm backbone đa tỷ lệ Res2Net và các khối Vision Transformer thử nghiệm.
3.  **Kỹ Thuật Thiết Kế Hàm Loss:** Tích lũy kinh nghiệm trong việc thiết kế và cân bằng một hàm loss tổng hợp (BCE, Dice, Boundary) để tối ưu hóa cho các đặc điểm phân vùng cụ thể như độ sắc nét của các cạnh.
4.  **Xây Dựng Quy Trình ML Hoàn Chỉnh:** Phát triển kỹ năng tạo ra một quy trình làm việc end-to-end, bao gồm tiền xử lý dữ liệu, huấn luyện mô hình, trực quan hóa kết quả và xây dựng một ứng dụng tương tác cho việc sử dụng trong thực tế.
5.  **Tính Tái Tạo và Cấu Hình:** Học được tầm quan trọng của việc thiết lập cấu trúc mã nguồn rõ ràng, tài liệu hóa đầy đủ và sử dụng các tệp cấu hình bên ngoài (`.json`) để đảm bảo các thí nghiệm dễ dàng tái tạo và sửa đổi.
