# Hướng Dẫn Chạy Các Stage trong BCP Pipeline

## 📋 TỔNG QUAN 3 STAGES

Pipeline có **3 stages**:

```
Stage 1: Baseline Training
    ↓ Train baseline model trên labeled data
    ↓ Output: models/baseline.pth
    
Stage 2: Pseudo-Label Generation (CÓ THỂ BỎ QUA)
    ↓ Generate pseudo-labels từ unlabeled video
    ↓ Output: data/pseudo_labels/*.png
    
Stage 3: Final Training
    ↓ Train final model
    ↓ Output: models/final.pth
```

---

## 🎯 CÁC CÁCH CHẠY

### **CÁCH 1: Chạy từng stage riêng lẻ**

```bash
# Stage 1: Train baseline
python -m src.main baseline

# Stage 2: Generate pseudo-labels (KHÔNG BẮT BUỘC nếu dùng Mean Teacher)
python -m src.main pseudo

# Stage 3: Train final model
python -m src.main final                    # Không dùng Mean Teacher
python -m src.main final --use_mean_teacher # Dùng Mean Teacher
```

---

### **CÁCH 2: Chạy tất cả stages cùng lúc**

```bash
# Chạy cả 3 stages
python -m src.main all

# Chạy với Mean Teacher (bỏ qua Stage 2)
python -m src.main all --use_mean_teacher
```

---

### **CÁCH 3: Chạy Stage 3 tự động (KHuyên dùng)**

**Stage 3 tự động chạy Stage 1 và Stage 2 nếu cần:**

```bash
# Không dùng Mean Teacher → Tự động chạy Stage 1 + Stage 2 + Stage 3
python -m src.main final

# Dùng Mean Teacher → Tự động chạy Stage 1 + Stage 3 (BỎ QUA Stage 2)
python -m src.main final --use_mean_teacher
```

---

## ❓ STAGE 2 CÓ CẦN KHÔNG?

### **✅ CẦN Stage 2 khi:**

**KHÔNG dùng Mean Teacher** (`python -m src.main final`)

```
Stage 1 → Stage 2 → Stage 3
```

**Lý do:**
- Stage 3 cần pseudo-labels để train
- Stage 2 tạo pseudo-labels từ unlabeled video
- Nếu không có Stage 2 → Stage 3 sẽ tự động chạy Stage 2

**Flow:**
```python
# Stage 3 kiểm tra:
if not use_mean_teacher:
    if not os.path.exists(pseudo_dir):
        # Tự động chạy Stage 2
        generate_pseudo_labels(config, logger)
```

---

### **❌ KHÔNG CẦN Stage 2 khi:**

**Dùng Mean Teacher** (`python -m src.main final --use_mean_teacher`)

```
Stage 1 → Stage 3 (BỎ QUA Stage 2)
```

**Lý do:**
- Mean Teacher dùng unlabeled data trực tiếp
- Không cần tạo pseudo-labels trước
- Consistency loss được tính trong training

**Flow:**
```python
# Stage 3 với Mean Teacher:
if use_mean_teacher:
    # Load unlabeled data trực tiếp
    unlabeled_ds = VideoDataset(...)
    # Train với consistency loss
    trainer.train(labeled_loader, unlabeled_loader, ...)
```

---

## 📊 SO SÁNH 2 MODES

| **Aspect** | **Không Mean Teacher** | **Mean Teacher** |
|------------|----------------------|------------------|
| **Stage 2?** | ✅ Bắt buộc | ❌ Không cần |
| **Command** | `python -m src.main final` | `python -m src.main final --use_mean_teacher` |
| **Pseudo-labels** | Tạo trước (offline) | Không tạo (online) |
| **Unlabeled data** | Dùng qua pseudo-labels | Dùng trực tiếp |
| **Loss** | Supervised only | Supervised + Consistency |

---

## 🚀 VÍ DỤ THỰC TẾ

### **Ví dụ 1: Chạy với Pseudo-Labels (truyền thống)**

```bash
# Bước 1: Train baseline
python -m src.main baseline
# Output: models/baseline.pth

# Bước 2: Generate pseudo-labels
python -m src.main pseudo
# Output: data/pseudo_labels/*.png

# Bước 3: Train final model
python -m src.main final
# Output: models/final.pth
# Sử dụng: 60 labeled + 68 pseudo-labeled = 128 samples
```

---

### **Ví dụ 2: Chạy với Mean Teacher (khuyến nghị)**

```bash
# Chỉ cần 1 lệnh (tự động chạy Stage 1 nếu chưa có baseline.pth)
python -m src.main final --use_mean_teacher
# Output: models/final.pth
# Sử dụng: 60 labeled + 68 unlabeled (trực tiếp) = 128 samples
# Stage 2 được BỎ QUA
```

---

### **Ví dụ 3: Chạy tất cả tự động**

```bash
# Tự động chạy Stage 1 → Stage 2 → Stage 3
python -m src.main all

# Hoặc với Mean Teacher (bỏ qua Stage 2)
python -m src.main all --use_mean_teacher
```

---

## ⚙️ TỰ ĐỘNG HÓA TRONG STAGE 3

**Stage 3 tự động:**

1. ✅ **Kiểm tra baseline.pth**
   - Nếu không có → Tự động chạy Stage 1
   - Nếu có → Load weights để khởi tạo model

2. ✅ **Kiểm tra pseudo-labels** (chỉ khi không dùng Mean Teacher)
   - Nếu không có → Tự động chạy Stage 2
   - Nếu có → Load và dùng

3. ✅ **Train final model**
   - Với hoặc không có Mean Teacher

---

## 🎯 KHUYẾN NGHỊ

### **Nên dùng Mean Teacher vì:**

1. ✅ **Đơn giản hơn**: Chỉ cần 1 lệnh
2. ✅ **Hiệu quả hơn**: Pseudo-labels được update động
3. ✅ **Không cần Stage 2**: Tiết kiệm thời gian
4. ✅ **Consistency loss**: Giúp model học tốt hơn

### **Command khuyến nghị:**

```bash
# Cách đơn giản nhất (tự động chạy Stage 1 nếu cần)
python -m src.main final --use_mean_teacher
```

---

## 📝 TÓM TẮT

**Stage 2 CÓ CẦN KHÔNG?**

- ❌ **KHÔNG CẦN** nếu dùng Mean Teacher (`--use_mean_teacher`)
- ✅ **CẦN** nếu không dùng Mean Teacher (nhưng Stage 3 sẽ tự động chạy nếu thiếu)

**Cách chạy đơn giản nhất:**

```bash
python -m src.main final --use_mean_teacher
```

→ Tự động chạy Stage 1 (nếu cần) → Stage 3 (bỏ qua Stage 2)

