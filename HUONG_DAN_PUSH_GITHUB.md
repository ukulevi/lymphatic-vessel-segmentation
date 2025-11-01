# Hướng Dẫn Push Code Lên GitHub

## 📋 Các Lệnh Để Push Code Lên GitHub

### **Bước 1: Kiểm tra trạng thái Git**
```bash
git status
```

### **Bước 2: Thêm các thay đổi vào staging area**
```bash
# Thêm tất cả các file đã thay đổi
git add .

# Hoặc thêm file cụ thể
git add <tên_file>
```

### **Bước 3: Commit các thay đổi**
```bash
git commit -m "Mô tả thay đổi của bạn"
```

**Ví dụ:**
```bash
git commit -m "Update code và thêm tính năng mới"
```

### **Bước 4: Push code lên GitHub**
```bash
git push -u origin main
```

**Lưu ý:** Lần đầu tiên push sẽ yêu cầu xác thực:
- **Username:** Tuancoolboy
- **Password:** Sử dụng Personal Access Token (không phải mật khẩu GitHub)

---

## 🔐 Cách Tạo Personal Access Token

1. Vào GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Đặt tên token và chọn quyền `repo` (full control)
4. Click "Generate token"
5. **Copy token ngay** (chỉ hiển thị 1 lần)
6. Khi push code, dán token vào phần Password

---

## 📝 Quy Trình Hoàn Chỉnh (Lần Đầu Tiên)

```bash
# 1. Khởi tạo Git (nếu chưa có)
git init

# 2. Thêm remote origin
git remote add origin https://github.com/Tuancoolboy/Lymphatic_vessels.git

# 3. Đổi tên branch thành main
git branch -M main

# 4. Thêm tất cả files
git add .

# 5. Commit
git commit -m "Initial commit"

# 6. Push lên GitHub
git push -u origin main
```

---

## 🔄 Quy Trình Cập Nhật Code (Các Lần Sau)

```bash
# 1. Kiểm tra thay đổi
git status

# 2. Thêm thay đổi
git add .

# 3. Commit
git commit -m "Mô tả thay đổi"

# 4. Push lên GitHub
git push origin main
```

---

## 📥 Cách Pull Code Từ GitHub

```bash
# Lấy code mới nhất từ GitHub
git pull origin main

# Hoặc nếu đã set upstream
git pull
```

---

## 🛠️ Các Lệnh Git Hữu Ích Khác

```bash
# Xem lịch sử commit
git log

# Xem các thay đổi chưa commit
git diff

# Xem các branch
git branch

# Tạo branch mới
git checkout -b <tên_branch>

# Chuyển branch
git checkout <tên_branch>

# Xóa file khỏi Git (nhưng giữ lại ở local)
git rm --cached <tên_file>

# Hoàn tác commit (giữ lại thay đổi)
git reset --soft HEAD~1
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Luôn commit trước khi push** - Git yêu cầu có ít nhất 1 commit trước khi push
2. **Kiểm tra .gitignore** - Đảm bảo các file không cần thiết (như .DS_Store, venv/, __pycache__/) đã được thêm vào .gitignore
3. **Commit message rõ ràng** - Viết mô tả ngắn gọn về thay đổi
4. **Pull trước khi push** - Nếu làm việc nhóm, nên pull code mới nhất trước khi push

---

## 🚀 Lệnh Nhanh Để Push Code Ngay

```bash
git add . && git commit -m "Update code" && git push origin main
```

Chúc bạn thành công! 🎉

