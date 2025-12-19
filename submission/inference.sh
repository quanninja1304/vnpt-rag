#!/bin/bash

# In ra log để biết pipeline bắt đầu chạy
echo "🚀 [INFO] Starting Inference Pipeline..."

# Chạy file Python chính
# Lưu ý: Python sẽ tự tìm các module trong thư mục hiện tại (/code)
python3 predict.py

# Kiểm tra mã lỗi trả về của Python (nếu có lỗi thì báo ngay)
if [ $? -eq 0 ]; then
    echo "✅ [SUCCESS] Inference finished successfully."
else
    echo "❌ [ERROR] Inference failed!"
    exit 1
fi