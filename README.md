# 🧪 TEST_LIGHT_RAG

## ✅ Cách sử dụng

1. Chạy file `.py`:
   ```bash
   python your_file.py
   ```

2. Mở đường link Gradio được hiển thị trên terminal (thường là `http://127.0.0.1:9621`).

3. Tải lên tài liệu cần xử lý:
   - Hỗ trợ các định dạng: `.docx`, `.txt`
   - Nhấn nút **"📥 Thêm tài liệu"** để hệ thống xử lý nội dung.

4. Nhập câu hỏi vào khung chat. Chatbot sẽ trả lời dựa trên nội dung của tài liệu bạn đã tải lên.

---

## ⚙️ Yêu cầu

- Cài đặt [Ollama](https://ollama.com/)
- Tải mô hình cần thiết bằng lệnh:
   ```bash
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text
   ```

---

> ✨ Project sử dụng [LightRAG](https://github.com/HKUDS/LightRAG) kết hợp Ollama và Gradio để tạo chatbot trích xuất thông tin từ tài liệu tải lên.
