# Reflection — Lab 19

**Tên:** Đinh Văn Sinh
**Cohort:** A20 — 2A202601613
**Path đã chạy:** lite (fastembed + Qdrant in-memory + SQLite Feast)

---

## Câu hỏi (≤ 200 chữ)

Trên golden set 50 queries với 3 mode:

- **`exact` queries**: BM25 (keyword) thắng hoặc ngang hybrid — vì các từ kỹ thuật xuất hiện verbatim trong corpus, TF-IDF signal đã đủ mạnh, vector embedding không bổ sung thêm tín hiệu hữu ích.
- **`paraphrase` queries**: Vector (semantic) thắng — câu hỏi dùng từ đồng nghĩa nhưng embedding nắm được semantic similarity. Với `bge-small-en` (English-trained), tiếng Việt paraphrase bị giảm điểm so với `bge-m3`.
- **`mixed` queries**: Hybrid (RRF k=60) thắng rõ — kết hợp BM25 cho exact term lẫn vector cho phần paraphrase.

**Khi KHÔNG nên dùng hybrid:**
1. Corpus thuần keyword (văn bản pháp lý theo số hiệu) → BM25 đủ, hybrid tốn thêm compute vô ích.
2. Latency budget cực thấp (< 5ms) → semantic search quá chậm.
3. Embedding model không hỗ trợ ngôn ngữ corpus → semantic signal nhiễu hơn BM25.

---

## Điều ngạc nhiên nhất khi làm lab này

Post-filter recall sập hoàn toàn về 0 khi filter chỉ ~4% corpus mà không có exception hay log lỗi nào — hệ thống trả kết quả sai hoàn toàn một cách im lặng. Đây là lỗi production nguy hiểm nhất vì rất khó phát hiện.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _không có_

### Kết quả triển khai `HybridMemoryAgent` (`bonus/`):
Đã xây dựng thành công kiến trúc bộ nhớ AI cá nhân hoá kết hợp **Qdrant Vector Store** (Episodic memory) và **Feast Feature Store** (User Profile + Query Velocity):
1. **Episodic Recall**: Truy vấn *"Tôi đã đọc gì về Kubernetes?"* truy xuất chính xác 3 tài liệu liên quan về K8s/HPA/EKS với score > 0.82.
2. **Personalization**: Nạp realtime hồ sơ người dùng từ Feast (`topic_affinity=cloud`, `reading_speed=187 wpm`, `queries_last_1h=11`) để ghép ngữ cảnh phục vụ LLM.
3. **Paraphrase Handling**: Câu hỏi diễn đạt lại *"tự động mở rộng hạ tầng container"* đạt score 0.797 hướng đúng về cluster Kubernetes.
4. **Multi-tenant Isolation**: Kiểm thử bảo mật chứng minh user `u_002` hoàn toàn bị cô lập và không nhìn thấy bất kỳ ký ức nào của user `u_001`.
5. **Tài liệu thiết kế**: Hoàn thiện [`bonus/ARCHITECTURE.md`](file:///home/thviet/Test2024/Day19-Track2-DinhVanSinh-2A202601613/bonus/ARCHITECTURE.md) với sơ đồ ASCII, phân tích tradeoff 3 quyết định kiến trúc, đặc thù tiếng Việt (NĐ 13/2023/NĐ-CP, tokenization) và nhật ký Vibe-coding.

