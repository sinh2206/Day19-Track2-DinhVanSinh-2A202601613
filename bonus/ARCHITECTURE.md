# AI Memory Architecture — Hybrid Memory Agent

**Contributors:** Đinh Văn Sinh — 2A202601613  
**Lab:** Day 19 · Track 2 · VinUniversity AICB A20 2026  

---

## Đề bài tóm tắt

Thiết kế một **trợ lý AI cá nhân cho người dùng Việt Nam** có khả năng *nhớ*:
- **Episodic memory** — hội thoại, tài liệu đã đọc → **Vector Store** (Qdrant)
- **Stable user profile** — ngôn ngữ, tốc độ đọc, lĩnh vực → **Feature Store** (Feast)
- **Recent activity** — query 1 giờ qua, topic đang quan tâm → Feature Store (streaming view)

---

## Sơ đồ kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│                      USER REQUEST                           │
│              "Tôi đã đọc gì về Kubernetes?"                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  HybridMemory  │
                    │     Agent      │
                    └──────┬─────────┘
           ┌───────────────┴───────────────┐
           │                               │
   ┌───────▼──────────┐          ┌─────────▼──────────┐
   │   FEAST ONLINE   │          │   QDRANT VECTOR    │
   │   STORE (SQLite) │          │   STORE (in-memory)│
   │                  │          │                    │
   │ user_profile:    │          │ Collection:        │
   │  topic_affinity  │          │  user_memories     │
   │  reading_speed   │          │  (filtered by      │
   │  preferred_lang  │          │   user_id payload) │
   │                  │          │                    │
   │ query_velocity:  │          │ Each point:        │
   │  queries_last_1h │          │  vector (384-dim)  │
   │  distinct_topics │          │  payload: text,    │
   └───────┬──────────┘          │  timestamp, type   │
           │                     └─────────┬──────────┘
           │  features dict                │  top-K memories
           └───────────────┬───────────────┘
                           │
                  ┌────────▼─────────┐
                  │  Context Builder │
                  └────────┬─────────┘
                           │
              "User thích cloud/AI, đọc 220wpm.
               Recent: hỏi về K8s 3 lần/giờ.
               Top memories: [doc1, doc2, doc3]"
```

**Data flow khi `remember(text)`:**
```
text → chunk (per-message) → embed (fastembed 384d)
     → upsert Qdrant với payload {user_id, timestamp, type}
```

**Data flow khi `recall(query)`:**
```
query → [Feast online lookup] → user profile + recent activity
      → [Qdrant hybrid search] filtered by user_id
      → assemble_context(profile, memories)
      → return context string
```

---

## 3 Quyết định kiến trúc với tradeoff explicit

### Quyết định 1: Chunking Strategy — Per-message vs Per-conversation

**Lựa chọn: Per-message chunking** (mỗi tin nhắn = 1 vector point).

| | Per-message | Per-conversation | Semantic break |
|---|---|---|---|
| Retrieval quality | Cao — tìm đúng câu | Thấp — vector nhiễu | Cao nhưng phức tạp |
| Storage cost | O(n_messages) | O(n_conversations) | O(n_segments) |
| Context window | Nhỏ — dễ fit LLM | Lớn — dễ overflow | Trung bình |
| Freshness | Realtime | Batch sau conv | Realtime |

**Tại sao chọn per-message:** Người dùng Việt Nam thường hỏi lại câu hỏi cũ theo nhiều cách diễn đạt khác nhau. Per-message retrieval tìm đúng câu đó. Per-conversation trả về toàn bộ ngữ cảnh, dễ vượt context window và chứa nhiều noise.

**Rejected:** Semantic break chunking — tốt hơn nhưng cần VN-aware sentence splitter (underthesea). Với `bge-small-en`, boundary detection trên tiếng Việt không đáng tin.

---

### Quyết định 2: Feature Schema — Tabular vs Embedding features

**Lựa chọn: Tabular features thuần** (topic_affinity: string, reading_speed_wpm: int, preferred_lang: string).

| | Tabular features | Embedding features (latent prefs) |
|---|---|---|
| Interpretability | Cao — biết rõ user thích gì | Thấp — latent space khó giải thích |
| Freshness | Batch refresh hàng ngày | Cần pipeline streaming |
| Storage | Nhỏ (<1KB/user) | Lớn (6KB/user với 1536-dim) |
| Feast TTL fit | Tốt — daily TTL=30d | Phức tạp — TTL nào cho embedding? |

**Tại sao chọn tabular:** Tabular feature có **TTL rõ ràng** — topic_affinity thay đổi theo tuần (TTL=30d), queries_last_hour TTL=1h. Embedding feature không có TTL tự nhiên và khó debug khi personalization sai.

**Rejected:** Lưu embedding user preference vào Feature Store — re-index cycle của preference embedding (weekly) khác hẳn episodic memory (per-message), nên tách riêng là đúng kiến trúc. Đây là bài học PIT join của NB4: hai signal có freshness khác nhau không nên chung một entity timeline.

---

### Quyết định 3: Freshness Strategy — Phân tầng theo loại data

**Lựa chọn: 3 tầng freshness:**

| Data | Freshness cần | Giải pháp | TTL |
|---|---|---|---|
| Episodic memory (new doc) | Sub-second | Qdrant upsert synchronous trong `remember()` | Không giới hạn |
| query_velocity (queries_last_hour) | 5 phút | Feast micro-batch | 1 giờ |
| user_profile (topic_affinity) | Daily | Feast materialize-incremental | 30 ngày |

**3 use cases cụ thể:**
1. User vừa lưu tài liệu → `recall()` ngay phải thấy doc đó → **sub-second** (Qdrant upsert sync)
2. User chuyển topic từ cloud → AI sau 30 phút → 5 phút lag là OK → **micro-batch**
3. Reading speed thay đổi theo tháng → daily refresh là đủ → **daily batch**

**Rejected:** Sub-second cho tất cả — Redis streaming pipeline quá phức tạp cho POC. Batch cho episodic memory — user hỏi ngay sau khi lưu sẽ thấy MISS (freshness paradox: episodic phải fresh ngay, profile không cần).

---

## Vietnamese-context Considerations

### 1. Code-switching (vi/en mix)
Người dùng VN thường viết *"làm sao để debug Python code trên production?"* — nửa Việt nửa Anh. `bge-small-en` xử lý phần Anh tốt nhưng phần Việt yếu.

**Giải pháp trong POC:** Dùng `bge-small-en` với fallback — nếu query > 50% ký tự Unicode ngoài ASCII, log warning. Trong production: dùng `bge-m3` (multilingual, 1024d).

### 2. Tokenization cho BM25
Whitespace split không đúng cho tiếng Việt đa âm tiết (*"học máy"* = 2 tokens nhưng cùng nghĩa với *"machine learning"*).

**Tradeoff:** `underthesea.word_tokenize()` tăng ~15% recall trên paraphrase queries tiếng Việt nhưng thêm dependency ~30MB và chậm hơn ~2x. POC dùng whitespace split (honest limitation).

### 3. Privacy — Nghị định 13/2023/NĐ-CP
Episodic memory lưu nội dung hội thoại → dữ liệu cá nhân theo NĐ 13. Vector payload lưu plaintext → cần encrypt trước khi upsert (AES-256, key per-user trong KMS).

---

## Honest Limitations — What this POC doesn't handle yet

1. **Privacy isolation per user** — dùng payload filter `user_id` (soft isolation). Nếu filter bị miss (bug như NB5!), user A đọc được memory của user B.
2. **Memory decay / forgetting** — không có TTL cho episodic memory. Sau 1 năm → 100K vectors → search chậm.
3. **CRUD on memories** — chưa implement delete memory cụ thể.
4. **Multi-device sync** — Qdrant in-memory mất data khi restart.
5. **Encryption at rest** — vector payload lưu plaintext.

---

## Vibe Coding Workflow Log

**Prompt hiệu quả nhất:**
> *"Implement `HybridMemoryAgent.recall()` với spec: input=query+user_id, output=dict với keys features/doc_ids/affinity_used. Sub-steps: 1) Feast online lookup user_profile+query_velocity, 2) Qdrant hybrid search filtered by user_id, 3) assemble context string. Không gọi LLM thật."*

AI sinh đúng lần đầu vì spec đủ input/output/constraint.

**Prompt fail:**
> *"Make the agent personalized"* — AI thêm random re-ranking không có basis, phải rollback. Bài học: mơ hồ in → mơ hồ out.
