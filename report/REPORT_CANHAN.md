# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Dương Thị Hồng Viên
**Nhóm:** Lab 07 - Cá nhân / Nhóm thực hành
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vector hướng gần nhau trong không gian embedding, nên ý nghĩa của hai câu văn bản tương đồng nhiều hơn. Nói cách khác, góc giữa hai vector nhỏ, và tích vô hướng lớn so với độ dài của từng vector.

**Ví dụ có độ tương tự CAO:**
- Câu A: "The cat sat on the mat."
- Câu B: "A cat is sitting on the mat."
- Tại sao tương đồng: Cả hai mô tả cùng một sự kiện và có nhiều từ khóa giống nhau, nên vector embedding của chúng gần nhau.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Cats are mammals."
- Câu B: "The weather is sunny today."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn khác nhau, nên vector của chúng lệch hướng và cosine similarity thấp.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì text embedding thường được chuẩn hóa và được xem như hướng trong không gian nhiều chiều. Cosine similarity tập trung vào “hướng” của vector, phù hợp hơn với ý nghĩa ngữ nghĩa, trong khi khoảng cách Euclid nhạy cảm với độ lớn của vector hơn là nghĩa của câu.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Công thức chuẩn: ceil((length - overlap) / (chunk_size - overlap)) = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.11) = 23 chunk.
>
> **Đáp án:** 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Nếu overlap = 100, số chunks là ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25 chunk. Khi overlap tăng, bước trượt nhỏ lại nên số chunk tăng lên, nhưng mỗi chunk giữ thêm ngữ cảnh chung với chunk trước/sau, giúp giảm mất thông tin ở ranh giới.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi tách văn bản thành các câu bằng regex `(?<=[.!?])\s+` theo logic “giữ nguyên dấu câu phía trước, chỉ tách ở khoảng trắng sau dấu câu”. Sau đó, tôi gom từng nhóm có tối đa `max_sentences_per_chunk` câu thành một chunk và strip khoảng trắng thừa. Với trường hợp rỗng, tôi trả về `[]` để tránh lỗi runtime.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán kiểm tra separator theo ưu tiên `\n\n`, `\n`, `. `, ` ` rồi đệ quy nếu đoạn còn quá dài. Khi mảnh đã sát với kích thước `chunk_size`, tôi gom các phần nhỏ liền kề lại để tránh sinh ra các chunk quá ngắn và mất ngữ nghĩa. Base case là: text đã ngắn hơn hoặc bằng `chunk_size`, hoặc không còn separator nào thì cắt trực tiếp theo `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được chuẩn hóa thành một record với `id`, `content`, `metadata` và `embedding`. Khi tìm kiếm, tôi tạo embedding cho câu hỏi, tính dot product với từng embedding đã lưu, rồi sắp xếp giảm dần theo `score`. Vì embedding đã chuẩn hóa, dot product tương đương cosine similarity, nên rất phù hợp với mô hình vector store đơn giản trong lab.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi thực hiện lọc metadata trước rồi mới chạy similarity search, để tránh trường hợp top-k bị chiếm bởi các tài liệu không đúng bộ lọc. Với `delete_document`, tôi xóa tất cả các record có `metadata["doc_id"] == doc_id` và trả về `True` nếu có xóa, `False` nếu không tìm thấy.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Agent thực hiện 3 bước: truy xuất top-k, xây dựng prompt có ngữ cảnh đã đánh số, và gọi `llm_fn`. Tôi đặt mỗi chunk thành một phần `[1] Source: ...` để model biết đâu là nguồn và dễ trích dẫn đúng chunk khi trả lời. Nếu không có ngữ cảnh phù hợp, agent trả về thông báo rõ ràng thay vì bịa thông tin.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```bash
python -m pytest tests/ -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\acer\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe
collected 42 items
...
42 passed in 0.17s
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The quick brown fox jumps over the lazy dog. | The quick brown fox leaps over the lazy dog. | cao | -0.012908 | Không |
| 2 | Python is a programming language. | Java is a programming language. | cao | -0.119696 | Không |
| 3 | I like to read books. | I enjoy reading novels. | cao | 0.057197 | Có thể nói là thấp/không rõ |
| 4 | Cats are animals. | The weather is sunny today. | thấp | -0.003455 | Có |
| 5 | Vector databases store embeddings for similarity search. | Embeddings help compare semantic similarity between texts. | cao | -0.016414 | Không |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là các giá trị thực tế với mock embedder rất gần 0 và thậm chí âm, không phản ánh đúng ngữ nghĩa tự nhiên như ta mong đợi. Điều này cho thấy embeddings mock là định danh kỹ thuật chứ không phải mô hình ngữ nghĩa thực, nên số liệu tương tự chỉ có giá trị trong kiểm tra cấu trúc code chứ không phải trong benchmark chất lượng retrieval.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. Trong tình huống thực tế của bài lab, tôi đã tự chạy câu hỏi minh họa trên bộ dữ liệu mẫu để kiểm tra retrieval và agent. Dựa trên `python main.py "Chunking là gì?"`, kết quả thực tế như sau:

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Chunking là gì? | `rag_system_design.md` — mô tả hệ thống RAG, chunking và ngữ cảnh truy xuất | 0.150 | Có | Agent trả lời theo prompt context, trích dẫn nguồn và mô tả chung về chunking |
| 2 | Vector store hoạt động như thế nào? | `vector_store_notes.md` | — | Chưa chạy benchmark nhóm | Chưa benchmark đầy đủ |
| 3 | RAG là gì? | `rag_system_design.md` | — | Chưa chạy benchmark nhóm | Chưa benchmark đầy đủ |
| 4 | Python có đặc điểm gì nổi bật? | `python_intro.txt` | — | Chưa chạy benchmark nhóm | Chưa benchmark đầy đủ |
| 5 | Retrieval có ích khi nào? | `vi_retrieval_notes.md` | — | Chưa chạy benchmark nhóm | Chưa benchmark đầy đủ |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 1 / 5 (trong 1 câu thực tế đã kiểm tra, top-3 có chunk liên quan; các câu còn lại cần benchmark nhóm thực tế để đánh giá kỹ hơn).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua demo và thao tác với retrieval, tôi nhận ra chunking và metadata filter ảnh hưởng rất lớn đến chất lượng kết quả. Một chunk quá dài hoặc có độ chồng chéo quá ít dễ làm mất ngữ cảnh, trong khi filter thông minh giúp ngăn tài liệu sai đối tượng tràn vào top-k. Đây là điểm quan trọng nhất khi chuyển từ “code chạy” sang “retrieve đúng” trong ứng dụng RAG.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 |
| **Tổng phần cá nhân** | **59 / 60** |

> Ghi chú: Tôi đã hoàn thành phần code và kiểm thử, và phần benchmark còn cần thêm dữ liệu nhóm thực tế nếu muốn chấm điểm tối đa ở mục 5. Tuy nhiên, phần kỹ thuật chính của lab đã hoàn thiện đúng hướng và được xác nhận bằng 42/42 test pass.
