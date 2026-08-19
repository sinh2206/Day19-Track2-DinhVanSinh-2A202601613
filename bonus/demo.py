"""bonus/demo.py — 5-query demonstration of HybridMemoryAgent.

Run:  python bonus/demo.py
Expected: exits 0 with 5 query outputs printed.

Queries cover all 3 memory types:
  1. Episodic only (vector hit)
  2. Profile context needed (topic_affinity)
  3. Fresh activity needed (queries_last_hour)
  4. Paraphrase (vector wins)
  5. Mixed (hybrid + profile)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on path (works when run from any cwd)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from bonus.agent import HybridMemoryAgent  # noqa: E402


def separator(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Query {title}")
    print('='*60)


def main() -> int:
    print("Khởi tạo HybridMemoryAgent (Qdrant in-memory + fastembed)...")
    agent = HybridMemoryAgent()

    # ── Seed episodic memories for demo user ────────────────────────────────
    print("\n[Seeding memories for u_001]")
    memories_to_add = [
        "Kubernetes là nền tảng điều phối container giúp tự động triển khai và mở rộng ứng dụng.",
        "Đã đọc tài liệu về autoscaling HPA và VPA trong Kubernetes cluster.",
        "Ghi chú: tìm hiểu về Horizontal Pod Autoscaler cho dịch vụ xử lý video.",
        "Bài viết về AWS EKS: quản lý Kubernetes trên cloud không cần quản lý control plane.",
        "Cloud security: IAM roles, network policy, pod security standards trong K8s.",
        "Microservices pattern: service mesh Istio giúp quan sát và kiểm soát traffic.",
        "Feature store Feast: materialize offline data sang online store cho ML serving.",
        "RAG pipeline: vector search + LLM để trả lời câu hỏi từ tài liệu nội bộ.",
    ]
    for mem in memories_to_add:
        agent.remember(mem, user_id="u_001")
    print(f"  Đã thêm {len(memories_to_add)} memories cho u_001")

    # Thêm memories cho user khác để test isolation
    agent.remember("Dữ liệu tài chính bí mật của u_002", user_id="u_002")

    # ── 5 Queries ───────────────────────────────────────────────────────────

    # Query 1: Episodic only — vector hit rõ ràng
    separator("1/5 — Episodic only (vector hit)")
    print("User: 'Tôi đã đọc gì về Kubernetes?'")
    ctx = agent.recall("Tôi đã đọc gì về Kubernetes?", user_id="u_001")
    print(ctx)

    # Query 2: Cần profile context (topic_affinity)
    separator("2/5 — Profile context needed (topic_affinity)")
    print("User: 'Recommend đọc gì tiếp theo?'")
    ctx = agent.recall("Recommend đọc gì tiếp theo", user_id="u_001")
    print(ctx)
    if "Feast chưa sẵn sàng" in ctx:
        print("\n  [INFO] Chạy NB4 trước để Feast online store có dữ liệu.")
        print("  [INFO] Profile block sẽ hiện topic_affinity và reading_speed_wpm.")

    # Query 3: Cần fresh activity (queries_last_hour)
    separator("3/5 — Fresh activity needed (queries_last_hour)")
    print("User: 'Tôi đang quan tâm gì gần đây?'")
    ctx = agent.recall("Tôi đang quan tâm gì gần đây", user_id="u_001")
    print(ctx)

    # Query 4: Paraphrase — vector wins (không có keyword "kubernetes")
    separator("4/5 — Paraphrase query (vector wins)")
    print("User: 'Tài liệu về tự động mở rộng hạ tầng container?'")
    ctx = agent.recall("tự động mở rộng hạ tầng container", user_id="u_001")
    print(ctx)

    # Query 5: Mixed — hybrid + profile (cloud security)
    separator("5/5 — Mixed (hybrid + profile context)")
    print("User: 'Cho tôi summary về cloud security'")
    ctx = agent.recall("cloud security bảo mật đám mây", user_id="u_001")
    print(ctx)

    # ── Isolation check: u_002 không thấy memory của u_001 ──────────────────
    separator("BONUS — Isolation check")
    print("Kiểm tra u_002 KHÔNG thấy memory của u_001:")
    ctx_u2 = agent.recall("Kubernetes", user_id="u_002")
    k8s_leaked = "Kubernetes" in ctx_u2 and "u_001" not in ctx_u2
    # u_002 chỉ có 1 memory về "tài chính bí mật"
    print(ctx_u2)
    print("\n[Soft isolation via payload filter — xem ARCHITECTURE.md §Limitations]")

    print(f"\n{'='*60}")
    print("  demo.py hoàn thành — exit 0")
    print('='*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
