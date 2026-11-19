"""文档的进程内存储。

MVP 阶段刻意不接数据库：诊断与重构都是单次请求内完成的无状态计算，唯一需要
跨请求保留的是**掩码映射表** —— 导出时要靠它把 `{{CITE_1}}` 还原成 `[12]`，
丢了它就等于丢了原文的公式与引文。

进程内存储的后果必须明确：多 worker 部署时请求可能落到没有该文档的进程上。
因此 M4 接入 SQLAlchemy 前，服务端只能单 worker 运行（诊断的并发靠
ProcessPoolExecutor 而非多 worker，见项目记忆）。get_document 找不到文档时
抛 DocumentNotFoundError，由路由层转成 404，而不是静默返回空文档。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.schemas.diagnosis import Lang
from app.schemas.document import MaskRegistry, ParsedParagraph


class DocumentNotFoundError(KeyError):
    """文档 ID 不存在。路由层据此返回 404。"""


@dataclass
class StoredDocument:
    document_id: str
    lang: Lang
    paragraphs: list[ParsedParagraph]
    registry: MaskRegistry
    locked_terms: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def paragraph(self, paragraph_id: str) -> ParsedParagraph:
        for para in self.paragraphs:
            if para.paragraph_id == paragraph_id:
                return para
        raise DocumentNotFoundError(f"段落不存在: {paragraph_id}")


class DocumentStore:
    """线程安全的进程内文档表。

    FastAPI 的同步端点跑在线程池里，多个请求可能并发读写同一张表，
    因此这里用锁而非裸 dict。
    """

    def __init__(self, max_documents: int = 64) -> None:
        self._documents: dict[str, StoredDocument] = {}
        self._lock = threading.Lock()
        self._max_documents = max_documents

    def create(
        self,
        lang: Lang,
        paragraphs: list[ParsedParagraph],
        registry: MaskRegistry,
    ) -> StoredDocument:
        document = StoredDocument(
            document_id=f"doc_{uuid.uuid4().hex[:12]}",
            lang=lang,
            paragraphs=paragraphs,
            registry=registry,
        )
        with self._lock:
            # 简单的容量上限：超出时淘汰最旧的文档，避免长跑进程内存无界增长
            if len(self._documents) >= self._max_documents:
                oldest = min(self._documents.values(), key=lambda d: d.created_at)
                del self._documents[oldest.document_id]
            self._documents[document.document_id] = document
        return document

    def get(self, document_id: str) -> StoredDocument:
        with self._lock:
            document = self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"文档不存在或已过期: {document_id}")
        return document

    def set_locked_terms(self, document_id: str, terms: list[str]) -> StoredDocument:
        document = self.get(document_id)
        with self._lock:
            document.locked_terms = list(terms)
        return document

    def clear(self) -> None:
        """测试用。"""
        with self._lock:
            self._documents.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._documents)


_STORE = DocumentStore()


def get_store() -> DocumentStore:
    """FastAPI 依赖注入入口，测试可覆盖。"""
    return _STORE
