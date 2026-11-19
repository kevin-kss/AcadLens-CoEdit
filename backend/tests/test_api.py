"""API 层单测。

LLM 端点通过依赖覆盖注入 ScriptedClient，完全不碰网络。
重点覆盖：掩码无损往返、404 语义、实时重算只影响单段、模型不可用时返回 503
而非伪造空结果。
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.v1.assist import get_llm_client
from app.main import app
from app.schemas.api import DocumentResponse, ParagraphView
from app.services.document_store import get_store
from app.services.llm.client import ChatMessage, LLMError, ScriptedClient

TEXT = (
    "在当今社会，多头自注意力机制的研究日益重要。值得注意的是，该方法具有多维度的应用价值。"
    "此外，其显著提升了系统的整体效能 [12]。综上所述，本研究具有重要的理论意义。\n\n"
    "我们在 ImageNet-1k 上训练 ResNet-50。批大小 256，学习率 0.1。"
    "Top-1 准确率为 76.3%。去掉残差连接后降至 71.8%，公式见 $E=mc^2$。"
)

STAGE1_JSON = json.dumps(
    {
        "facts": [{"subject": "多头自注意力机制", "predicate": "提升", "obj": "效能"}],
        "slots": [{"kind": "metric", "prompt": "请补充具体指标数值"}],
    },
    ensure_ascii=False,
)

GOOD_DRAFT = "多头自注意力机制把 Top-1 准确率提升到 76.3% {{CITE_1}}。"


@pytest.fixture(autouse=True)
def _clean_store() -> Iterator[None]:
    get_store().clear()
    yield
    get_store().clear()
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def create_doc(client: TestClient, text: str = TEXT) -> DocumentResponse:
    response = client.post("/api/v1/documents/parse", json={"text": text, "lang": "zh"})
    assert response.status_code == 201
    return DocumentResponse.model_validate(response.json())


class TestHealth:
    def test_health(self, client: TestClient):
        assert client.get("/health").json() == {"status": "ok"}


class TestParse:
    def test_returns_paragraphs_and_diagnosis(self, client: TestClient):
        doc = create_doc(client)
        assert len(doc.paragraphs) == 2
        for para in doc.paragraphs:
            assert 0 <= para.diagnosis.overall_risk <= 100

    def test_citations_and_formulas_masked(self, client: TestClient):
        doc = create_doc(client)
        masked = " ".join(p.masked_text for p in doc.paragraphs)
        assert "[12]" not in masked
        assert "$E=mc^2$" not in masked
        assert "{{CITE_" in masked

    def test_plain_text_is_restored(self, client: TestClient):
        doc = create_doc(client)
        plain = " ".join(p.plain_text for p in doc.paragraphs)
        assert "[12]" in plain
        assert "$E=mc^2$" in plain

    def test_ai_paragraph_scores_higher(self, client: TestClient):
        doc = create_doc(client)
        risks = [p.diagnosis.overall_risk for p in doc.paragraphs]
        assert risks[0] > risks[1]

    def test_summary_consistent(self, client: TestClient):
        doc = create_doc(client)
        assert doc.summary.paragraph_count == len(doc.paragraphs)

    def test_empty_text_rejected(self, client: TestClient):
        response = client.post("/api/v1/documents/parse", json={"text": "", "lang": "zh"})
        assert response.status_code == 422

    def test_issue_offsets_valid_against_masked_text(self, client: TestClient):
        """前端下划线定位直接依赖这些偏移。"""
        doc = create_doc(client)
        for para in doc.paragraphs:
            text = para.masked_text
            for issue in para.diagnosis.span_issues:
                assert text[issue.start_char : issue.end_char] == issue.matched_text


class TestGetAndNotFound:
    def test_get_document(self, client: TestClient):
        doc_id = create_doc(client).document_id
        assert client.get(f"/api/v1/documents/{doc_id}").status_code == 200

    def test_unknown_document_404(self, client: TestClient):
        assert client.get("/api/v1/documents/doc_missing").status_code == 404

    def test_unknown_paragraph_404(self, client: TestClient):
        doc_id = create_doc(client).document_id
        response = client.put(
            f"/api/v1/documents/{doc_id}/paragraphs/p999",
            json={"masked_text": "新文本。"},
        )
        assert response.status_code == 404


class TestUpdateParagraph:
    def test_recomputes_single_paragraph(self, client: TestClient):
        doc = create_doc(client)
        doc_id = doc.document_id
        before = doc.paragraphs[0].diagnosis.overall_risk

        response = client.put(
            f"/api/v1/documents/{doc_id}/paragraphs/p0",
            json={"masked_text": "多头自注意力机制把准确率提升到 76.3% {{CITE_1}}。"},
        )
        assert response.status_code == 200
        view = ParagraphView.model_validate(response.json())
        assert view.diagnosis.overall_risk < before

    def test_other_paragraph_untouched(self, client: TestClient):
        doc = create_doc(client)
        doc_id = doc.document_id
        p1_before = doc.paragraphs[1].diagnosis.overall_risk

        client.put(
            f"/api/v1/documents/{doc_id}/paragraphs/p0",
            json={"masked_text": "简短文本。"},
        )
        after = DocumentResponse.model_validate(client.get(f"/api/v1/documents/{doc_id}").json())
        assert after.paragraphs[1].diagnosis.overall_risk == p1_before

    def test_update_persists(self, client: TestClient):
        doc_id = create_doc(client).document_id
        client.put(
            f"/api/v1/documents/{doc_id}/paragraphs/p0",
            json={"masked_text": "替换后的文本。"},
        )
        doc = DocumentResponse.model_validate(client.get(f"/api/v1/documents/{doc_id}").json())
        assert doc.paragraphs[0].masked_text == "替换后的文本。"


class TestLockedTerms:
    def test_set_terms(self, client: TestClient):
        doc_id = create_doc(client).document_id
        response = client.put(
            f"/api/v1/documents/{doc_id}/terms", json={"terms": ["多头自注意力机制"]}
        )
        assert response.status_code == 200

    def test_terms_reach_the_guard(self, client: TestClient):
        """锁定术语必须真的作用到门禁上，否则这个端点只是装饰。"""
        doc_id = create_doc(client).document_id
        client.put(f"/api/v1/documents/{doc_id}/terms", json={"terms": ["多头自注意力机制"]})

        # 草稿丢掉该术语 -> 门禁应拦下
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(
            default_response="该注意力方法提升了准确率 {{CITE_1}}。"
        )
        response = client.post(
            f"/api/v1/assist/restructure?document_id={doc_id}",
            json={"paragraph_id": "p0", "slots": [], "strategies": ["plain"]},
        )
        assert response.status_code == 200
        assert not response.json()["candidates"][0]["guard_result"]["passed"]


class TestExport:
    def test_roundtrip_is_lossless(self, client: TestClient):
        """非正文无损率 100% 是硬验收指标。"""
        doc_id = create_doc(client).document_id
        exported = client.get(f"/api/v1/documents/{doc_id}/export").text
        assert "[12]" in exported
        assert "$E=mc^2$" in exported
        assert "{{" not in exported

    def test_export_unknown_404(self, client: TestClient):
        assert client.get("/api/v1/documents/doc_nope/export").status_code == 404


class TestAssistDeconstruct:
    def test_returns_facts_and_slots(self, client: TestClient):
        doc_id = create_doc(client).document_id
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(responses=[STAGE1_JSON])
        response = client.post(
            f"/api/v1/assist/deconstruct?document_id={doc_id}",
            json={"paragraph_id": "p0"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["facts"]
        assert payload["slots"][0]["prompt"] == "请补充具体指标数值"

    def test_model_unavailable_returns_503(self, client: TestClient):
        """不伪造空结果：空槽位列表会让用户以为本段无需补数据。"""
        doc_id = create_doc(client).document_id

        class Dead:
            def complete(
                self,
                messages: list[ChatMessage],
                *,
                temperature: float = 0.2,
                max_tokens: int | None = None,
                response_format_json: bool = False,
            ) -> str:
                raise LLMError("connection refused")

        app.dependency_overrides[get_llm_client] = lambda: Dead()
        response = client.post(
            f"/api/v1/assist/deconstruct?document_id={doc_id}",
            json={"paragraph_id": "p0"},
        )
        assert response.status_code == 503
        assert "模型服务不可用" in response.json()["detail"]

    def test_unknown_paragraph_404(self, client: TestClient):
        doc_id = create_doc(client).document_id
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(responses=[STAGE1_JSON])
        response = client.post(
            f"/api/v1/assist/deconstruct?document_id={doc_id}",
            json={"paragraph_id": "nope"},
        )
        assert response.status_code == 404


class TestAssistRestructure:
    def test_returns_three_candidates(self, client: TestClient):
        doc_id = create_doc(client).document_id
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(
            default_response=GOOD_DRAFT
        )
        response = client.post(
            f"/api/v1/assist/restructure?document_id={doc_id}",
            json={"paragraph_id": "p0", "slots": []},
        )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["candidates"]) == 3
        assert {c["strategy"] for c in payload["candidates"]} == {
            "plain",
            "varied",
            "blind_outline",
        }

    def test_risk_comparison_present(self, client: TestClient):
        doc_id = create_doc(client).document_id
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(
            default_response=GOOD_DRAFT
        )
        response = client.post(
            f"/api/v1/assist/restructure?document_id={doc_id}",
            json={"paragraph_id": "p0", "slots": []},
        )
        candidate = response.json()["candidates"][0]
        assert candidate["risk_before"] > 0
        assert "risk_after" in candidate

    def test_rejected_candidates_are_returned_not_dropped(self, client: TestClient):
        """静默丢弃会让用户以为系统只生成了一版。"""
        doc_id = create_doc(client).document_id
        client.put(f"/api/v1/documents/{doc_id}/terms", json={"terms": ["多头自注意力机制"]})
        app.dependency_overrides[get_llm_client] = lambda: ScriptedClient(
            default_response="完全改写，术语没了。"
        )
        response = client.post(
            f"/api/v1/assist/restructure?document_id={doc_id}",
            json={"paragraph_id": "p0", "slots": []},
        )
        candidates = response.json()["candidates"]
        assert len(candidates) == 3
        assert all(c["reject_reason"] for c in candidates)
        assert all(c["masked_text"] for c in candidates)

    def test_filled_slot_values_used(self, client: TestClient):
        doc_id = create_doc(client).document_id
        scripted = ScriptedClient(default_response=GOOD_DRAFT)
        app.dependency_overrides[get_llm_client] = lambda: scripted
        client.post(
            f"/api/v1/assist/restructure?document_id={doc_id}",
            json={
                "paragraph_id": "p0",
                "slots": [
                    {
                        "slot_id": "s1",
                        "kind": "metric",
                        "prompt": "指标",
                        "value": "76.3%",
                    }
                ],
                "strategies": ["plain"],
            },
        )
        assert "76.3%" in scripted.last_prompt


class TestOpenAPI:
    def test_schema_generates(self, client: TestClient):
        """OpenAPI 生成失败通常意味着某个响应模型无法序列化。"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        for expected in (
            "/api/v1/documents/parse",
            "/api/v1/documents/{document_id}",
            "/api/v1/assist/deconstruct",
            "/api/v1/assist/restructure",
        ):
            assert expected in paths
