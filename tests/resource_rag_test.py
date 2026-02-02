"""RAG tests using real documents from the resources directory."""

from __future__ import annotations

from pathlib import Path
import unittest

from tests.rag_test_suite import build_medical_agent_workspace

RESOURCE_DIR = Path(__file__).resolve().parents[1] / "resources"


def load_resource(doc_name: str) -> str:
    return (RESOURCE_DIR / doc_name).read_text(encoding="utf-8")


class ResourceRAGTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workspace = build_medical_agent_workspace(persist_dir="./resource_test_kb")
        tooth_decay_text = load_resource("Tooth Decay.md")
        glaucoma_text = load_resource("medlineplus_Glaucoma.md")

        cls.workspace.add_document(
            doc_id="doc_tooth_decay",
            title="Tooth Decay",
            text=tooth_decay_text,
        )
        cls.workspace.add_document(
            doc_id="doc_glaucoma",
            title="Glaucoma",
            text=glaucoma_text,
        )

    def assert_answer_contains(self, answer: str, keywords: list[str]) -> None:
        missing = [keyword for keyword in keywords if keyword not in answer]
        if missing:
            joined = ", ".join(missing)
            self.fail(f"Answer missing expected keywords: {joined}")

    def assert_cited_docs(self, doc_ids: set[str], response_doc_ids: set[str]) -> None:
        if not (doc_ids & response_doc_ids):
            self.fail(f"Expected citations from {doc_ids}, got {response_doc_ids}")

    def test_tooth_decay_definition(self) -> None:
        question = "What is tooth decay?"
        response = self.workspace.query(question)

        self.assert_answer_contains(
            response.answer,
            ["damage", "enamel", "bacteria"],
        )
        cited_doc_ids = {cite.doc_id for cite in response.citations}
        self.assert_cited_docs({"doc_tooth_decay"}, cited_doc_ids)

    def test_tooth_decay_symptoms(self) -> None:
        question = "What are the symptoms of tooth decay and cavities?"
        response = self.workspace.query(question)

        self.assert_answer_contains(
            response.answer,
            ["toothache", "sensitivity", "white", "brown", "cavity"],
        )
        cited_doc_ids = {cite.doc_id for cite in response.citations}
        self.assert_cited_docs({"doc_tooth_decay"}, cited_doc_ids)

    def test_glaucoma_diagnosis(self) -> None:
        question = "How is glaucoma diagnosed?"
        response = self.workspace.query(question)

        self.assert_answer_contains(
            response.answer,
            ["dilated", "visual field", "tonometry"],
        )
        cited_doc_ids = {cite.doc_id for cite in response.citations}
        self.assert_cited_docs({"doc_glaucoma"}, cited_doc_ids)

    def test_glaucoma_treatment(self) -> None:
        question = "What are the treatments for glaucoma?"
        response = self.workspace.query(question)

        self.assert_answer_contains(
            response.answer,
            ["Prescription eye drops", "oral medicines", "laser", "surgery"],
        )
        cited_doc_ids = {cite.doc_id for cite in response.citations}
        self.assert_cited_docs({"doc_glaucoma"}, cited_doc_ids)

    def test_glaucoma_risk_factors(self) -> None:
        question = "Who is at higher risk for glaucoma?"
        response = self.workspace.query(question)

        self.assert_answer_contains(
            response.answer,
            ["over age 60", "family history", "Black"],
        )
        cited_doc_ids = {cite.doc_id for cite in response.citations}
        self.assert_cited_docs({"doc_glaucoma"}, cited_doc_ids)

    def test_hallucination_no_supporting_info(self) -> None:
        question = "青光眼患者适合做哪些高强度运动？"
        response = self.workspace.query(question)

        self.assertIn("无法回答", response.answer)
        self.assertFalse(response.citations)
        self.assertFalse(response.is_grounded)


if __name__ == "__main__":
    unittest.main()
