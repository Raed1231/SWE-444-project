"""Independent Sprint 1 acceptance: real PDF parsing, synthetic mocked AI.

Copy into backend/tests, then run: python -m unittest discover -s tests -v.
Staged execution also works with PYTHONPATH set to the existing backend directory.
No real API client can be created by this suite.
"""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

import ai_gateway
import kb_routing
import main
from contracts import AttachmentAnalysis, RoutingDecision


SCHOOL_SERVICE = "svc.education.complaint"
SCHOOL_EVIDENCE = "scn.education.moe_001"
SCHOOL_COMPLAINT = "أعترض على رفع الرسوم الدراسية في المدرسة الأهلية من 10000 إلى 12000 ريال."
NEUTRAL_COMPLAINT = "أعترض على المبلغ الموضح في المرفق وأريد تقديم شكوى."
NEUTRAL_FOLLOWUP = "المبلغ رسوم إضافية فرضتها مدرسة أهلية على الرسوم الدراسية."
SCHOOL_ANALYSIS = {
    "summary": "إشعار وهمي بزيادة رسوم مدرسة أهلية من 10000 إلى 12000 ريال.",
    "extracted_facts": ["رسوم سابقة 10000 ريال", "رسوم جديدة 12000 ريال", "زيادة 20%"],
    "missing_information": [],
    "unreadable_content": "",
}
NEUTRAL_ANALYSIS = {
    "summary": "إيصال دفع تجريبي بمبلغ 250 ريال دون تحديد الجهة أو مجال الخدمة.",
    "extracted_facts": ["المبلغ 250 ريال", "اسم الجهة غير موجود في الإيصال"],
    "missing_information": ["الجهة المصدرة ونوع الخدمة المرتبطة بالمبلغ"],
    "unreadable_content": "",
}
SCHOOL_MATCH = {
    "status": "MATCHED",
    "service_id": SCHOOL_SERVICE,
    "question": "",
    "reason": "الشكوى عن رسوم مدرسة أهلية ضمن السيناريو الموثق المحدود، دون إثبات أهلية رسمية أو إرسال حكومي.",
    "evidence_ids": [SCHOOL_EVIDENCE],
}
NEUTRAL_CLARIFICATION = {
    "status": "NEEDS_CLARIFICATION",
    "service_id": None,
    "question": "ما الجهة التي فرضت المبلغ، وما نوع الخدمة المرتبطة به؟",
    "reason": "",
    "evidence_ids": [],
}


def neutral_receipt_pdf():
    """Return a reproducible, readable, domain-ambiguous PDF in memory only."""
    # Use the runtime parser's writer, so running tests needs no ReportLab extra.
    output = io.BytesIO()
    pdf = PdfWriter()
    page = pdf.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): pdf._add_object(font)}),
    })
    lines = [(725, "SYNTHETIC TEST RECEIPT"),
        (665, "Payment amount: SAR 250"),
        (625, "Issuing organization: not supplied"),
        (585, "Service or sector: not supplied"),
        (545, "The payer disputes this additional charge."),
        (465, "Fictional test data only. No personal information."),
    ]
    stream = DecodedStreamObject()
    stream.set_data("\n".join(f"BT /F1 14 Tf 55 {y} Td ({text}) Tj ET" for y, text in lines).encode("ascii"))
    page[NameObject("/Contents")] = pdf._add_object(stream)
    pdf.add_metadata({"/Title": "Synthetic neutral payment receipt", "/Author": "Wajhah synthetic acceptance tests"})
    pdf.write(output)
    return output.getvalue()


class ScriptedModels:
    def __init__(self):
        self.outputs = []
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outputs:
            raise AssertionError("Unexpected AI call; this acceptance suite forbids real network requests")
        output = self.outputs.pop(0)
        return SimpleNamespace(text=json.dumps(output, ensure_ascii=False))


class Sprint1FinalAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.fake = ScriptedModels()
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        sdk = SimpleNamespace(aio=SimpleNamespace(models=self.fake))
        for patcher in (
            patch.object(ai_gateway, "get_client", return_value=sdk),
            patch.object(ai_gateway.genai, "Client", side_effect=AssertionError("Real SDK construction forbidden")),
            patch.object(main, "USE_COMBINED_ATTACHMENTS", False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        # Resolve via the imported backend rather than a machine-specific path.
        self.fixtures = Path(kb_routing.__file__).with_name("tests") / "fixtures"

    def install(self, *outputs):
        self.assertFalse(self.fake.outputs, "A prior expected AI response was not consumed")
        self.fake.outputs = list(outputs)

    def upload_pdf(self, content, complaint, filename="synthetic-proof.pdf"):
        return self.client.post(
            "/route-with-attachment",
            data={"complaint": complaint},
            files={"file": (filename, content, "application/pdf")},
        )

    def assert_exact_official_route(self, response, service_id, nested=False):
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["routing"] if nested else response.json()
        service, authority, channel = kb_routing.validate_service(service_id)
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["service_id"], service_id)
        self.assertEqual(result["service_name"], service["name_ar"])
        self.assertEqual(result["authority"], authority["name_ar"])
        self.assertEqual(result["official_url"], channel["url"])
        self.assertEqual(result["link_verification"], channel["verification_state"])
        self.assertNotIn("evidence_ids", result)
        return result

    def assert_safe_ai_error(self, response, code):
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)
        self.assertNotIn("official_url", response.json())

    def test_tc12_real_text_pdf_uses_two_stages_and_exact_official_route(self):
        content = (self.fixtures / "text-based.pdf").read_bytes()
        reader = PdfReader(io.BytesIO(content))
        text = reader.pages[0].extract_text()
        self.assertIn("10,000", text)
        self.assertIn("12,000", text)
        self.assertEqual(len(reader.pages[0].images), 0)
        self.install(SCHOOL_ANALYSIS, SCHOOL_MATCH)
        response = self.upload_pdf(content, SCHOOL_COMPLAINT)
        self.assert_exact_official_route(response, SCHOOL_SERVICE, nested=True)
        self.assertEqual(response.json()["attachment_analysis"], SCHOOL_ANALYSIS)
        self.assertEqual(len(self.fake.calls), 2)
        self.assertFalse(self.fake.outputs)
        analysis_call, routing_call = self.fake.calls
        self.assertEqual(analysis_call["config"].response_json_schema, AttachmentAnalysis.model_json_schema())
        self.assertEqual(routing_call["config"].response_json_schema, RoutingDecision.model_json_schema())
        pdf_part = analysis_call["contents"][0]
        self.assertEqual(pdf_part.inline_data.mime_type, "application/pdf")
        self.assertEqual(pdf_part.inline_data.data, content)
        self.assertIsInstance(routing_call["contents"], str)
        conversation = json.loads(routing_call["contents"])["untrusted_conversation"]
        for fact in SCHOOL_ANALYSIS["extracted_facts"]:
            self.assertIn(fact, conversation[0]["text"])
        self.assertEqual(response.headers["X-AI-Attempts"], "2")

    def test_tc13_ambiguous_pdf_context_survives_json_followup_without_reupload(self):
        content = neutral_receipt_pdf()
        text = PdfReader(io.BytesIO(content)).pages[0].extract_text()
        self.assertIn("SAR 250", text)
        self.assertIn("not supplied", text)
        self.assertNotIn("school", text.lower())
        self.assertNotIn("tuition", text.lower())
        self.install(NEUTRAL_ANALYSIS, NEUTRAL_CLARIFICATION, SCHOOL_MATCH)
        first = self.upload_pdf(content, NEUTRAL_COMPLAINT, "neutral-receipt.pdf")
        self.assertEqual(first.status_code, 200, first.text)
        payload = first.json()
        self.assertEqual(payload["status"], "NEEDS_CLARIFICATION")
        history = payload["routing"]["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertIn(NEUTRAL_COMPLAINT, history[0]["text"])
        for fact in NEUTRAL_ANALYSIS["extracted_facts"]:
            self.assertIn(fact, history[0]["text"])
        self.assertEqual(history[-1], {"role": "assistant", "text": NEUTRAL_CLARIFICATION["question"]})
        self.assertNotIn("official_url", payload["routing"])
        self.assertEqual(len(self.fake.calls), 2)
        followup = self.client.post("/route", json={"message": NEUTRAL_FOLLOWUP, "history": history})
        self.assert_exact_official_route(followup, SCHOOL_SERVICE)
        self.assertEqual(len(self.fake.calls), 3)
        self.assertFalse(self.fake.outputs)
        self.assertEqual(sum(isinstance(call["contents"], list) for call in self.fake.calls), 1)
        last_call = self.fake.calls[-1]
        self.assertEqual(last_call["config"].response_json_schema, RoutingDecision.model_json_schema())
        sent = json.loads(last_call["contents"])["untrusted_conversation"]
        self.assertEqual(sent[:-1], history)
        self.assertEqual(sent[-1], {"role": "user", "text": NEUTRAL_FOLLOWUP})
        self.assertEqual(followup.headers["X-AI-Attempts"], "1")

    def test_tc14_every_eligible_service_returns_only_its_exact_kb_channel(self):
        self.assertEqual(len(kb_routing.SERVICES), 6)
        for service_id in kb_routing.SERVICES:
            with self.subTest(service_id=service_id):
                evidence = kb_routing.evidence_ids_for(service_id)
                self.assertTrue(evidence)
                self.install({**SCHOOL_MATCH, "service_id": service_id, "evidence_ids": [evidence[0]]})
                response = self.client.post("/route", json={"message": "شكوى اختبار وهمية لعقد البيانات فقط."})
                self.assert_exact_official_route(response, service_id)
        self.assertFalse(self.fake.outputs)

    def test_tc14_scenarioless_investment_services_keep_distinction_evidence(self):
        for service_id in ("svc.investment.service_contact", "svc.investment.integrity"):
            with self.subTest(service_id=service_id):
                self.assertEqual(kb_routing.SERVICES[service_id]["supported_scenario_ids"], [])
                self.assertIn("rr.investment.route_distinction", kb_routing.evidence_ids_for(service_id))

    def test_tc14_model_cannot_supply_an_official_url_or_authority(self):
        for extra in ({"official_url": "https://invented.invalid/service"}, {"authority": "جهة مخترعة"}):
            with self.subTest(extra=extra):
                self.install({**SCHOOL_MATCH, **extra})
                response = self.client.post("/route", json={"message": SCHOOL_COMPLAINT})
                self.assert_safe_ai_error(response, "AI_INVALID_RESPONSE")

    def test_tc14_fabricated_service_and_unrelated_evidence_are_rejected(self):
        for changes in (
            {"service_id": "svc.fabricated"},
            {"evidence_ids": ["scn.hrsd.hrsd_cc_19911_bad_treatment"]},
            {"evidence_ids": ["rr.fabricated"]},
        ):
            with self.subTest(changes=changes):
                self.install({**SCHOOL_MATCH, **changes})
                response = self.client.post("/route", json={"message": SCHOOL_COMPLAINT})
                self.assert_safe_ai_error(response, "AI_INVALID_ROUTE")

    def test_tc14_wrong_authority_and_unsafe_channel_are_rejected(self):
        channel_id = kb_routing.SERVICES[SCHOOL_SERVICE]["entry_channel_id"]
        for changes in ({"authority_id": "auth.moh"}, {"url": "http://unsafe.invalid/service"}):
            channel = {**deepcopy(kb_routing.CHANNELS[channel_id]), **changes}
            with self.subTest(changes=changes), patch.dict(kb_routing.CHANNELS, {channel_id: channel}):
                self.install(SCHOOL_MATCH)
                response = self.client.post("/route", json={"message": SCHOOL_COMPLAINT})
                self.assert_safe_ai_error(response, "AI_INVALID_ROUTE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
