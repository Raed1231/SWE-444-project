"""Provider evidence-ID enum and unchanged local guards; all HTTP is mocked."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi.testclient import TestClient

import ai_gateway
import kb_routing as kb
import main
from contracts import AttachmentDecision, RoutingDecision


# Exact structured decision captured from synthetic live L10 sample 6 in
# diagnostic-real-l10.json. Samples 10 and 11 had the same invalid ID sequence.
CAPTURED_L10_DECISION = {
    'status': 'MATCHED',
    'service_id': 'svc.education.complaint',
    'question': '',
    'reason': 'تمت مطابقة الشكوى مع خدمة تقديم شكوى في إدارة رعاية المستفيدين بوزارة التعليم بناءً على توضيح المستخدم بأن المبلغ يخص رسومًا إضافية فرضتها مدرسة أهلية على الرسوم الدراسية، وهو سيناريو مدعوم بالدليل المعرفي الموثق.',
    'evidence_ids': ['svc.education.complaint', 'scn.education.moe_001', 'rr.education.001'],
}
VALID_MATCH = {**CAPTURED_L10_DECISION, 'evidence_ids': ['scn.education.moe_001', 'rr.education.001']}
ANALYSIS = {'summary': 'إيصال وهمي لا يحدد الجهة.', 'extracted_facts': ['مبلغ وهمي 250 ريال'], 'missing_information': ['نوع الخدمة'], 'unreadable_content': ''}
FOLLOWUP = {'message': 'المبلغ رسوم إضافية فرضتها مدرسة أهلية على الرسوم الدراسية.', 'history': [
    {'role': 'user', 'text': 'شكوى وهمية عن مبلغ 250 ريال في إيصال لا يحدد الجهة.'},
    {'role': 'assistant', 'text': 'ما الجهة التي فرضت المبلغ، وما نوع الخدمة المرتبطة به؟'},
]}


def routing_schema(schema):
    return schema['$defs']['RoutingDecision'] if '$defs' in schema and 'RoutingDecision' in schema['$defs'] else schema


class EvidenceProviderSchemaTests(unittest.IsolatedAsyncioTestCase):
    def assert_source_enum(self, schema):
        routing = routing_schema(schema)
        items = routing['properties']['evidence_ids']['items']
        self.assertEqual(items['enum'], sorted(kb.KNOWN_EVIDENCE_IDS))
        self.assertEqual(len(items['enum']), 21)
        self.assertTrue(set(items['enum']).isdisjoint(kb.SERVICES))
        self.assertTrue(all(identifier.startswith(('scn.', 'rr.')) for identifier in items['enum']))
        self.assertEqual(items['type'], 'string')
        self.assertEqual(items['maxLength'], 128)
        self.assertEqual(routing['properties']['evidence_ids']['maxItems'], 8)
        self.assertFalse(routing['additionalProperties'])

    async def test_top_level_and_nested_combined_schema_allow_exact_source_evidence_only(self):
        for model in (RoutingDecision, AttachmentDecision):
            with self.subTest(schema=model.__name__):
                self.assert_source_enum(model.model_json_schema())

    async def test_real_sdk_wire_preserves_evidence_enum_in_both_response_schemas(self):
        outputs = [VALID_MATCH, {'analysis': ANALYSIS, 'routing': VALID_MATCH}]
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            config = body['generationConfig']
            self.assertIn('responseJsonSchema', config)
            self.assertNotIn('responseSchema', config)
            self.assert_source_enum(config['responseJsonSchema'])
            output = outputs[len(requests) - 1]
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(output, ensure_ascii=False)}]}}]})

        http_client = httpx.AsyncClient(timeout=60, transport=httpx.MockTransport(handler))
        sdk = ai_gateway.genai.Client(api_key='synthetic-offline-enum-test-key', http_options=ai_gateway.types.HttpOptions(httpx_async_client=http_client, retry_options=ai_gateway.types.HttpRetryOptions(attempts=1)))
        try:
            with patch.object(ai_gateway, 'get_client', return_value=sdk):
                for model, expected in zip((RoutingDecision, AttachmentDecision), outputs):
                    with self.subTest(schema=model.__name__):
                        result = await ai_gateway.generate(model, 'Synthetic fixed policy.', 'Synthetic complaint.', 'offline_enum_wire')
                        self.assertEqual(result.model_dump(), expected)
            self.assertEqual(len(requests), 2)
        finally:
            await sdk.aio.aclose()
            sdk.close()
            await http_client.aclose()

    async def test_provider_enum_does_not_sanitize_or_replace_independent_local_guard(self):
        decision = RoutingDecision.model_validate(deepcopy(CAPTURED_L10_DECISION))
        self.assertEqual(decision.model_dump(), CAPTURED_L10_DECISION)
        self.assertNotIn(decision.evidence_ids[0], kb.KNOWN_EVIDENCE_IDS)


class EvidenceGuardReplayTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)
        self.output = deepcopy(CAPTURED_L10_DECISION)
        self.calls = []
        async def generate_content(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(text=json.dumps(self.output, ensure_ascii=False))
        sdk = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
        patcher = patch.object(ai_gateway, 'get_client', return_value=sdk)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_rejection(self, response, logs, expected_reason):
        self.assertEqual(response.status_code, 502, response.text)
        self.assertEqual(response.json()['detail']['code'], 'AI_INVALID_ROUTE')
        events = []
        for record in logs.records:
            try:
                event = json.loads(record.getMessage())
            except (ValueError, TypeError):
                continue
            if event.get('event') == 'route_rejected':
                events.append(event)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['reason_code'], expected_reason)
        self.assertEqual(event['request_id'], response.headers['x-request-id'])
        self.assertNotIn('official_url', response.json())
        self.assertEqual(len(self.calls), 1)
        return event

    def test_exact_captured_l10_decision_still_rejected_with_unknown_evidence_code(self):
        before = deepcopy(self.output)
        with self.assertLogs('wajhah', level='INFO') as logs:
            response = self.client.post('/route', json=FOLLOWUP)
        event = self.assert_rejection(response, logs, 'EVIDENCE_ID_UNKNOWN')
        self.assertEqual(event['evidence_count'], 3)
        self.assertEqual(event['unknown_evidence_count'], 1)
        self.assertEqual(event['unassociated_evidence_count'], 1)
        self.assertEqual(self.output, before)

    def test_nested_combined_invalid_decision_is_also_rejected_without_repair(self):
        self.output = {'analysis': ANALYSIS, 'routing': deepcopy(CAPTURED_L10_DECISION)}
        before = deepcopy(self.output)
        with patch.object(main, 'USE_COMBINED_ATTACHMENTS', True), self.assertLogs('wajhah', level='INFO') as logs:
            response = self.client.post('/route-with-attachment', data={'complaint': FOLLOWUP['message']}, files={'file': ('synthetic-proof.pdf', (Path(__file__).with_name('fixtures') / 'text-based.pdf').read_bytes(), 'application/pdf')})
        self.assert_rejection(response, logs, 'EVIDENCE_ID_UNKNOWN')
        self.assertEqual(self.output, before)

    def test_known_evidence_from_another_service_remains_rejected(self):
        self.output = {**VALID_MATCH, 'evidence_ids': ['scn.media.gmedia_01']}
        self.assertIn(self.output['evidence_ids'][0], kb.KNOWN_EVIDENCE_IDS)
        with self.assertLogs('wajhah', level='INFO') as logs:
            response = self.client.post('/route', json=FOLLOWUP)
        event = self.assert_rejection(response, logs, 'EVIDENCE_SERVICE_MISMATCH')
        self.assertEqual(event['unknown_evidence_count'], 0)
        self.assertEqual(event['unassociated_evidence_count'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
