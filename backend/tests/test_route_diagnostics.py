"""Synthetic offline classification/privacy tests; never contact an AI provider."""
import ast
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kb_routing as kb

SERVICE = 'svc.education.complaint'
PRIVATE = 'SYNTHETIC_PRIVATE_PERSON_KEY_123'
MATCH = {'status': 'MATCHED', 'service_id': SERVICE, 'question': '', 'reason': 'سبب اختبار وهمي', 'evidence_ids': ['scn.education.moe_001']}


class KnowledgeRouteDiagnosticTests(unittest.TestCase):
    def assert_reason(self, reason, *, identifier=SERVICE, patches=()):
        with ExitStack() as stack:
            for mapping, updates in patches:
                stack.enter_context(patch.dict(mapping, updates))
            with self.assertRaises(kb.RouteValidationError) as captured:
                kb.validate_service(identifier)
        error = captured.exception
        self.assertIsInstance(error, ValueError)
        self.assertEqual(error.reason_code, reason)
        self.assertEqual(error.args, (reason,))
        self.assertEqual(str(error), reason)
        self.assertNotIn(PRIVATE, repr(error))
        self.assertNotIn(identifier, str(error))
        return reason

    def test_all_exact_validate_service_rejection_branches_have_stable_private_free_codes(self):
        channel_id = kb.SERVICES[SERVICE]['entry_channel_id']
        service, channel = deepcopy(kb.SERVICES[SERVICE]), deepcopy(kb.CHANNELS[channel_id])
        observed = {self.assert_reason('KB_UNKNOWN_SERVICE', identifier=PRIVATE)}
        cases = [
            ('KB_SERVICE_COVERAGE', {'coverage_status': PRIVATE}),
            ('KB_SERVICE_GUIDANCE', {'guidance_suitability': PRIVATE}),
            ('KB_SERVICE_READINESS', {'routing_readiness': {'status': PRIVATE}}),
            ('KB_SERVICE_SCOPE', {'scope_description': None}),
            ('KB_AUTHORITY_OR_CHANNEL_MISSING', {'authority_id': PRIVATE}),
            ('KB_AUTHORITY_OR_CHANNEL_MISSING', {'entry_channel_id': PRIVATE}),
            ('KB_SCENARIO_REFERENCE', {'supported_scenario_ids': [PRIVATE]}),
            ('KB_RULE_REFERENCE', {'routing_rule_ids': [PRIVATE]}),
        ]
        for reason, updates in cases:
            with self.subTest(reason=reason, fields=list(updates)):
                observed.add(self.assert_reason(reason, patches=[(kb.SERVICES, {SERVICE: {**service, **updates}})]))
        channel_cases = [
            ('KB_CHANNEL_AUTHORITY', {'authority_id': PRIVATE}),
            ('KB_CHANNEL_STATUS', {'official_status': PRIVATE}),
            ('KB_CHANNEL_STATUS', {'usage_status': PRIVATE}),
            ('KB_CHANNEL_URL', {'url': None}),
            ('KB_CHANNEL_URL', {'url': 'https://example.test/' + PRIVATE + ' '}),
            ('KB_CHANNEL_URL_PARSE', {'url': 'https://[' + PRIVATE}),
            ('KB_CHANNEL_URL_POLICY', {'url': 'http://example.test/' + PRIVATE}),
            ('KB_CHANNEL_URL_POLICY', {'url': 'https://' + PRIVATE + ':password@example.test/'}),
            ('KB_CHANNEL_URL_POLICY', {'url': 'https:///' + PRIVATE}),
            ('KB_CHANNEL_PORT', {'url': 'https://example.test:' + PRIVATE + '/'}),
            ('KB_CHANNEL_PORT', {'url': 'https://example.test:70000/' + PRIVATE}),
        ]
        for reason, updates in channel_cases:
            with self.subTest(reason=reason, fields=list(updates)):
                observed.add(self.assert_reason(reason, patches=[(kb.CHANNELS, {channel_id: {**channel, **updates}})]))
        module = ast.parse(Path(kb.__file__).read_text())
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_service')
        source_codes = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                self.assertNotEqual(node.exc.func.id, 'ValueError', 'Unclassified explicit ValueError remains')
                if node.exc.func.id == 'RouteValidationError':
                    source_codes.add(ast.literal_eval(node.exc.args[0]))
        self.assertEqual(observed, source_codes)
        self.assertEqual(len(observed), 14)

    def test_schema_valid_matched_unknown_service_reaches_typed_guard(self):
        from contracts import RoutingDecision
        decision = RoutingDecision.model_validate({**MATCH, 'service_id': PRIVATE})
        self.assertEqual(self.assert_reason('KB_UNKNOWN_SERVICE', identifier=decision.service_id), 'KB_UNKNOWN_SERVICE')

    def test_scenario_and_rule_known_but_wrong_service_or_authority_are_classified(self):
        scenario_id, rule_id = 'scn.education.moe_001', 'rr.education.001'
        for field in ('service_id', 'authority_id'):
            with self.subTest(kind='scenario', field=field):
                record = {**deepcopy(kb._SCENARIOS[scenario_id]), field: PRIVATE}
                self.assert_reason('KB_SCENARIO_REFERENCE', patches=[(kb._SCENARIOS, {scenario_id: record})])
            with self.subTest(kind='rule', field=field):
                record = {**deepcopy(kb._RULES[rule_id]), field: PRIVATE}
                self.assert_reason('KB_RULE_REFERENCE', patches=[(kb._RULES, {rule_id: record})])

    def test_invalid_reason_input_cannot_become_a_loggable_private_message(self):
        for value in (PRIVATE, ['private'], None):
            with self.subTest(type=type(value).__name__):
                error = kb.RouteValidationError(value)
                self.assertEqual(error.reason_code, 'KB_ROUTE_INVALID')
                self.assertEqual(error.args, ('KB_ROUTE_INVALID',))
                self.assertNotIn(PRIVATE, repr(error))

    def test_known_evidence_is_immutable_exact_source_union(self):
        self.assertIsInstance(kb.KNOWN_EVIDENCE_IDS, frozenset)
        self.assertEqual(kb.KNOWN_EVIDENCE_IDS, frozenset(kb._SCENARIOS) | frozenset(kb._RULES))
        self.assertNotIn(PRIVATE, kb.KNOWN_EVIDENCE_IDS)

    def test_success_returns_identical_source_objects_and_preserves_knowledge(self):
        before = deepcopy(kb.KB)
        for service_id in kb.SERVICES:
            with self.subTest(service_id=service_id):
                service, authority, channel = kb.validate_service(service_id)
                self.assertIs(service, kb.SERVICES[service_id])
                self.assertIs(authority, kb.AUTHORITIES[service['authority_id']])
                self.assertIs(channel, kb.CHANNELS[service['entry_channel_id']])
        self.assertEqual(kb.KB, before)


class RouteDiagnosticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        import ai_gateway
        from fastapi.testclient import TestClient
        cls.main, cls.gateway, cls.TestClient = main, ai_gateway, TestClient

    def setUp(self):
        self.client = self.TestClient(self.main.app)
        self.addCleanup(self.client.close)
        self.output = dict(MATCH)
        async def generate_content(**kwargs):
            return SimpleNamespace(text=json.dumps(self.output, ensure_ascii=False))
        sdk = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
        patcher = patch.object(self.gateway, 'get_client', return_value=sdk)
        patcher.start()
        self.addCleanup(patcher.stop)

    def call(self):
        return self.client.post('/route', json={'message': PRIVATE, 'history': []})

    def event(self, logs, event_name):
        events = []
        for record in logs.records:
            try:
                event = json.loads(record.getMessage())
            except (ValueError, TypeError):
                continue
            if event.get('event') == event_name:
                events.append(event)
        self.assertEqual(len(events), 1, logs.output)
        return events[0]

    def assert_safe_rejection(self, reason):
        with self.assertLogs('wajhah', level='INFO') as logs:
            response = self.call()
        self.assertEqual(response.status_code, 502, response.text)
        detail = response.json()['detail']
        self.assertEqual(detail['code'], 'AI_INVALID_ROUTE')
        event = self.event(logs, 'route_rejected')
        self.assertEqual(event['reason_code'], reason)
        self.assertEqual(event['request_id'], detail['request_id'])
        self.assertEqual(event['request_id'], response.headers['x-request-id'])
        for field in ('evidence_count', 'unknown_evidence_count', 'unassociated_evidence_count'):
            self.assertIs(type(event[field]), int)
        self.assertIsInstance(event['branch'], str)
        for private in (PRIVATE, self.output['reason'], self.output['service_id'], *self.output['evidence_ids']):
            self.assertNotIn(private, '\n'.join(logs.output))
            self.assertNotIn(private, response.text)
        return event

    def test_unknown_service_has_exact_safe_reason_and_correlated_request(self):
        self.output = {**MATCH, 'service_id': PRIVATE, 'reason': PRIVATE}
        self.assert_safe_rejection('KB_UNKNOWN_SERVICE')

    def test_unknown_evidence_is_distinct_from_known_unassociated_evidence(self):
        self.output = {**MATCH, 'evidence_ids': [PRIVATE], 'reason': PRIVATE}
        event = self.assert_safe_rejection('EVIDENCE_ID_UNKNOWN')
        self.assertEqual(event['unknown_evidence_count'], 1)
        self.output = {**MATCH, 'evidence_ids': ['scn.media.gmedia_01'], 'reason': PRIVATE}
        event = self.assert_safe_rejection('EVIDENCE_SERVICE_MISMATCH')
        self.assertEqual(event['unknown_evidence_count'], 0)
        self.assertEqual(event['unassociated_evidence_count'], 1)

    def test_ineligible_known_service_has_classified_private_free_metadata_reason(self):
        service = {**deepcopy(kb.SERVICES[SERVICE]), 'coverage_status': PRIVATE}
        with patch.dict(kb.SERVICES, {SERVICE: service}):
            self.assert_safe_rejection('KB_SERVICE_COVERAGE')

    def test_validated_success_is_unchanged_and_correlated_without_model_values_in_log(self):
        with self.assertLogs('wajhah', level='INFO') as logs:
            response = self.call()
        self.assertEqual(response.status_code, 200, response.text)
        service, authority, channel = kb.validate_service(SERVICE)
        from conversation import ConversationRef
        payload = response.json()
        reference = ConversationRef.model_validate(payload.pop('conversation'))
        self.assertEqual(reference.turn_count, 0)
        self.assertTrue(reference.token)
        self.assertEqual(payload, {'status': 'MATCHED', 'service_id': SERVICE, 'service_name': service['name_ar'], 'authority': authority['name_ar'], 'official_url': channel['url'], 'link_verification': channel.get('verification_state'), 'reason': MATCH['reason']})
        event = self.event(logs, 'route_validated')
        self.assertEqual(event['request_id'], response.headers['x-request-id'])
        for private in (PRIVATE, SERVICE, MATCH['reason'], *MATCH['evidence_ids']):
            self.assertNotIn(private, '\n'.join(logs.output))


if __name__ == '__main__':
    unittest.main(verbosity=2)
