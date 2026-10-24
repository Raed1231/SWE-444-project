"""Offline overall-budget regressions. Slow 25/60s cases require explicit opt-in."""
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from google.genai import errors
from starlette.responses import JSONResponse

import ai_gateway
import main
import request_deadline
from contracts import RoutingDecision
from request_deadline import RequestDeadline, _DeadlineExpired, remaining_seconds, request_budget

MATCH = {'status': 'MATCHED', 'service_id': 'svc.education.complaint', 'question': '', 'reason': 'شكوى وهمية عن رسوم مدرسة أهلية.', 'evidence_ids': ['scn.education.moe_001']}
ANALYSIS = {'summary': 'زيادة وهمية في رسوم مدرسة أهلية.', 'extracted_facts': ['مدرسة أهلية', 'زيادة رسوم'], 'missing_information': [], 'unreadable_content': ''}
MESSAGE = 'اختبار وهمي: أريد تقديم شكوى عن زيادة الرسوم الدراسية في مدرسة أهلية.'
FIXTURES = Path(__file__).with_name('fixtures')


class Models:
    def __init__(self, *plans):
        self.plans = list(plans)
        self.calls, self.timeouts = [], []
        self.active = 0
        self.started, self.cancelled = asyncio.Event(), asyncio.Event()

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        self.timeouts.append(kwargs['config'].http_options.timeout)
        self.active += 1
        self.started.set()
        try:
            delay, result = self.plans.pop(0)
            await asyncio.sleep(delay)
            if isinstance(result, BaseException):
                raise result
            return SimpleNamespace(text=json.dumps(result, ensure_ascii=False))
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.active -= 1


def provider_error(code):
    error_type = errors.ServerError if code >= 500 else errors.ClientError
    return error_type(code, {'error': {'code': code, 'message': 'synthetic private provider detail', 'status': 'TEST'}})


class RequestBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, models, *, budget=0.2, attachment=False):
        sdk = SimpleNamespace(aio=SimpleNamespace(models=models))
        with patch.object(ai_gateway, 'get_client', return_value=sdk), patch.object(request_deadline, 'REQUEST_TIMEOUT_SECONDS', budget), patch.object(main, 'USE_COMBINED_ATTACHMENTS', False):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
                if attachment:
                    return await client.post('/route-with-attachment', data={'complaint': MESSAGE}, files={'file': ('text-based.pdf', (FIXTURES / 'text-based.pdf').read_bytes(), 'application/pdf')})
                return await client.post('/route', json={'message': MESSAGE, 'history': []})

    def assert_timeout(self, response):
        self.assertEqual(response.status_code, 504, response.text)
        detail = response.json()['detail']
        self.assertEqual(detail['code'], 'AI_TIMEOUT')
        self.assertTrue(detail['retryable'])
        self.assertEqual(detail['request_id'], response.headers['x-request-id'])
        self.assertIn('total;dur=', response.headers['server-timing'])

    async def test_success_returns_as_soon_as_model_completes(self):
        models = Models((0.025, MATCH))
        started = time.monotonic()
        response = await self.request(models, budget=0.3)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertEqual(len(models.calls), 1)

    async def test_extraction_and_routing_inherit_one_shrinking_budget(self):
        models = Models((0.05, ANALYSIS), (0.02, MATCH))
        response = await self.request(models, budget=0.3, attachment=True)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(models.calls), 2)
        self.assertLess(models.timeouts[1], models.timeouts[0] - 35)
        self.assertLessEqual(models.timeouts[0], 300)
        self.assertEqual(response.headers['x-ai-attempts'], '2')

    async def test_combined_stages_exceed_overall_budget_cancel_routing_without_orphan(self):
        models = Models((0.06, ANALYSIS), (0.2, MATCH))
        started = time.monotonic()
        response = await self.request(models, budget=0.15, attachment=True)
        self.assert_timeout(response)
        self.assertLess(time.monotonic() - started, 0.35)
        self.assertTrue(models.cancelled.is_set())
        self.assertEqual(models.active, 0)
        await asyncio.sleep(0.03)
        self.assertEqual(len(models.calls), 2)

    async def test_retry_backoff_consumes_overall_budget_and_no_new_call_starts_after_expiry(self):
        models = Models((0.01, provider_error(503)), (0, MATCH))
        response = await self.request(models, budget=0.08)
        self.assert_timeout(response)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(models.active, 0)

    async def test_allowed_503_retry_uses_remaining_budget_after_unchanged_half_second_backoff(self):
        models = Models((0.01, provider_error(503)), (0, MATCH))
        started = time.monotonic()
        response = await self.request(models, budget=0.8)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(time.monotonic() - started, 0.5)
        self.assertEqual(len(models.calls), 2)
        self.assertLess(models.timeouts[1], models.timeouts[0] - 450)
        self.assertEqual(ai_gateway.RETRY_DELAY_SECONDS, 0.5)

    async def test_provider_504_before_deadline_is_distinguished_and_not_replayed(self):
        models = Models((0, provider_error(504)), (0, MATCH))
        with self.assertLogs('wajhah', level='WARNING') as logs:
            response = await self.request(models)
        self.assert_timeout(response)
        self.assertEqual(len(models.calls), 1)
        self.assertIn('source=provider', '\n'.join(logs.output))
        self.assertNotIn('source=request_deadline', '\n'.join(logs.output))
        self.assertNotIn('synthetic private provider detail', '\n'.join(logs.output))

    async def test_quota_and_ambiguous_network_errors_remain_single_attempt(self):
        for error, status in ((provider_error(429), 429), (httpx.ConnectError('synthetic private network detail'), 502)):
            with self.subTest(status=status):
                models = Models((0, error), (0, MATCH))
                response = await self.request(models)
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(len(models.calls), 1)

    async def test_attachment_validation_is_inside_request_deadline(self):
        models = Models((0, ANALYSIS), (0, MATCH))
        cancelled = asyncio.Event()

        async def delayed_validation(file):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.set()

        with patch.object(main, 'read_attachment', side_effect=delayed_validation):
            response = await self.request(models, budget=0.04, attachment=True)
        self.assert_timeout(response)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(models.calls, [])

    async def test_external_request_cancellation_propagates_and_cancels_model(self):
        models = Models((1, MATCH))
        task = asyncio.create_task(self.request(models, budget=0.5))
        await asyncio.wait_for(models.started.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(models.cancelled.is_set())
        self.assertEqual(models.active, 0)

    async def test_nested_budget_never_extends_original_deadline(self):
        with request_budget(0.2) as first:
            with request_budget(60) as second:
                self.assertEqual(first, second)
                self.assertLessEqual(remaining_seconds(), 0.2)

    async def test_real_sdk_wire_receives_remaining_milliseconds_not_a_fresh_60_seconds(self):
        calls = []

        def handler(request):
            calls.append(request)
            self.assertEqual(request.headers['x-server-timeout'], '1')
            for value in request.extensions['timeout'].values():
                self.assertGreater(value, 0.2)
                self.assertLess(value, 0.65)
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(MATCH)}]}}]})

        http_client = httpx.AsyncClient(timeout=60, transport=httpx.MockTransport(handler))
        sdk = ai_gateway.genai.Client(api_key='synthetic-offline-key', http_options=ai_gateway.types.HttpOptions(timeout=60000, httpx_async_client=http_client, retry_options=ai_gateway.types.HttpRetryOptions(attempts=1)))
        try:
            with request_budget(0.65):
                await asyncio.sleep(0.05)
                with patch.object(ai_gateway, 'get_client', return_value=sdk):
                    result = await ai_gateway.generate(RoutingDecision, 'Synthetic policy.', MESSAGE, 'budget_wire')
            self.assertEqual(result.status, 'MATCHED')
            self.assertEqual(len(calls), 1)
        finally:
            await sdk.aio.aclose()
            sdk.close()
            await http_client.aclose()

    async def exercise_asgi(self, app, budget, receive=None):
        messages = []
        async def default_receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        async def send(message):
            messages.append(message)
        await RequestDeadline(app, timeout=budget)({'type': 'http', 'method': 'POST', 'path': '/test', 'headers': []}, receive or default_receive, send)
        return messages

    async def test_slow_request_body_is_also_bounded(self):
        async def receive():
            await asyncio.sleep(1)
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        async def app(scope, receive, send):
            await receive()
            await JSONResponse({'status': 'unexpected success'})(scope, receive, send)
        messages = await self.exercise_asgi(app, 0.03, receive)
        self.assertEqual(messages[0]['status'], 504)

    async def test_sync_work_cannot_send_late_success_before_timer_callback_runs(self):
        async def app(scope, receive, send):
            time.sleep(0.025)
            await JSONResponse({'status': 'late success'})(scope, receive, send)
        messages = await self.exercise_asgi(app, 0.01)
        self.assertEqual(messages[0]['status'], 504)

    async def test_unrelated_inner_timeout_is_not_misclassified_as_request_expiry(self):
        async def app(scope, receive, send):
            raise TimeoutError('synthetic unrelated timeout')
        with self.assertRaisesRegex(TimeoutError, 'synthetic unrelated timeout'):
            await self.exercise_asgi(app, 1)

    async def test_expiry_after_headers_never_sends_duplicate_response_start(self):
        messages = []
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await asyncio.sleep(1)
        async def receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        async def send(message):
            messages.append(message)
        with self.assertRaises(_DeadlineExpired):
            await RequestDeadline(app, timeout=0.01)({'type': 'http', 'method': 'GET', 'path': '/', 'headers': []}, receive, send)
        self.assertEqual(len([message for message in messages if message['type'] == 'http.response.start']), 1)

    @unittest.skipUnless(os.getenv('WAJHAH_RUN_SLOW_BUDGET_TESTS') == '1', 'Opt-in real elapsed 25-second offline test')
    async def test_tc15_25_second_model_succeeds_under_real_60_second_overall_budget(self):
        models = Models((25, MATCH))
        started = time.monotonic()
        response = await self.request(models, budget=60)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(time.monotonic() - started, 25)
        self.assertLess(time.monotonic() - started, 35)

    @unittest.skipUnless(os.getenv('WAJHAH_RUN_SLOW_BUDGET_TESTS') == '1', 'Opt-in real elapsed 60-second offline test')
    async def test_tc16_model_exceeding_real_60_second_budget_is_cancelled(self):
        models = Models((65, MATCH))
        started = time.monotonic()
        response = await self.request(models, budget=60)
        self.assert_timeout(response)
        self.assertGreaterEqual(time.monotonic() - started, 59)
        self.assertLess(time.monotonic() - started, 63)
        self.assertEqual(models.active, 0)
        self.assertTrue(models.cancelled.is_set())


if __name__ == '__main__':
    unittest.main(verbosity=2)
