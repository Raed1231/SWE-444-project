"""Question-specific replies over consecutive turns; no real provider requests.

The selector is scripted, so these tests verify server composition, history,
bounds and guards. Live selection/classification remains separate acceptance.
"""
import base64
import hashlib
import hmac
import json
from pathlib import Path
import sys
from typing import get_args
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from pydantic import ValidationError

import ai_gateway
import conversation
import main
import test_conversation as existing


CHAINS = {
    'ar': [
        'وش الأوراق اللي أجهزها لشكوى الرسوم؟',
        'يعني أرجع للموقع الرسمي عشان أتأكد؟',
        'وش أتحقق منه هناك بالضبط؟',
        'هل فاتورة الرسوم إلزامية بالتحديد؟',
        'ما عندي فاتورة، وش أقدر أرتب من المعلومات المتاحة عندي؟',
        'والدخول هل هو عن طريق نفاذ؟',
        'طيب اختصر لي موضوع الأوراق والتجهيز بدون ما تعيد قائمة الدخول والإقرار.',
    ],
    'en': [
        'What papers should I prepare for my tuition-fee complaint?',
        'So should I check the official website to confirm?',
        'What exactly should I check there?',
        'Is the tuition invoice specifically mandatory?',
        'I do not have an invoice. What available information can I organize?',
        'Is sign-in through Nafath?',
        'Briefly recap the papers and preparation, without repeating sign-in and declaration.',
    ],
}
PLANS = [
    existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents', 'document_requirement']),
    existing.plan('PROCEDURE', 'OFFICIAL_CHANNEL', fact_ids=['channel']),
    existing.plan('PROCEDURE', 'EXACT_STEPS', fact_ids=['document_check']),
    existing.plan('PROCEDURE', 'EXACT_STEPS', fact_ids=['document_requirement']),
    existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['document_alternatives', 'preparation_advice', 'document_requirement']),
    existing.plan('PROCEDURE', 'EXACT_STEPS', fact_ids=['login', 'login_method']),
    existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents', 'preparation_advice']),
]


class ConversationQualityTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, token, *responses, message='وش المقصود؟', language='ar'):
        models = existing.ScriptedModels(*responses)
        with patch.object(ai_gateway, 'get_client', return_value=existing.fake_client(models)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
                response = await client.post('/follow-up', json={
                    'message': message, 'conversation_token': token, 'language': language,
                })
        return response, models

    def ok(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(set(data), {'reply', 'intent', 'route', 'route_changed', 'conversation'})
        self.assertTrue(data['reply'].strip())
        self.assertLessEqual(len(data['reply']), 6000)
        conversation.ConversationRef.model_validate(data['conversation'])
        return data

    async def chain(self, language):
        token = existing.seed()
        initial = conversation._open(token)
        route = conversation._public_route(initial)
        replies = []
        for index, (message, output) in enumerate(zip(CHAINS[language], PLANS)):
            with self.subTest(language=language, turn=index + 1):
                before = conversation._open(token)
                response, models = await self.request(token, output, message=message, language=language)
                data = self.ok(response)
                reply = data['reply']
                self.assertFalse(data['route_changed'])
                self.assertEqual(data['route'], route)
                self.assertEqual(len(models.calls), 1, 'A procedural question must not reroute.')
                payload = json.loads(models.calls[0]['contents'])
                self.assertEqual(payload['untrusted_message'], message)
                self.assertEqual(payload['untrusted_recent_dialogue'], [item.model_dump() for item in before.recent])
                self.assertIn(existing.ORIGINAL, payload['untrusted_fact_candidates'].values())
                self.assertEqual(payload['discussed'], before.discussed)
                if index:
                    self.assertEqual(payload['untrusted_recent_dialogue'][-2:], [
                        {'role': 'user', 'text': CHAINS[language][index - 1]},
                        {'role': 'assistant', 'text': replies[-1]},
                    ])
                state = conversation._open(data['conversation']['token'])
                self.assertEqual(state.original, existing.ORIGINAL)
                self.assertEqual(state.updates, [], 'Procedural questions are not complaint facts.')
                self.assertEqual(state.turn_count, index + 1)
                self.assertLessEqual(len(state.recent), 10)
                self.assertLessEqual(sum(len(item.text) for item in state.recent), 12000)
                self.assertLessEqual(len(state.discussed), 15)
                self.assertEqual(len(state.discussed), len(set(state.discussed)))
                # The original duplicated login/documents/declaration checklist
                # cannot satisfy any of these focused questions.
                self.assertNotIn('•', reply)
                for unrelated in (('قبول الإقرار', 'شروط الأهلية') if language == 'ar' else ('Accept the declaration', 'eligibility rules')):
                    self.assertNotIn(unrelated, reply)
                if index != 5:
                    self.assertNotIn('تسجيل الدخول' if language == 'ar' else 'Sign-in is documented', reply)
                if index in (2, 3, 5):
                    self.assertNotIn('ما عندي خطوات صفحات' if language == 'ar' else 'page-by-page steps', reply)
                if index == 0:
                    self.assertIn('مستندات أو مرفقات داعمة' if language == 'ar' else 'supporting documents or attachments', reply)
                    self.assertIn('قائمة تفصيلية' if language == 'ar' else 'does not name them', reply)
                    self.assertNotIn('preparation_advice', state.discussed)
                    self.assertNotIn('هذا المستند بعينه' if language == 'ar' else 'this particular document', reply)
                    self.assertNotIn('document_requirement', state.discussed)
                elif index == 1:
                    self.assertIn('رابط الخدمة الرسمي' if language == 'ar' else 'official service link', reply)
                    self.assertNotIn('قائمة' if language == 'ar' else 'document list', reply)
                elif index == 2:
                    for relevant in (('أسماء المستندات', 'إلزامي', 'الصيغ والأحجام') if language == 'ar' else ('which documents', 'mandatory', 'file types and size limits')):
                        self.assertIn(relevant, reply)
                    self.assertNotIn('5 MB', reply)
                    self.assertNotIn('PDF', reply)
                elif index == 3:
                    self.assertIn('هذا المستند بعينه إلزامي' if language == 'ar' else 'this particular document is mandatory', reply)
                    self.assertIn('ما أقدر أؤكد قبول بديل' if language == 'ar' else 'cannot confirm that an alternative', reply)
                    self.assertNotIn('الصيغ والأحجام' if language == 'ar' else 'file types', reply)
                elif index == 4:
                    self.assertIn('preparation_advice', state.discussed)
                    self.assertIn('إذا ما عندك المستند' if language == 'ar' else 'If you do not have the document', reply)
                    self.assertIn('الرسوم محل الاعتراض' if language == 'ar' else 'disputed fees', reply)
                    self.assertIn('تاريخ إبلاغك' if language == 'ar' else 'when you were notified', reply)
                    self.assertIn('قبول البديل مو موثّق' if language == 'ar' else 'acceptance of alternatives is not documented', reply)
                    self.assertEqual(reply.count('كتجهيز عام' if language == 'ar' else 'as general preparation'), 1)
                    self.assertNotIn('هذا المستند بعينه' if language == 'ar' else 'this particular document', reply)
                    self.assertNotIn('للتجهيز فقط' if language == 'ar' else 'For preparation only', reply)
                    self.assertIn('document_alternatives', state.discussed)
                    self.assertNotIn('بديل مقبول' if language == 'ar' else 'accepted substitute', reply)
                elif index == 5:
                    self.assertIn('نفاذ' if language == 'ar' else 'Nafath', reply)
                    self.assertIn('غير محسومين' if language == 'ar' else 'not established', reply)
                    self.assertNotIn('المستندات' if language == 'ar' else 'documents', reply)
                    self.assertEqual(reply.count('تسجيل الدخول موثّق' if language == 'ar' else 'Sign-in is documented'), 1)
                else:
                    self.assertLess(len(reply), len(conversation._catalog(state, language)['preparation_advice']))
                    self.assertEqual(reply.count('للتجهيز فقط' if language == 'ar' else 'For preparation only'), 1)
                    self.assertIn('قائمة الأوراق الإلزامية مو متوفرة' if language == 'ar' else 'required document list', reply)
                    self.assertIn('الرسوم محل الاعتراض' if language == 'ar' else 'disputed fees', reply)
                    self.assertNotEqual(reply, replies[0])
                    self.assertNotEqual(reply, replies[4])
                    self.assertNotIn('إذا ما عندك المستند' if language == 'ar' else 'If you do not have the document', reply)
                if language == 'en':
                    self.assertNotRegex(reply, r'[\u0600-\u06ff]')
                replies.append(reply)
                token = data['conversation']['token']
        self.assertEqual(len(replies), 7)
        self.assertEqual(len(set(replies)), 7)

    async def test_seven_consecutive_arabic_followups_answer_the_latest_question(self):
        await self.chain('ar')

    async def test_seven_consecutive_english_followups_answer_the_latest_question(self):
        await self.chain('en')

    async def test_granular_facts_override_broad_exact_steps_topic(self):
        for key, marker in [('document_check', 'أسماء المستندات'), ('document_requirement', 'هذا المستند بعينه'), ('login_method', 'نفاذ')]:
            with self.subTest(key=key):
                response, _ = await self.request(existing.seed(), existing.plan('PROCEDURE', 'EXACT_STEPS', fact_ids=[key]))
                reply = self.ok(response)['reply']
                self.assertIn(marker, reply)
                self.assertNotIn('ما عندي خطوات صفحات', reply)
        response, _ = await self.request(existing.seed(), existing.plan('PROCEDURE', 'EXACT_STEPS'))
        self.assertIn('ما عندي خطوات صفحات', self.ok(response)['reply'])

    async def test_unavailable_or_forged_answer_id_rejected_locally_without_state_commit(self):
        token = existing.seed()
        for key in ('preparation', 'limitations', 'invented_requirement'):
            with self.subTest(key=key):
                response, models = await self.request(token, existing.plan('PROCEDURE', 'PREPARATION', fact_ids=[key]))
                self.assertEqual(response.status_code, 502, response.text)
                self.assertEqual(response.json()['detail']['code'], 'AI_INVALID_RESPONSE')
                self.assertNotIn('conversation', response.json())
                self.assertEqual(conversation._open(token).discussed, [])
                schema = models.calls[0]['config'].response_json_schema
                enum = schema['properties']['fact_ids']['items']['enum']
                self.assertNotIn('preparation', enum)
                self.assertNotIn('limitations', enum)
                self.assertNotIn('invented_requirement', enum)
                self.assertIn('documents', enum)
                self.assertEqual(len(models.calls), 1)

    async def test_provider_error_does_not_mark_answer_discussed_and_manual_retry_is_full_answer(self):
        token = existing.seed()
        response, models = await self.request(token, existing.provider_error(429))
        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(len(models.calls), 1)
        self.assertEqual(conversation._open(token).discussed, [])
        self.assertNotIn('conversation', response.json())
        response, _ = await self.request(token, existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents']))
        data = self.ok(response)
        self.assertIn('المسار الموثق يذكر', data['reply'])
        self.assertEqual(conversation._open(data['conversation']['token']).discussed, ['documents'])

    async def test_only_rendered_answers_are_marked_and_duplicates_do_not_grow_metadata(self):
        token = existing.seed()
        for output in (
            existing.plan(topic='COMPLAINT_FACTS', fact_ids=['documents'], context_refs=['fact.0']),
            existing.plan('OUT_OF_SCOPE', 'UNSUPPORTED', fact_ids=['documents']),
            existing.plan(topic='HYPOTHETICAL_ROUTE', fact_ids=['documents']),
        ):
            response, _ = await self.request(token, output)
            data = self.ok(response)
            self.assertEqual(conversation._open(data['conversation']['token']).discussed, [])
            token = data['conversation']['token']
        for _ in range(6):
            response, _ = await self.request(token, existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents', 'documents']))
            data = self.ok(response)
            token = data['conversation']['token']
            self.assertEqual(conversation._open(token).discussed, ['documents'])

    async def test_route_change_resets_explained_metadata_but_fact_ledger_survives(self):
        token = existing.seed()
        response, _ = await self.request(token, existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents']))
        token = self.ok(response)['conversation']['token']
        self.assertEqual(conversation._open(token).discussed, ['documents'])
        correction = 'التصحيح: شكواي عن سوء تعامل موظف مركز الاتصال 19911.'
        response, _ = await self.request(token, existing.plan('FACTS_UPDATE'), existing.matched('svc.hrsd.call_center_19911'), message=correction)
        data = self.ok(response)
        self.assertTrue(data['route_changed'])
        state = conversation._open(data['conversation']['token'])
        self.assertEqual(state.discussed, [])
        self.assertEqual(state.original, existing.ORIGINAL)
        self.assertEqual(state.updates[-1].text, correction)

    async def test_old_signed_session_without_discussed_stays_usable(self):
        data = conversation._open(existing.seed()).model_dump()
        data.pop('discussed')
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode()
        encoded = base64.urlsafe_b64encode(payload).decode().rstrip('=')
        signature = hmac.new(conversation._SIGNING_KEY, payload, hashlib.sha256).hexdigest()
        token = encoded + '.' + signature
        self.assertEqual(conversation._open(token).discussed, [])
        response, _ = await self.request(token, existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents']))
        self.assertIn('المسار الموثق يذكر', self.ok(response)['reply'])

    def test_explained_metadata_is_bounded_to_fifteen_known_answer_purposes(self):
        facts = list(get_args(conversation.Fact))
        self.assertEqual(len(facts), 15)
        state = conversation._open(existing.seed())
        state.discussed = facts
        restored = conversation._open(conversation._seal(state)['token'])
        self.assertEqual(restored.discussed, facts)
        with self.assertRaises(ValidationError):
            conversation.SessionState.model_validate({**state.model_dump(), 'discussed': [*facts, 'documents']})
        with self.assertRaises(ValidationError):
            conversation.SessionState.model_validate({**state.model_dump(), 'discussed': ['untrusted_free_text']})

    def test_split_catalog_preserves_other_services_documented_prerequisites(self):
        cases = {
            'svc.media.report': ['Prepare complaint narrative/content', 'Choose complaint classification, complaint type, and offender type'],
            'svc.health.technical_access': ['Select provider type, city, and operator before ticket type becomes actionable'],
            'svc.investment.service_contact': ['Choose Reason For Contact = Complaint and complete contact form fields'],
        }
        for service_id, requirements in cases.items():
            with self.subTest(service=service_id):
                state = conversation._open(existing.seed(existing.matched(service_id)))
                catalog = conversation._catalog(state, 'en')
                for requirement in requirements:
                    self.assertIn(requirement, catalog['preparation'])
        school = conversation._open(existing.seed())
        self.assertNotIn('preparation', conversation._catalog(school, 'en'))

    async def test_known_login_is_separate_from_unknown_provider_and_document_mandatory_claim(self):
        response, _ = await self.request(existing.seed(), existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['login']), language='en')
        reply = self.ok(response)['reply']
        self.assertIn('documented service path', reply)
        self.assertIn('Authenticated login', reply)
        self.assertNotIn('Nafath', reply)
        self.assertNotIn('documents', reply)
        response, _ = await self.request(existing.seed(), existing.plan('PROCEDURE', 'PREPARATION', fact_ids=['documents']), language='en')
        reply = self.ok(response)['reply']
        self.assertIn('does not name them', reply)
        self.assertIn('mandatory for every case', reply)
        self.assertNotIn('Authenticated login', reply)

    async def test_rejected_reroute_keeps_existing_guards_and_answer_memory(self):
        state = conversation._open(existing.seed())
        state.discussed = ['documents']
        token = conversation._seal(state)['token']
        invalid = {**existing.SCHOOL_MATCH, 'evidence_ids': ['svc.education.complaint']}
        with self.assertLogs('wajhah', level='WARNING') as logs:
            response, models = await self.request(token, existing.plan('FACTS_UPDATE'), invalid, message='واقعة إضافية عن المدرسة.')
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()['detail']['code'], 'AI_INVALID_ROUTE')
        self.assertIn('EVIDENCE_ID_UNKNOWN', '\n'.join(logs.output))
        self.assertEqual(conversation._open(token).discussed, ['documents'])
        self.assertEqual(conversation._open(token).updates, [])
        self.assertEqual(len(models.calls), 2)


if __name__ == '__main__':
    unittest.main()
