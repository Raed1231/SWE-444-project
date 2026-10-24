import test from 'node:test'
import assert from 'node:assert/strict'
import {
  ApiError, normalizeConversation, normalizeFollowup, normalizeResult, sendComplaint, sendFollowup,
} from '../src/api.js'

const token = 'opaque.signed-token_0'
const nextToken = 'opaque.signed-token_1'
const conversation = { token, turn_count: 0 }
const matched = {
  status: 'MATCHED', service_id: 'synthetic.service.school', service_name: 'خدمة اختبار وهمية',
  authority: 'جهة اختبار وهمية', official_url: 'https://example.gov.sa/synthetic',
  link_verification: 'REPORTED_OBSERVED_NOT_RECHECKED', reason: 'هذه بيانات اختبار وهمية.',
}
const history = [
  { role: 'user', text: 'شكوى وهمية وحقائق المرفق المستخرجة' },
  { role: 'assistant', text: 'هل المشكلة متعلقة برسوم المدرسة؟' },
]
const clarify = { status: 'NEEDS_CLARIFICATION', question: history[1].text, history }
const unsupported = { status: 'NOT_SUPPORTED', reason: 'لا توجد خدمة مدعومة في هذا الاختبار.' }
const followup = (changes = {}) => ({
  reply: 'تظل الخدمة المقترحة مناسبة وفق المعلومات المتاحة.', intent: 'ROUTE_QUESTION',
  route: matched, route_changed: false, conversation: { token: nextToken, turn_count: 1 }, ...changes,
})
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})
const png = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], 'synthetic-proof.png', { type: 'image/png' })

test('initial text and multipart results preserve conversation tokens across all three routing statuses', () => {
  for (const route of [matched, clarify, unsupported]) {
    const text = normalizeResult({ ...route, conversation })
    assert.deepEqual(text.conversation, conversation)
    const multipart = normalizeResult({
      routing: route, conversation, attachment_analysis: { summary: 'حقائق وهمية من مرفق' },
    }, true)
    assert.deepEqual(multipart.conversation, conversation)
    assert.equal(multipart.attachmentAnalysis.summary, 'حقائق وهمية من مرفق')
    if (route.status === 'NEEDS_CLARIFICATION') assert.deepEqual(multipart.history, history)
    // Legacy responses remain usable by the original routing UI.
    assert.equal(normalizeResult(route).conversation, null)
    assert.equal(normalizeResult({ routing: route }, true).conversation, null)
  }
})

test('conversation envelopes preserve opaque tokens and reject malformed or unbounded state', () => {
  assert.deepEqual(normalizeConversation(conversation), conversation)
  const maximumEnvelope = { token: 'a'.repeat(350000), turn_count: 1 }
  assert.deepEqual(normalizeConversation(maximumEnvelope), maximumEnvelope)
  assert.equal(normalizeConversation(undefined), null)
  assert.equal(normalizeConversation(null), null)
  for (const invalid of [
    {}, [], 'raw-token', { token: '', turn_count: 0 }, { token: ' ', turn_count: 0 },
    { token: 'invalid token', turn_count: 0 }, { token: '<unsafe>', turn_count: 0 },
    { token: 'x'.repeat(350001), turn_count: 0 }, { token },
    { token, turn_count: -1 }, { token, turn_count: 1.5 },
    { token, turn_count: '1' }, { token, turn_count: Number.MAX_SAFE_INTEGER + 1 },
  ]) assert.throws(() => normalizeConversation(invalid), { code: 'invalidResponse' })
})

test('attachment routing transitions to JSON follow-up with its exact token and no binary reupload', async () => {
  const file = png()
  const initial = await sendComplaint({ message: 'شكوى وهمية مع مستند', file }, async (url, options) => {
    assert.equal(url, '/api/route-with-attachment')
    assert.equal(options.body.get('file').name, file.name)
    return json({ routing: matched, conversation, attachment_analysis: { summary: 'سياق المرفق الوهمي' } })
  })
  const controller = new AbortController()
  const result = await sendFollowup({
    message: ' هل تتغير الجهة؟ ', conversationToken: initial.conversation.token,
    language: 'ar', signal: controller.signal, file,
  }, async (url, options) => {
    assert.equal(url, '/api/follow-up')
    assert.equal(options.method, 'POST')
    assert.equal(options.signal, controller.signal)
    assert.equal(options.headers['Content-Type'], 'application/json')
    assert.equal(options.headers.Accept, 'application/json')
    assert.equal(typeof options.body, 'string')
    assert.deepEqual(JSON.parse(options.body), {
      message: 'هل تتغير الجهة؟', conversation_token: token, language: 'ar',
    })
    return json(followup())
  })
  assert.deepEqual(result.conversation, { token: nextToken, turn_count: 1 })
  assert.equal(result.route.service_id, matched.service_id)
})

test('Arabic and English follow-up messages retain their content and explicit requested language', async () => {
  for (const [language, message] of [['ar', 'بدأت الزيادة هذا الفصل، مو السنة الماضية.'], ['en', 'The increase started this term, not last year.']]) {
    await sendFollowup({ message, language, conversationToken: token }, async (_, options) => {
      assert.deepEqual(JSON.parse(options.body), { message, conversation_token: token, language })
      return json(followup({ intent: 'FACTS_UPDATE' }))
    })
  }
})

test('follow-up replies require readable bounded text, a known intent, valid routing and reusable context', () => {
  for (const changes of [
    { reply: '' }, { reply: ' \n ' }, { reply: {} }, { reply: 'x'.repeat(6001) },
    { intent: 'UNKNOWN' }, { intent: null }, { route_changed: 'false' },
    { route_changed: undefined }, { route: null }, { conversation: null },
    { conversation: undefined }, { conversation: { token: 'bad token', turn_count: 1 } },
  ]) assert.throws(() => normalizeFollowup(followup(changes)), { code: 'invalidResponse' })
  for (const invalid of [null, [], {}, { reply: 'مجرد نص دون عقد المتابعة' }]) {
    assert.throws(() => normalizeFollowup(invalid), { code: 'invalidResponse' })
  }
})

test('all follow-up intents and routing states normalize without deriving a route from answer text', () => {
  for (const intent of ['ROUTE_QUESTION', 'FACTS_UPDATE', 'REASSESS', 'PROCEDURE', 'OUT_OF_SCOPE']) {
    const result = normalizeFollowup(followup({ intent }))
    assert.equal(result.intent, intent)
    assert.equal(result.route_changed, false)
    assert.equal(result.route.service_id, matched.service_id)
  }
  for (const route of [clarify, unsupported, { ...matched, service_id: 'synthetic.service.alternative', service_name: 'خدمة بديلة وهمية' }]) {
    const result = normalizeFollowup(followup({ route, route_changed: true, intent: 'REASSESS' }))
    assert.equal(result.route.status, route.status)
    assert.equal(result.route_changed, true)
    if (route.status === 'NEEDS_CLARIFICATION') assert.deepEqual(result.route.history, history)
  }
})

test('follow-up routing rejects missing service identifiers and unsafe links while legacy initial results remain compatible', () => {
  const { service_id: omittedId, ...legacyMatched } = matched
  assert.ok(omittedId)
  assert.equal(normalizeResult(legacyMatched).status, 'MATCHED')
  for (const route of [
    legacyMatched, { ...matched, service_id: '' }, { ...matched, service_id: ' ' },
    { ...matched, service_id: {} }, { ...matched, service_name: '' },
    { ...matched, official_url: 'javascript:alert(1)' },
    { ...matched, official_url: 'http://example.gov.sa/' },
    { ...matched, official_url: 'https://user:secret@example.gov.sa/' },
    { status: 'UNKNOWN' },
  ]) assert.throws(() => normalizeFollowup(followup({ route })), { code: 'invalidResponse' })
  // Service-ID eligibility remains the backend's strict KB responsibility.
  assert.equal(normalizeFollowup(followup()).route.official_url, matched.official_url)
})

test('blank or excessive messages and missing tokens fail before dispatching a follow-up request', async () => {
  for (const [message, code] of [[' \n ', 'required'], ['x'.repeat(6001), 'tooLong']]) {
    await assert.rejects(sendFollowup({ message, conversationToken: token, language: 'ar' }, () => assert.fail('must not fetch')), { code })
  }
  await assert.rejects(sendFollowup({ message: 'سؤال وهمي', conversationToken: '', language: 'ar' }, () => assert.fail('must not fetch')), (error) => error instanceof ApiError)
})

test('provider and conversation-limit errors remain safe and never replay POST or mutate the caller’s draft/context', async () => {
  const input = { message: 'سؤال وهمي عن الخدمة', conversationToken: token, language: 'ar' }
  const before = structuredClone(input)
  for (const [status, backendCode, code, retryable] of [
    [429, 'AI_RATE_LIMITED', 'rateLimit', true], [503, 'AI_OVERLOADED', 'aiOverloaded', true],
    [504, 'AI_TIMEOUT', 'aiTimeout', true], [502, 'AI_INVALID_ROUTE', 'invalidRoute', true],
    [400, 'CONVERSATION_LIMIT', 'conversationLimit', false],
    [400, 'CONVERSATION_EXPIRED', 'conversationExpired', false],
  ]) {
    let calls = 0
    await assert.rejects(sendFollowup(input, async () => {
      calls += 1
      return json({ detail: { code: backendCode, message: 'private diagnostic', retryable, request_id: 'synthetic-test' } }, status)
    }), { code, message: code, retryable })
    assert.equal(calls, 1)
    assert.deepEqual(input, before)
  }
})

test('one explicit retry reuses the same conversation token and draft, then returns the next bounded token', async () => {
  const input = { message: ' ما المعلومات المطلوبة؟ ', conversationToken: token, language: 'ar' }
  const before = structuredClone(input)
  const bodies = []
  const fetcher = async (_, options) => {
    bodies.push(JSON.parse(options.body))
    return bodies.length === 1
      ? json({ detail: { code: 'AI_TIMEOUT', retryable: true } }, 504)
      : json(followup({ intent: 'PROCEDURE' }))
  }
  await assert.rejects(sendFollowup(input, fetcher), { code: 'aiTimeout' })
  assert.equal(bodies.length, 1)
  assert.deepEqual(input, before)
  const result = await sendFollowup(input, fetcher)
  assert.equal(bodies.length, 2)
  assert.deepEqual(bodies[0], bodies[1])
  assert.deepEqual(input, before)
  assert.deepEqual(result.conversation, { token: nextToken, turn_count: 1 })
})

test('network, malformed responses and cancellation have explicit safe outcomes without losing the current token', async () => {
  const input = { message: 'سؤال وهمي', conversationToken: token, language: 'en' }
  const before = structuredClone(input)
  await assert.rejects(sendFollowup(input, async () => { throw new TypeError('Failed to fetch') }), { code: 'network' })
  await assert.rejects(sendFollowup(input, async () => new Response('proxy unavailable', { status: 502 })), { code: 'serverUnavailable' })
  await assert.rejects(sendFollowup(input, async () => json({ reply: 'missing required fields' })), { code: 'invalidResponse' })
  const controller = new AbortController()
  controller.abort(new DOMException('Canceled', 'AbortError'))
  await assert.rejects(sendFollowup({ ...input, signal: controller.signal }, () => assert.fail('must not fetch')), { name: 'AbortError' })
  const pendingController = new AbortController()
  await assert.rejects(sendFollowup({ ...input, signal: pendingController.signal }, async (_, options) => {
    assert.equal(options.signal, pendingController.signal)
    pendingController.abort(new DOMException('Canceled', 'AbortError'))
    throw pendingController.signal.reason
  }), { name: 'AbortError' })
  assert.deepEqual(input, before)
})
