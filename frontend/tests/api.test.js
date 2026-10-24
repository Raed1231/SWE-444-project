import test from 'node:test'
import assert from 'node:assert/strict'
import { ApiError, MAX_FILE_SIZE, fileError, messageError, normalizeResult, requestHistory, safeOfficialUrl, sendComplaint } from '../src/api.js'
import { translations } from '../src/i18n.js'

const matched = { status: 'MATCHED', service_name: 'خدمة اختبار', authority: 'جهة اختبار', official_url: 'https://example.gov.sa/service', reason: 'خدمة مناسبة' }
const original = 'شكوى المستخدم: شراء منتج\nمعلومات مستخرجة من المرفق: رقم الفاتورة 123'
const attachmentHistory = [{ role: 'user', text: original }, { role: 'assistant', text: 'هل تواصلت مع المتجر؟' }]
const png = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])], 'receipt.PNG', { type: 'image/png' })
const response = (value, status = 200) => new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json' } })

test('blank and oversized complaints fail before any request', async () => {
  assert.equal(messageError(' \n '), 'required')
  assert.equal(messageError('أ'.repeat(6001)), 'tooLong')
  await assert.rejects(sendComplaint({ message: ' ' }, () => assert.fail('must not fetch')), { code: 'required' })
})

test('file boundaries accept the backend limit and reject empty, large, unsupported and mismatched types', () => {
  assert.equal(fileError({ name: 'proof.pdf', type: 'application/pdf', size: MAX_FILE_SIZE }), '')
  assert.equal(fileError({ name: 'proof.pdf', type: '', size: MAX_FILE_SIZE + 1 }), 'fileLarge')
  assert.equal(fileError(new File([], 'proof.jpg', { type: 'image/jpeg' })), 'fileEmpty')
  assert.equal(fileError({ name: 'proof.svg', type: 'image/svg+xml', size: 20 }), 'fileType')
  assert.equal(fileError({ name: 'proof.png', type: 'application/pdf', size: 20 }), 'fileType')
  assert.equal(fileError({ name: 'pdf', type: 'application/pdf', size: 20 }), 'fileType')
  assert.equal(fileError({ name: '.pdf', type: 'application/pdf', size: 20 }), 'fileType')
})

test('valid JPG, JPEG, PNG and PDF signatures all use the multipart contract', async () => {
  for (const [name, type, bytes] of [
    ['sample.jpg', 'image/jpeg', [255, 216, 255]],
    ['sample.JPEG', 'image/jpeg', [255, 216, 255]],
    ['sample.png', 'image/png', [137, 80, 78, 71, 13, 10, 26, 10]],
    ['sample.pdf', 'application/pdf', [37, 80, 68, 70, 45]],
  ]) {
    const result = await sendComplaint({ message: 'اختبار', file: new File([new Uint8Array(bytes)], name, { type }) }, async (url) => {
      assert.equal(url, '/api/route-with-attachment')
      return response({ routing: matched, attachment_analysis: { summary: 'test' } })
    })
    assert.equal(result.status, 'MATCHED')
  }
})

test('renamed files with invalid signatures are rejected before sending', async () => {
  await assert.rejects(sendComplaint({ message: 'شكوى', file: new File(['not a pdf'], 'fake.pdf', { type: 'application/pdf' }) }, () => assert.fail('must not fetch')), { code: 'fileContent' })
})

test('text requests use the same-origin JSON route with history separate from the new message', async () => {
  const result = await sendComplaint({ message: ' نعم ', history: attachmentHistory }, async (url, options) => {
    assert.equal(url, '/api/route')
    assert.equal(options.headers['Content-Type'], 'application/json')
    assert.deepEqual(JSON.parse(options.body), { message: 'نعم', history: attachmentHistory })
    return response(matched)
  })
  assert.equal(result.status, 'MATCHED')
  assert.equal(result.service_name, matched.service_name)
})

test('multipart results retain nested history and follow up through JSON with the exact extracted context', async () => {
  const first = await sendComplaint({ message: 'شكوى شراء', file: png() }, async (url, options) => {
    assert.equal(url, '/api/route-with-attachment')
    assert.equal(options.headers['Content-Type'], undefined)
    assert.equal(options.body.get('complaint'), 'شكوى شراء')
    assert.equal(options.body.get('file').name, 'receipt.PNG')
    assert.equal(options.body.get('history'), null)
    return response({ status: 'NEEDS_CLARIFICATION', routing: { status: 'NEEDS_CLARIFICATION', question: attachmentHistory[1].text, history: attachmentHistory }, attachment_analysis: { summary: 'فاتورة شراء' } })
  })
  assert.deepEqual(first.history, attachmentHistory)
  assert.equal(first.attachmentAnalysis.summary, 'فاتورة شراء')
  const final = await sendComplaint({ message: 'نعم', history: first.history }, async (url, options) => {
    assert.equal(url, '/api/route')
    assert.equal(JSON.parse(options.body).history[0].text, original)
    return response(matched)
  })
  assert.equal(final.status, 'MATCHED')
})

test('the original attachment context survives more than five clarification rounds', () => {
  const history = [...attachmentHistory, ...Array.from({ length: 12 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', text: `turn ${i}` }))]
  const sent = requestHistory(history)
  assert.equal(sent.length, 10)
  assert.equal(sent.slice(-10)[0].text, original)
  assert.deepEqual(sent.slice(1), history.slice(-9))
  assert.equal(history.length, 14)
})

test('missing multipart clarification history fails visibly instead of losing attachment context', () => {
  assert.throws(() => normalizeResult({ routing: { status: 'NEEDS_CLARIFICATION', question: 'متى؟' } }, true), { code: 'attachmentHistory' })
})

test('attachments cannot silently restart a conversation during a follow-up', async () => {
  await assert.rejects(sendComplaint({ message: 'توضيح', history: attachmentHistory, file: png() }, () => assert.fail('must not fetch')), { code: 'followupAttachment' })
})

test('all routing statuses normalize for text and multipart, including a second clarification', () => {
  for (const routing of [matched, { status: 'NOT_SUPPORTED', reason: 'خارج التغطية' }, { status: 'NEEDS_CLARIFICATION', question: 'متى؟', history: attachmentHistory }]) {
    assert.equal(normalizeResult(routing).status, routing.status)
    assert.equal(normalizeResult({ routing, attachment_analysis: {} }, true).status, routing.status)
  }
  assert.throws(() => normalizeResult({ status: 'SOMETHING_ELSE' }), { code: 'invalidResponse' })
  assert.throws(() => normalizeResult({ status: 'MATCHED' }), { code: 'invalidResponse' })
  assert.throws(() => normalizeResult({ ...matched, reason: { invalid: 'object' } }), { code: 'invalidResponse' })
  assert.throws(() => normalizeResult({ status: 'NEEDS_CLARIFICATION', question: ' ', history: attachmentHistory }), { code: 'invalidResponse' })
})

test('external links permit HTTPS only and reject scripts or embedded credentials', () => {
  assert.equal(safeOfficialUrl(matched.official_url), matched.official_url)
  for (const value of ['javascript:alert(1)', 'data:text/html,hello', 'http://example.com', '//example.com', 'https://user:secret@example.com', null]) assert.equal(safeOfficialUrl(value), null)
})

test('backend validation and AI errors become safe, translated error codes', async () => {
  for (const [status, detail, code] of [[413, 'File exceeds 5 MB', 'fileLarge'], [400, 'Invalid file content', 'fileContent'], [422, [], 'invalidRequest'], [502, 'Gemini request failed', 'aiUnavailable'], [429, 'busy', 'rateLimit'], [500, 'secret backend diagnostic', 'serverUnavailable']]) {
    await assert.rejects(sendComplaint({ message: 'شكوى' }, async () => response({ detail }, status)), { code, message: code })
  }
})

test('HTML proxy failures and incomplete successful responses have readable failures', async () => {
  await assert.rejects(sendComplaint({ message: 'شكوى' }, async () => new Response('proxy unavailable', { status: 502 })), { code: 'serverUnavailable' })
  await assert.rejects(sendComplaint({ message: 'شكوى' }, async () => response(null)), { code: 'invalidResponse' })
})

test('structured backend failures preserve safe codes and retry metadata without exposing diagnostics', async () => {
  const cases = [
    ['ATTACHMENT_TOO_LARGE', 'fileLarge', 413], ['UNSUPPORTED_FILE_TYPE', 'fileType', 400],
    ['EMPTY_FILE', 'fileEmpty', 400], ['INVALID_FILE_CONTENT', 'fileContent', 400],
    ['INVALID_PDF', 'invalidPdf', 400], ['ENCRYPTED_PDF', 'encryptedPdf', 400],
    ['PDF_PAGE_LIMIT', 'pdfPageLimit', 400], ['INVALID_IMAGE', 'invalidImage', 400],
    ['MESSAGE_TOO_LONG', 'tooLong', 422], ['INVALID_HISTORY', 'invalidHistory', 422],
    ['AI_OVERLOADED', 'aiOverloaded', 503], ['AI_RATE_LIMITED', 'rateLimit', 429],
    ['AI_TIMEOUT', 'aiTimeout', 504], ['AI_CONNECTION_ERROR', 'aiConnection', 502],
    ['AI_UPSTREAM_ERROR', 'aiUnavailable', 502], ['AI_INVALID_RESPONSE', 'invalidResponse', 502],
    ['AI_INVALID_ROUTE', 'invalidRoute', 502],
  ]
  for (const [backendCode, code, status] of cases) {
    for (const retryable of [false, true]) {
      await assert.rejects(sendComplaint({ message: 'اختبار' }, async () => response({ detail: {
        code: backendCode, message: 'private provider diagnostic', retryable, request_id: 'test-request-123',
      } }, status)), (error) => {
        assert.ok(error instanceof ApiError)
        assert.equal(error.message, code)
        assert.equal(error.code, code)
        assert.equal(error.backendCode, backendCode)
        assert.equal(error.status, status)
        assert.equal(error.retryable, retryable)
        assert.equal(error.requestId, 'test-request-123')
        return true
      })
    }
  }
})

test('unknown structured codes, prototype names and unsafe request IDs cannot become displayed diagnostics', async () => {
  for (const code of ['UNKNOWN', '__proto__', 'constructor']) {
    await assert.rejects(sendComplaint({ message: 'اختبار' }, async () => response({ detail: {
      code, message: 'private provider diagnostic', retryable: true, request_id: '<script>unsafe</script>',
    } }, 400)), { code: 'requestFailed', message: 'requestFailed', requestId: '' })
  }
  assert.equal(new ApiError('aiTimeout').retryable, true)
  assert.equal(new ApiError('invalidPdf').retryable, false)
  assert.equal(new ApiError('aiTimeout', { requestId: 'x'.repeat(101) }).requestId, '')
})

test('network failures do not automatically repeat POSTs and an explicit retry retains clarification context', async () => {
  const input = { message: ' نعم ', history: structuredClone(attachmentHistory) }
  const before = structuredClone(input)
  let calls = 0
  await assert.rejects(sendComplaint(input, async () => {
    calls += 1
    throw new TypeError('Failed to fetch')
  }), { code: 'network', retryable: true })
  assert.equal(calls, 1)
  assert.deepEqual(input, before)
  await sendComplaint(input, async (_, options) => {
    calls += 1
    assert.deepEqual(JSON.parse(options.body), { message: 'نعم', history: before.history })
    return response(matched)
  })
  assert.equal(calls, 2)
  assert.deepEqual(input, before)
})

test('recoverable multipart failure permits explicit retry with the same complaint and file', async () => {
  const file = png()
  const input = { message: 'شكوى مدرسة خاصة', file }
  let calls = 0
  const fetcher = async (_, options) => {
    calls += 1
    assert.equal(options.body.get('complaint'), input.message)
    assert.equal(options.body.get('file').name, file.name)
    assert.deepEqual(await options.body.get('file').arrayBuffer(), await file.arrayBuffer())
    return calls === 1
      ? response({ detail: { code: 'AI_OVERLOADED', retryable: true } }, 503)
      : response({ routing: matched, attachment_analysis: { summary: 'اختبار' } })
  }
  await assert.rejects(sendComplaint(input, fetcher), { code: 'aiOverloaded' })
  assert.equal(calls, 1)
  const result = await sendComplaint(input, fetcher)
  assert.equal(result.status, 'MATCHED')
  assert.equal(calls, 2)
  assert.equal(input.file, file)
  assert.equal(input.message, 'شكوى مدرسة خاصة')
})

test('cancellation prevents dispatch and remains distinct from a connection failure', async () => {
  const controller = new AbortController()
  controller.abort(new DOMException('Canceled', 'AbortError'))
  await assert.rejects(sendComplaint({ message: 'اختبار', file: png(), signal: controller.signal }, () => assert.fail('must not fetch')), { name: 'AbortError' })
  const pendingController = new AbortController()
  await assert.rejects(sendComplaint({ message: 'اختبار', signal: pendingController.signal }, async (_, options) => {
    assert.equal(options.signal, pendingController.signal)
    pendingController.abort(new DOMException('Canceled', 'AbortError'))
    throw pendingController.signal.reason
  }), { name: 'AbortError' })
})

test('both interface languages provide matching keys and all error translations', () => {
  assert.deepEqual(Object.keys(translations.ar).sort(), Object.keys(translations.en).sort())
  for (const language of Object.values(translations)) {
    for (const key of ['required', 'tooLong', 'fileLarge', 'fileEmpty', 'fileContent', 'fileType', 'invalidPdf', 'encryptedPdf', 'pdfPageLimit', 'invalidImage', 'invalidHistory', 'invalidRequest', 'aiUnavailable', 'aiOverloaded', 'aiTimeout', 'aiConnection', 'invalidRoute', 'retrySubmission', 'serverUnavailable', 'rateLimit', 'requestFailed', 'network', 'invalidResponse', 'attachmentHistory', 'followupAttachment', 'timeout', 'canceled']) assert.ok(language[key], key)
  }
})
