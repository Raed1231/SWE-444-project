import test from 'node:test'
import assert from 'node:assert/strict'
import { sendComplaint } from '../src/api.js'
import { startRequestProgress } from '../src/request-state.js'
import { translations } from '../src/i18n.js'

const matched = { status: 'MATCHED', service_name: 'خدمة اختبار', reason: 'اختبار وهمي' }
const response = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})

test('waiting is announced immediately, then at 10 and 30 seconds, with one bounded expiry', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const phases = []
  const controller = new AbortController()
  let expiries = 0
  startRequestProgress({
    onProgress: (phase) => phases.push(phase), signal: controller.signal,
    onTimeout: () => { expiries += 1; controller.abort('timeout') },
  })
  assert.deepEqual(phases, ['processing'])
  t.mock.timers.tick(9999)
  assert.deepEqual(phases, ['processing'])
  t.mock.timers.tick(1)
  assert.deepEqual(phases, ['processing', 'processingLong'])
  t.mock.timers.tick(19999)
  assert.equal(phases.length, 2)
  t.mock.timers.tick(1)
  assert.equal(phases.at(-1), 'processingExtended')
  t.mock.timers.tick(39999)
  assert.equal(expiries, 0)
  t.mock.timers.tick(1)
  assert.equal(expiries, 1)
  assert.equal(controller.signal.reason, 'timeout')
  t.mock.timers.tick(180000)
  assert.equal(expiries, 1)
  assert.equal(phases.length, 3)
})

test('a fast successful routing response settles immediately and cancels every waiting notice', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const phases = []
  let expiries = 0
  const stop = startRequestProgress({
    onProgress: (phase) => phases.push(phase), onTimeout: () => { expiries += 1 },
  })
  const result = await sendComplaint({ message: 'شكوى وهمية' }, async () => response(matched)).finally(stop)
  assert.equal(result.status, 'MATCHED')
  assert.deepEqual(phases, ['processing'])
  t.mock.timers.tick(180000)
  assert.deepEqual(phases, ['processing'])
  assert.equal(expiries, 0)
})

test('recoverable failure after a long wait preserves exact clarification input and an explicit retry starts fresh', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const phases = []
  const history = [
    { role: 'user', text: 'الشكوى الأصلية وحقائق مرفق وهمي' },
    { role: 'assistant', text: 'هل تواصلت مع الجهة؟' },
  ]
  const input = { message: ' نعم، ولم تُحل المشكلة ', history }
  const before = structuredClone(input)
  let finishRequest
  let calls = 0
  let expiries = 0
  const fetcher = async (_, options) => {
    calls += 1
    assert.deepEqual(JSON.parse(options.body), { message: input.message.trim(), history: before.history })
    if (calls === 1) return new Promise((resolve) => { finishRequest = resolve })
    return response(matched)
  }
  const callbacks = { onProgress: (phase) => phases.push(phase), onTimeout: () => { expiries += 1 } }
  const stopFirst = startRequestProgress(callbacks)
  const first = sendComplaint(input, fetcher).finally(stopFirst)
  t.mock.timers.tick(31000)
  assert.equal(phases.at(-1), 'processingExtended')
  finishRequest(response({ detail: { code: 'AI_TIMEOUT', retryable: true } }, 504))
  await assert.rejects(first, { code: 'aiTimeout', retryable: true })
  t.mock.timers.tick(90000)
  assert.equal(expiries, 0)
  assert.equal(calls, 1)
  assert.deepEqual(input, before)
  assert.equal(input.history, history)

  const stopRetry = startRequestProgress(callbacks)
  assert.equal(phases.at(-1), 'processing')
  const result = await sendComplaint(input, fetcher).finally(stopRetry)
  assert.equal(result.status, 'MATCHED')
  assert.equal(calls, 2)
  assert.deepEqual(input, before)
  t.mock.timers.tick(90000)
  assert.equal(phases.at(-1), 'processing')
  assert.equal(expiries, 0)
})

test('browser cancellation and component cleanup stop notices without scheduling another request', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  for (const reason of ['user', 'cleanup']) {
    const phases = []
    const controller = new AbortController()
    let expiries = 0
    const stop = startRequestProgress({
      onProgress: (phase) => phases.push(phase), signal: controller.signal,
      onTimeout: () => { expiries += 1 },
    })
    t.mock.timers.tick(15000)
    controller.abort(reason)
    stop()
    stop()
    t.mock.timers.tick(180000)
    assert.deepEqual(phases, ['processing', 'processingLong'])
    assert.equal(expiries, 0)
  }
})

test('an already canceled request starts no waiting lifecycle', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const controller = new AbortController()
  controller.abort('cleanup')
  const stop = startRequestProgress({
    signal: controller.signal,
    onProgress: () => assert.fail('canceled request cannot announce progress'),
    onTimeout: () => assert.fail('canceled request cannot expire again'),
  })
  stop()
  t.mock.timers.tick(180000)
})

test('cleanup from a prior request cannot cancel a later request’s notices', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] })
  const firstPhases = []
  const secondPhases = []
  const stopFirst = startRequestProgress({ onProgress: (phase) => firstPhases.push(phase), onTimeout: () => assert.fail('old request expired') })
  stopFirst()
  const stopSecond = startRequestProgress({ onProgress: (phase) => secondPhases.push(phase), onTimeout: () => assert.fail('new request expired') })
  stopFirst()
  t.mock.timers.tick(30000)
  assert.deepEqual(firstPhases, ['processing'])
  assert.deepEqual(secondPhases, ['processing', 'processingLong', 'processingExtended'])
  stopSecond()
  t.mock.timers.tick(90000)
})

test('all waiting announcements have Arabic and English text without fabricated percentages', () => {
  for (const language of ['ar', 'en']) {
    for (const phase of ['processing', 'processingLong', 'processingExtended']) {
      assert.equal(typeof translations[language][phase], 'string')
      assert.ok(translations[language][phase].trim())
      assert.doesNotMatch(translations[language][phase], /[%٪]/)
    }
  }
})
