export const MAX_FILE_SIZE = 5 * 1024 * 1024
export const MAX_MESSAGE_LENGTH = 6000
export const REQUEST_TIMEOUT_MS = 70000
export const ACCEPTED_FILES = '.jpg,.jpeg,.png,.pdf,image/jpeg,image/png,application/pdf'

const fileTypes = { jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png', pdf: 'application/pdf' }
const statuses = new Set(['MATCHED', 'NEEDS_CLARIFICATION', 'NOT_SUPPORTED'])
const retryableErrors = new Set(['network', 'serverUnavailable', 'requestFailed', 'aiUnavailable', 'aiOverloaded', 'aiTimeout', 'aiConnection', 'rateLimit', 'invalidResponse', 'invalidRoute'])
const backendErrorCodes = {
  EMPTY_COMPLAINT: 'required', INVALID_REQUEST: 'invalidRequest', REQUEST_TOO_LARGE: 'invalidRequest', AI_NOT_CONFIGURED: 'aiUnavailable',
  ATTACHMENT_TOO_LARGE: 'fileLarge', UNSUPPORTED_FILE_TYPE: 'fileType', EMPTY_FILE: 'fileEmpty',
  INVALID_FILE_CONTENT: 'fileContent', INVALID_PDF: 'invalidPdf', ENCRYPTED_PDF: 'encryptedPdf',
  PDF_PAGE_LIMIT: 'pdfPageLimit', INVALID_IMAGE: 'invalidImage', MESSAGE_TOO_LONG: 'tooLong',
  INVALID_HISTORY: 'invalidHistory', AI_OVERLOADED: 'aiOverloaded', AI_RATE_LIMITED: 'rateLimit',
  AI_TIMEOUT: 'aiTimeout', AI_CONNECTION_ERROR: 'aiConnection', AI_UPSTREAM_ERROR: 'aiUnavailable',
  AI_INVALID_RESPONSE: 'invalidResponse', AI_INVALID_ROUTE: 'invalidRoute',
  CONVERSATION_LIMIT: 'conversationLimit', CONVERSATION_EXPIRED: 'conversationExpired',
}

function fileExtension(file) {
  return file.name.lastIndexOf('.') > 0 ? file.name.split('.').pop().toLowerCase() : ''
}

export class ApiError extends Error {
  constructor(code, { retryable = retryableErrors.has(code), requestId = '', status, backendCode } = {}) {
    super(code)
    this.name = 'ApiError'
    this.code = code
    this.retryable = retryable
    this.requestId = typeof requestId === 'string' && /^[a-zA-Z0-9._-]{1,100}$/.test(requestId) ? requestId : ''
    this.status = status
    this.backendCode = backendCode
  }
}

export function messageError(message) {
  if (!message.trim()) return 'required'
  if (message.length > MAX_MESSAGE_LENGTH) return 'tooLong'
  return ''
}

export function fileError(file) {
  if (!file) return ''
  const extension = fileExtension(file)
  if (!fileTypes[extension] || (file.type && file.type !== fileTypes[extension])) return 'fileType'
  if (file.size === 0) return 'fileEmpty'
  if (file.size > MAX_FILE_SIZE) return 'fileLarge'
  return ''
}

async function validateFileContent(file) {
  const error = fileError(file)
  if (error) throw new ApiError(error)
  const bytes = new Uint8Array(await file.slice(0, 8).arrayBuffer())
  const extension = fileExtension(file)
  const signature = extension === 'pdf' ? [37, 80, 68, 70, 45]
    : extension === 'png' ? [137, 80, 78, 71, 13, 10, 26, 10] : [255, 216, 255]
  if (!signature.every((byte, index) => bytes[index] === byte)) throw new ApiError('fileContent')
}

// Accept older responses longer than the current 10-message server window while
// keeping the original complaint and extracted attachment facts for follow-ups.
export function requestHistory(history) {
  return history.length <= 10 ? history : [history[0], ...history.slice(-9)]
}

function isHistory(history) {
  return Array.isArray(history) && history.length >= 2 && history.every((entry) => (
    entry && ['user', 'assistant'].includes(entry.role) && typeof entry.text === 'string' && entry.text.trim()
  )) && history[0].role === 'user' && history.at(-1).role === 'assistant'
}

export function normalizeResult(payload, attachment = false) {
  const routing = attachment ? payload?.routing : payload
  if (!routing || !statuses.has(routing.status)) throw new ApiError('invalidResponse')
  for (const key of ['reason', 'authority', 'service_name', 'official_url', 'link_verification']) {
    if (routing[key] != null && typeof routing[key] !== 'string') throw new ApiError('invalidResponse')
  }
  if (routing.status === 'MATCHED' && (typeof routing.service_name !== 'string' || !routing.service_name.trim())) {
    throw new ApiError('invalidResponse')
  }
  if (routing.status === 'NEEDS_CLARIFICATION') {
    if (typeof routing.question !== 'string' || !routing.question.trim()) throw new ApiError('invalidResponse')
    // Never reconstruct multipart history from the visible complaint: that loses
    // the attachment facts which only exist in routing.history[0].text.
    if (!isHistory(routing.history)) throw new ApiError(attachment ? 'attachmentHistory' : 'invalidResponse')
  }
  return {
    ...routing,
    history: routing.status === 'NEEDS_CLARIFICATION' ? routing.history : [],
    attachmentAnalysis: attachment ? payload.attachment_analysis ?? null : null,
    conversation: normalizeConversation(payload?.conversation),
  }
}

export function normalizeConversation(value) {
  if (value == null) return null // Older routing responses remain supported.
  if (typeof value !== 'object' || typeof value.token !== 'string' || !/^[A-Za-z0-9_.-]{1,350000}$/.test(value.token)
    || !Number.isSafeInteger(value.turn_count) || value.turn_count < 0) throw new ApiError('invalidResponse')
  return { token: value.token, turn_count: value.turn_count }
}

const followupIntents = new Set(['ROUTE_QUESTION', 'FACTS_UPDATE', 'REASSESS', 'PROCEDURE', 'OUT_OF_SCOPE'])

export function normalizeFollowup(payload) {
  if (!payload || typeof payload.reply !== 'string' || !payload.reply.trim() || payload.reply.length > 6000
    || !followupIntents.has(payload.intent) || typeof payload.route_changed !== 'boolean') throw new ApiError('invalidResponse')
  const conversation = normalizeConversation(payload.conversation)
  if (!conversation) throw new ApiError('invalidResponse')
  const route = normalizeResult(payload.route)
  if (route.status === 'MATCHED' && (typeof route.service_id !== 'string' || !route.service_id.trim()
    || !safeOfficialUrl(route.official_url))) throw new ApiError('invalidResponse')
  return { reply: payload.reply.trim(), intent: payload.intent, route, route_changed: payload.route_changed, conversation }
}

export function safeOfficialUrl(value) {
  if (typeof value !== 'string') return null
  try {
    const url = new URL(value)
    return url.protocol === 'https:' && !url.username && !url.password ? url.href : null
  } catch {
    return null
  }
}

function responseError(status, detail) {
  if (detail && typeof detail === 'object' && !Array.isArray(detail) && Object.hasOwn(backendErrorCodes, detail.code)) {
    return new ApiError(backendErrorCodes[detail.code], {
      retryable: typeof detail.retryable === 'boolean' ? detail.retryable : undefined,
      requestId: detail.request_id,
      status,
      backendCode: detail.code,
    })
  }
  if (status === 413) return new ApiError('fileLarge', { status })
  const known = {
    'Unsupported file type': 'fileType',
    'Empty file': 'fileEmpty',
    'File content does not match its type': 'fileContent',
    'Invalid file content': 'fileContent',
    'Complaint cannot be empty': 'required',
  }
  if (typeof detail === 'string' && Object.hasOwn(known, detail)) return new ApiError(known[detail], { status })
  if (status === 422) return new ApiError('invalidRequest', { status })
  if (status === 502 && typeof detail === 'string' && /Gemini|analysis|AI|service ID|question/i.test(detail)) return new ApiError('aiUnavailable', { status })
  if (status === 429) return new ApiError('rateLimit', { status })
  if (status === 503) return new ApiError('aiOverloaded', { status })
  if (status === 504) return new ApiError('aiTimeout', { status })
  return new ApiError(status >= 500 ? 'serverUnavailable' : 'requestFailed', { status })
}

export async function sendComplaint({ message, history = [], file = null, signal }, fetcher = fetch) {
  const error = messageError(message)
  if (error) throw new ApiError(error)
  if (file && history.length) throw new ApiError('followupAttachment')
  if (file) await validateFileContent(file)
  signal?.throwIfAborted()

  let body
  const headers = { Accept: 'application/json' }
  if (file) {
    body = new FormData()
    body.append('complaint', message.trim())
    body.append('file', file)
    // The browser supplies the multipart boundary.
  } else {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify({ message: message.trim(), history: requestHistory(history) })
  }
  let response
  try {
    response = await fetcher(`/api/${file ? 'route-with-attachment' : 'route'}`, { method: 'POST', headers, body, signal })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError('network')
  }
  let payload
  try {
    payload = await response.json()
  } catch {
    throw new ApiError(response.ok ? 'invalidResponse' : 'serverUnavailable')
  }
  if (!response.ok) throw responseError(response.status, payload?.detail)
  return normalizeResult(payload, Boolean(file))
}

export async function checkHealth(signal) {
  const response = await fetch('/api/health', { signal, headers: { Accept: 'application/json' } })
  const body = await response.json()
  if (!response.ok || body.status !== 'ok' || body.project !== 'Wajhah') throw new ApiError('serverUnavailable')
}

export async function sendFollowup({ message, conversationToken, language = 'ar', signal }, fetcher = fetch) {
  const error = messageError(message)
  if (error) throw new ApiError(error)
  if (typeof conversationToken !== 'string' || !/^[A-Za-z0-9_.-]{1,350000}$/.test(conversationToken)
    || !['ar', 'en'].includes(language)) throw new ApiError('invalidRequest', { retryable: false })
  signal?.throwIfAborted()
  let response
  try {
    response = await fetcher('/api/follow-up', {
      method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'application/json' }, signal,
      body: JSON.stringify({ message: message.trim(), conversation_token: conversationToken, language }),
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new ApiError('network')
  }
  let payload
  try { payload = await response.json() } catch { throw new ApiError(response.ok ? 'invalidResponse' : 'serverUnavailable') }
  if (!response.ok) throw responseError(response.status, payload?.detail)
  return normalizeFollowup(payload)
}
