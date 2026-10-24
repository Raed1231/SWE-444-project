import { useEffect, useRef, useState } from 'react'
import { ACCEPTED_FILES, MAX_MESSAGE_LENGTH, checkHealth, fileError, messageError, safeOfficialUrl, sendComplaint, sendFollowup } from './api'
import { translations } from './i18n'
import { startRequestProgress } from './request-state'
import './App.css'

const invalidAttachmentCodes = new Set(['fileLarge', 'fileType', 'fileEmpty', 'fileContent', 'invalidPdf', 'encryptedPdf', 'pdfPageLimit', 'invalidImage'])

function Icon({ name, size = 22, className = '' }) {
  const paths = {
    compass: <><circle cx="12" cy="12" r="9" /><path d="m16 8-2.5 5.5L8 16l2.5-5.5Z" /></>,
    upload: <path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />,
    file: <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Zm0 0v6h6M8 13h8m-8 4h5" />,
    arrow: <path d="M19 12H5m6-6-6 6 6 6" />,
    external: <path d="M14 3h7v7m0-7L11 13M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5" />,
    check: <path d="m5 12 4 4L19 6" />,
    question: <><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3m0 4h.01" /><circle cx="12" cy="12" r="10" /></>,
    info: <><circle cx="12" cy="12" r="10" /><path d="M12 11v6m0-10h.01" /></>,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    globe: <><circle cx="12" cy="12" r="9" /><ellipse cx="12" cy="12" rx="4" ry="9" /><path d="M3 12h18" /></>,
    lock: <><rect x="5" y="10" width="14" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 5v2" /></>,
    reset: <path d="M3 4v6h6m-6 0a9 9 0 1 1 1 8" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" className={className}>{paths[name] || paths.compass}</svg>
}

function AnalysisValue({ value }) {
  if (value === null || value === undefined || value === '' || value === false) return null
  if (Array.isArray(value)) return <ul>{value.map((item, index) => <li key={index}><AnalysisValue value={item} /></li>)}</ul>
  if (typeof value === 'object') return <dl>{Object.entries(value).map(([key, item]) => <div key={key}><dt dir="auto">{key}</dt><dd><AnalysisValue value={item} /></dd></div>)}</dl>
  return <span dir="auto">{String(value)}</span>
}

function ResultCard({ result, pending, processingMessage, t, onReset, resetDisabled }) {
  const status = result?.status
  const officialUrl = safeOfficialUrl(result?.official_url)
  return <section className={`result-card ${status ? `result-${status.toLowerCase()}` : ''}`} aria-labelledby="result-heading" aria-busy={pending}>
    <div className="section-kicker"><span className="small-line" />{t.resultLabel}</div>
    <div aria-live="polite" aria-atomic="true">
      {pending || !result ? <div className="empty-result"><div className="result-orbit"><Icon name="compass" size={56} /><span className="orbit-dot" /></div><h2 id="result-heading">{pending ? t.loading : t.emptyTitle}</h2><p>{pending ? processingMessage : t.emptyText}</p>{pending ? <span className="loading-dots" aria-hidden="true">•••</span> : <><div className="empty-divider" /><p className="small-copy">{t.emptyFoot}</p></>}</div>
        : <div className="result-content">
          <div className="result-status"><span className="status-icon"><Icon name={status === 'MATCHED' ? 'check' : status === 'NEEDS_CLARIFICATION' ? 'question' : 'info'} /></span><h2 id="result-heading">{status === 'MATCHED' ? t.matched : status === 'NEEDS_CLARIFICATION' ? t.clarification : t.unsupported}</h2></div>
          {status === 'MATCHED' && <>
            <dl className="service-details"><div><dt>{t.service}</dt><dd dir="auto">{result.service_name}</dd></div>{result.authority && <div><dt>{t.authority}</dt><dd dir="auto">{result.authority}</dd></div>}</dl>
            {result.reason && <div className="reason"><h3>{t.reason}</h3><p dir="auto">{result.reason}</p></div>}
            {officialUrl ? <><a className="button primary official-button" href={officialUrl} target="_blank" rel="noopener noreferrer">{t.officialLink}<Icon name="external" size={18} /></a><p className="link-note">{t.linkNote}</p></> : <p className="notice">{t.unavailableLink}</p>}
            {officialUrl && result.link_verification !== 'VERIFIED' && <p className="notice">{t.unverified}</p>}
          </>}
          {status === 'NEEDS_CLARIFICATION' && <><p className="clarification-question" dir="auto">{result.question}</p><p className="muted small-copy">{t.replyHint}</p></>}
          {status === 'NOT_SUPPORTED' && <><p>{t.unsupportedText}</p>{result.reason && <p className="reason" dir="auto">{result.reason}</p>}</>}
          {status !== 'NEEDS_CLARIFICATION' && <button className="text-button start-again" type="button" onClick={onReset} disabled={resetDisabled}><Icon name="reset" size={17} />{t.newComplaint}</button>}
        </div>}
    </div><div className="result-disclaimer"><Icon name="info" size={18} /><p>{t.resultNotice}</p></div>
  </section>
}

function initialLanguage() {
  try { return localStorage.getItem('wajhah-language') === 'en' ? 'en' : 'ar' } catch { return 'ar' }
}

function App() {
  const [language, setLanguage] = useState(initialLanguage)
  const [message, setMessage] = useState('')
  const [file, setFile] = useState(null)
  const [errors, setErrors] = useState({})
  const [errorCode, setErrorCode] = useState('')
  const [errorRetryable, setErrorRetryable] = useState(false)
  const [noticeCode, setNoticeCode] = useState('')
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [transcript, setTranscript] = useState([])
  const [analysis, setAnalysis] = useState(null)
  const [attachedName, setAttachedName] = useState('')
  const [pending, setPending] = useState(false)
  const [conversation, setConversation] = useState(null)
  const [followupDraft, setFollowupDraft] = useState('')
  const [followupMessages, setFollowupMessages] = useState([])
  const [followupPending, setFollowupPending] = useState(false)
  const [followupError, setFollowupError] = useState('')
  const [followupRetryable, setFollowupRetryable] = useState(false)
  const [followupNotice, setFollowupNotice] = useState('')
  const [followupTrimmed, setFollowupTrimmed] = useState(false)
  const [progressPhase, setProgressPhase] = useState('processing')
  const [health, setHealth] = useState('checking')
  const [healthAttempt, setHealthAttempt] = useState(0)
  const inputRef = useRef(null)
  const fileRef = useRef(null)
  const removeFileRef = useRef(null)
  const controllerRef = useRef(null)
  const busyRef = useRef(false)
  const followupRef = useRef(null)
  const followupLogRef = useRef(null)
  const t = translations[language]
  const isFollowup = result?.status === 'NEEDS_CLARIFICATION'
  const finished = Boolean(result && !isFollowup)
  const currentStep = result ? result.status === 'MATCHED' ? 2 : 1 : 0
  const currentFileError = errors.file || fileError(file)
  const canRetry = errorRetryable || noticeCode === 'timeout'
  const busy = pending || followupPending

  useEffect(() => {
    document.documentElement.lang = language
    document.documentElement.dir = language === 'ar' ? 'rtl' : 'ltr'
    document.title = `${t.brand} | ${t.brandNote}`
    try { localStorage.setItem('wajhah-language', language) } catch { /* Storage may be disabled. */ }
  }, [language, t.brand, t.brandNote])

  useEffect(() => {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 8000)
    checkHealth(controller.signal).then(() => setHealth('online')).catch(() => {
      if (!controller.signal.aborted || controller.signal.reason?.name === 'AbortError') setHealth('offline')
    })
    return () => { controller.abort('cleanup'); clearTimeout(timeout) }
  }, [healthAttempt])
  useEffect(() => () => controllerRef.current?.abort('cleanup'), [])
  useEffect(() => {
    const log = followupLogRef.current
    if (log) log.scrollTop = log.scrollHeight
  }, [followupMessages])

  function reset() {
    if (busyRef.current) return
    setMessage(''); setFile(null); setErrors({}); setErrorCode(''); setErrorRetryable(false); setNoticeCode('')
    setResult(null); setHistory([]); setTranscript([]); setAnalysis(null); setAttachedName('')
    setConversation(null); setFollowupDraft(''); setFollowupMessages([]); setFollowupError(''); setFollowupRetryable(false); setFollowupNotice(''); setFollowupTrimmed(false)
    if (fileRef.current) fileRef.current.value = ''
    requestAnimationFrame(() => inputRef.current?.focus())
  }

  async function submit(event) {
    event.preventDefault()
    if (busyRef.current || finished) return
    const validation = { message: messageError(message), file: currentFileError }
    setErrors(validation); setErrorCode(''); setErrorRetryable(false); setNoticeCode('')
    if (validation.message || validation.file) {
      if (validation.message) inputRef.current?.focus()
      else removeFileRef.current?.focus()
      return
    }
    busyRef.current = true
    setPending(true)
    const controller = new AbortController()
    controllerRef.current = controller
    const stopProgress = startRequestProgress({
      onProgress: setProgressPhase,
      onTimeout: () => controller.abort('timeout'),
      signal: controller.signal,
    })
    try {
      const continued = isFollowup && conversation
        ? await sendFollowup({ message, conversationToken: conversation.token, language, signal: controller.signal })
        : null
      const next = continued ? { ...continued.route, conversation: continued.conversation }
        : await sendComplaint({ message, history, file, signal: controller.signal })
      if (controller.signal.aborted) return
      setResult(next); setHistory(next.history); setHealth('online')
      setConversation(next.conversation)
      if (next.attachmentAnalysis) setAnalysis(next.attachmentAnalysis)
      if (file) setAttachedName(file.name)
      const answer = continued?.reply || (next.status === 'NEEDS_CLARIFICATION' ? next.question : next.reason || next.service_name || '')
      setTranscript((previous) => [...previous.slice(-18), { role: 'user', text: message.trim(), filename: file?.name }, { role: 'assistant', text: answer, status: next.status }])
      setMessage(''); setFile(null)
      if (fileRef.current) fileRef.current.value = ''
      if (next.status === 'NEEDS_CLARIFICATION') requestAnimationFrame(() => inputRef.current?.focus())
    } catch (error) {
      if (controller.signal.aborted) {
        if (controller.signal.reason !== 'cleanup') setNoticeCode(controller.signal.reason === 'timeout' ? 'timeout' : 'canceled')
      } else {
        if (invalidAttachmentCodes.has(error.code)) {
          setErrors((previous) => ({ ...previous, file: error.code }))
          requestAnimationFrame(() => removeFileRef.current?.focus())
        } else {
          setErrorCode(error.code || 'network')
          setErrorRetryable(error.retryable ?? !error.code)
        }
        // A live backend can report an upstream AI failure. Only a browser
        // connection failure implies that this health indicator is offline.
        if (!error.code || error.code === 'network') setHealth('offline')
      }
    } finally {
      stopProgress()
      busyRef.current = false
      setPending(false)
      controllerRef.current = null
    }
  }

  async function submitFollowup(event) {
    event.preventDefault()
    if (busyRef.current || !conversation) return
    const validation = messageError(followupDraft)
    setFollowupError(validation); setFollowupRetryable(false); setFollowupNotice('')
    if (validation) { followupRef.current?.focus(); return }
    busyRef.current = true
    setFollowupPending(true)
    const controller = new AbortController()
    controllerRef.current = controller
    const stopProgress = startRequestProgress({
      onProgress: setProgressPhase,
      onTimeout: () => controller.abort('timeout'),
      signal: controller.signal,
    })
    try {
      const next = await sendFollowup({ message: followupDraft, conversationToken: conversation.token, language, signal: controller.signal })
      if (controller.signal.aborted) return
      setResult(next.route); setHistory(next.route.history); setConversation(next.conversation); setHealth('online')
      const messages = [...followupMessages, { role: 'user', text: followupDraft.trim() }, { role: 'assistant', text: next.reply }]
      setFollowupMessages(messages.slice(-12))
      if (messages.length > 12) setFollowupTrimmed(true)
      setFollowupDraft('')
      requestAnimationFrame(() => followupRef.current?.focus())
    } catch (error) {
      if (controller.signal.aborted) {
        if (controller.signal.reason !== 'cleanup') setFollowupNotice(controller.signal.reason === 'timeout' ? 'timeout' : 'canceled')
      } else {
        setFollowupError(error.code || 'network')
        setFollowupRetryable(error.retryable ?? !error.code)
        if (!error.code || error.code === 'network') setHealth('offline')
      }
      // Commit the new route, messages and signed context only on success.
    } finally {
      stopProgress(); busyRef.current = false; setFollowupPending(false); controllerRef.current = null
    }
  }

  function pickFile(event) {
    const selected = event.target.files?.[0] || null
    setFile(selected); setErrors((previous) => ({ ...previous, file: fileError(selected) })); setErrorCode(''); setErrorRetryable(false); setNoticeCode('')
    if (selected) requestAnimationFrame(() => removeFileRef.current?.focus())
  }
  function removeFile() {
    setFile(null); setErrors((previous) => ({ ...previous, file: '' })); setErrorCode(''); setErrorRetryable(false); setNoticeCode('')
    if (fileRef.current) fileRef.current.value = ''
    requestAnimationFrame(() => fileRef.current?.focus())
  }
  const statusText = { MATCHED: t.matched, NEEDS_CLARIFICATION: t.clarification, NOT_SUPPORTED: t.unsupported }
  return <div className="app-shell">
    <a className="skip-link" href="#complaint">{t.skip}</a>
    <header className="site-header page-width"><a className="brand" href="#" aria-label={t.brand}><span className="brand-mark"><Icon name="compass" size={32} /></span><span><span className="brand-name">{t.brand}</span><span className="brand-note">{t.brandNote}</span></span></a><div className="header-actions"><span className="beta-badge">{t.beta}</span><button className="language-button" type="button" onClick={() => setLanguage((current) => current === 'ar' ? 'en' : 'ar')} aria-label={t.switchLanguage}><Icon name="globe" size={17} /><span lang={language === 'ar' ? 'en' : 'ar'}>{t.language}</span></button></div></header>
    <main className="page-width">
      <section className="intro" aria-labelledby="page-title"><p className="eyebrow"><span />{t.eyebrow}</p><h1 id="page-title">{t.title}<br /><span>{t.titleAccent}</span></h1><p className="intro-copy">{t.intro}</p><div className="intro-decoration" aria-hidden="true"><Icon name="compass" size={164} /></div></section>
      <ol className="steps" aria-label={language === 'ar' ? 'خطوات التوجيه' : 'Routing steps'}>{t.steps.map((step, index) => <li key={index} className={currentStep === index ? 'active' : ''}><span className="step-number">{String(index + 1).padStart(2, '0')}</span><span>{step}</span></li>)}</ol>
      <div className="workspace"><div className="form-column">
        <section className="form-card" aria-labelledby="form-heading"><div className="form-heading"><div><h2 id="form-heading">{finished ? t.doneTitle : isFollowup ? t.followupTitle : t.formTitle}</h2><p>{finished ? t.doneText : isFollowup ? t.followupIntro : t.formIntro}</p>{isFollowup && conversation && <p>{t.clarificationChoice}</p>}</div><span className="form-number">{isFollowup ? '02' : '01'}</span></div>
          <form onSubmit={submit} noValidate aria-busy={pending}>
            {isFollowup && <div className="inline-question"><Icon name="question" size={20} /><p dir="auto">{result.question}</p></div>}
            {!finished && <>
              <label className="field-label" htmlFor="complaint">{isFollowup ? t.followupLabel : t.complaintLabel}<span aria-hidden="true">*</span></label>
              <textarea id="complaint" ref={inputRef} value={message} onChange={(event) => { setMessage(event.target.value); setErrors((previous) => ({ ...previous, message: '' })); setErrorCode(''); setErrorRetryable(false); setNoticeCode('') }} disabled={busy} dir="auto" placeholder={isFollowup ? t.followupPlaceholder : t.placeholder} rows={6} maxLength={MAX_MESSAGE_LENGTH} required aria-invalid={Boolean(errors.message)} aria-describedby={`input-help character-count${errors.message ? ' message-error' : ''}`} />
              <div className="input-meta"><p id="input-help">{isFollowup ? t.inputHelp : t.initialInputHelp}</p><span id="character-count" className="character-count" aria-label={`${message.length} / ${MAX_MESSAGE_LENGTH} ${t.characters}`}>{message.length.toLocaleString(language)} / {MAX_MESSAGE_LENGTH.toLocaleString(language)}</span></div>
              {errors.message && <p className="field-error" id="message-error" role="alert">{t[errors.message]}</p>}
              {!isFollowup ? <div className="upload-field" role="group" aria-labelledby="attachment-label" aria-invalid={Boolean(currentFileError)} aria-describedby={`file-hint${currentFileError ? ' file-error' : ''}`}><div className="field-label" id="attachment-label">{t.attachment}<span className="optional">{t.optional}</span></div><div className={`upload-box ${file ? 'has-file' : ''} ${currentFileError ? 'invalid-file' : ''}`}><Icon name={file ? 'file' : 'upload'} size={25} /><div className="upload-copy"><strong>{file ? <bdi>{file.name}</bdi> : t.uploadTitle}</strong><span>{file ? `${(file.size / 1024 / 1024).toLocaleString(language, { maximumFractionDigits: 2 })} MB` : t.uploadHint}</span></div>{file ? <button ref={removeFileRef} className="icon-button" type="button" onClick={removeFile} disabled={pending} aria-label={t.removeFile} aria-describedby={`file-hint${currentFileError ? ' file-error' : ''}`}><Icon name="close" size={18} /></button> : <label className={`file-picker ${pending ? 'is-disabled' : ''}`}>{t.uploadAction}<input ref={fileRef} type="file" accept={ACCEPTED_FILES} onChange={pickFile} disabled={pending} aria-label={t.uploadAction} aria-describedby="file-hint" /></label>}</div><p className="sr-only" id="file-hint">{t.uploadHint}</p>{currentFileError && <p className="field-error" id="file-error" role="alert">{t[currentFileError]}</p>}</div>
                : attachedName && <div className="attached-context"><Icon name="file" size={20} /><div><strong>{t.attachmentSent}: <bdi>{attachedName}</bdi></strong><p>{t.attachmentFollowup}</p></div></div>}
            </>}
            {errorCode && <div className="error-banner" role="alert"><Icon name="info" size={20} /><p>{t[errorCode] || t.requestFailed}</p></div>}
            {noticeCode && <p className="notice" role="status">{t[noticeCode]}</p>}
            {!finished && <div className="submit-row"><button className="button primary submit-button" type="submit" disabled={busy || Boolean(currentFileError)} aria-describedby={currentFileError ? 'file-error' : undefined}>{pending ? <><span className="spinner" />{file ? t.loadingFile : t.loading}</> : <>{canRetry ? t.retrySubmission : isFollowup ? t.submitFollowup : t.submit}<Icon name="arrow" size={20} className="direction-arrow" /></>}</button>{pending && <button className="text-button" type="button" onClick={() => controllerRef.current?.abort('user')}>{t.cancel}</button>}</div>}
            {finished ? <button className="button secondary" type="button" onClick={reset} disabled={busy}><Icon name="reset" size={18} />{t.newComplaint}</button> : <div className="privacy-note"><Icon name="lock" size={16} /><p>{t.privacy}</p></div>}
          </form>
        </section>
        {transcript.length > 0 && <details className="history-panel"><summary>{t.history}<span>{transcript.length / 2}</span></summary><ol>{transcript.map((entry, index) => <li key={index} className={`message message-${entry.role}`}><strong>{entry.role === 'user' ? t.you : t.assistant}</strong>{entry.text && <p dir="auto">{entry.text}</p>}{entry.status && entry.status !== 'NEEDS_CLARIFICATION' && <span className="transcript-status">{statusText[entry.status]}</span>}{entry.filename && <small>{t.fileAttached}: <bdi>{entry.filename}</bdi></small>}</li>)}</ol></details>}
      </div><aside className="result-column">
        <ResultCard result={result} pending={pending} processingMessage={t[progressPhase]} t={t} onReset={reset} resetDisabled={busy} />
        {result && conversation && <section className="followup-card" aria-labelledby="followup-heading">
          <h2 id="followup-heading">{t.conversationTitle}</h2><p className="followup-help" id="followup-help">{t.conversationHelp}</p>
          {followupTrimmed && <p className="small-copy muted">{t.recentMessages}</p>}
          {followupMessages.length > 0 && <ol ref={followupLogRef} className="followup-messages" aria-label={t.history} tabIndex={0}>{followupMessages.map((entry, index) => <li key={index} className={`message message-${entry.role}`}><strong>{entry.role === 'user' ? t.you : t.assistant}</strong><p dir="auto">{entry.text}</p></li>)}</ol>}
          <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">{followupPending ? t[progressPhase] : followupMessages.at(-1)?.text}</div>
          <form onSubmit={submitFollowup} noValidate aria-busy={followupPending}>
            <label htmlFor="followup-message" className="field-label">{t.conversationLabel}</label>
            <textarea id="followup-message" ref={followupRef} value={followupDraft} onChange={(event) => { setFollowupDraft(event.target.value); setFollowupError(''); setFollowupRetryable(false); setFollowupNotice('') }} dir="auto" rows={3} maxLength={MAX_MESSAGE_LENGTH} placeholder={t.conversationPlaceholder} disabled={busy} required aria-invalid={Boolean(followupError)} aria-describedby={`followup-help followup-privacy${followupError ? ' followup-error' : ''}`} />
            {followupError && <p className="field-error" id="followup-error" role="alert">{t[followupError] || t.requestFailed}</p>}
            {followupNotice && <p className="notice" role="status">{t[followupNotice]}</p>}
            {followupPending && <p className="followup-wait" aria-hidden="true">{t[progressPhase]}</p>}
            <div className="followup-actions"><button className="button primary" type="submit" disabled={busy}>{followupPending ? <><span className="spinner" />{t.conversationLoading}</> : followupRetryable || followupNotice === 'timeout' ? t.retrySubmission : t.conversationSend}</button>{followupPending ? <button type="button" className="text-button" onClick={() => controllerRef.current?.abort('user')}>{t.cancel}</button> : <button type="button" className="text-button" onClick={reset} disabled={busy}>{t.startNew}</button>}</div>
            <p className="followup-privacy" id="followup-privacy">{t.conversationPrivacy}</p>
          </form>
        </section>}
        {analysis && <details className="analysis-panel"><summary>{t.analysis}</summary>{[['summary', t.summary], ['extracted_facts', t.facts], ['missing_information', t.missing], ['unreadable_content', t.unreadable]].map(([key, label]) => analysis[key] ? <div className="analysis-item" key={key}><h3>{label}</h3><AnalysisValue value={analysis[key]} /></div> : null)}</details>}
        {health === 'offline' && <><div className="connection-bar"><span className="connection-dot offline" /><span role="status">{t.healthOffline}</span><button className="text-button" type="button" onClick={() => { setHealth('checking'); setHealthAttempt((value) => value + 1) }}>{t.retry}</button></div><p className="connection-help">{t.offlineHelp}</p></>}{result && <p className="original-language">{t.originalLanguage}</p>}
      </aside></div>
    </main><footer className="site-footer page-width"><span>{t.footer}</span><span>{t.footerNote}</span>{transcript.length > 0 && !finished && <button className="text-button" type="button" onClick={reset} disabled={busy}>{t.newComplaint}</button>}</footer>
  </div>
}

export default App
