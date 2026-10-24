import { REQUEST_TIMEOUT_MS } from './api.js'

// Elapsed time describes the wait, never the provider's internal progress.
// The same cleanup handles success, failure, browser cancellation and unmount.
export function startRequestProgress({ onProgress, onTimeout, signal, timeoutMs = REQUEST_TIMEOUT_MS }) {
  let active = !signal?.aborted
  const timers = []

  function stop() {
    active = false
    timers.forEach(clearTimeout)
    signal?.removeEventListener('abort', stop)
  }

  if (!active) return stop
  const announce = (phase) => { if (active) onProgress(phase) }
  timers.push(setTimeout(() => announce('processingLong'), 10000))
  timers.push(setTimeout(() => announce('processingExtended'), 30000))
  timers.push(setTimeout(() => {
    if (!active) return
    stop()
    onTimeout()
  }, timeoutMs))
  signal?.addEventListener('abort', stop, { once: true })
  announce('processing')
  return stop
}
