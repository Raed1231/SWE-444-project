"""One monotonic budget for validation, AI stages, retries, and response start."""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import time

from fastapi.responses import JSONResponse

from observability import logger, request_id
from service_errors import ServiceError

REQUEST_TIMEOUT_SECONDS = 60.0
current_deadline: ContextVar[float | None] = ContextVar('wajhah_deadline', default=None)


@contextmanager
def request_budget(seconds=None):
    deadline = time.monotonic() + (REQUEST_TIMEOUT_SECONDS if seconds is None else seconds)
    inherited = current_deadline.get()
    token = current_deadline.set(min(inherited, deadline) if inherited is not None else deadline)
    try:
        yield current_deadline.get()
    finally:
        current_deadline.reset(token)


def remaining_seconds():
    deadline = current_deadline.get()
    return max(0.0, deadline - time.monotonic()) if deadline is not None else REQUEST_TIMEOUT_SECONDS


class _DeadlineExpired(Exception):
    pass


class RequestDeadline:
    """Pure ASGI cancellation; no detached call_next task survives the deadline."""
    def __init__(self, app, timeout=None):
        self.app, self.timeout = app, timeout

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        response_started = False
        with request_budget(self.timeout) as deadline:
            def check_deadline():
                if time.monotonic() >= deadline:
                    raise _DeadlineExpired()

            async def bounded_receive():
                check_deadline()
                message = await receive()
                check_deadline()
                return message

            async def bounded_send(message):
                nonlocal response_started
                check_deadline()
                if message['type'] == 'http.response.start':
                    response_started = True
                await send(message)

            timer = asyncio.timeout_at(deadline)
            try:
                async with timer:
                    await self.app(scope, bounded_receive, bounded_send)
                return
            except TimeoutError:
                if not timer.expired():
                    raise
            except _DeadlineExpired:
                pass
            logger.warning('request_deadline request_id=%s source=request_deadline budget_s=%.1f',
                           request_id(), REQUEST_TIMEOUT_SECONDS if self.timeout is None else self.timeout)
            if response_started:
                # Status cannot change after headers. Terminate the response;
                # never send a second response-start event.
                raise _DeadlineExpired('Response deadline exceeded') from None
            error = ServiceError('AI_TIMEOUT', 504, 'انتهت مهلة الطلب. يمكنك إعادة المحاولة.', True)
            await JSONResponse(status_code=504, content={'detail': error.detail(request_id())})(scope, receive, send)
