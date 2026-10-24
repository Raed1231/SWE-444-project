"""Bound request bytes before JSON parsing or multipart temporary-file spooling."""
from fastapi.responses import JSONResponse
from observability import request_id
from service_errors import ServiceError


class RequestBodyLimit:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path = scope.get('path')
        multipart = path in {'/attachments', '/analyze-attachment', '/route-with-attachment'}
        if scope.get('method') != 'POST' or not (multipart or path in {'/route', '/follow-up'}):
            return await self.app(scope, receive, send)
        limit = 6 * 1024 * 1024 if multipart else 512 * 1024 if path == '/follow-up' else 256 * 1024
        buffer, total = bytearray(), 0
        headers = dict(scope.get('headers', []))
        declared = headers.get(b'content-length', b'')
        oversized = declared.isdigit() and int(declared) > limit
        while not oversized:
            event = await receive()
            if event['type'] == 'http.disconnect':
                return
            total += len(event.get('body', b''))
            if total > limit:
                oversized = True
                break
            buffer.extend(event.get('body', b''))
            if not event.get('more_body', False):
                break
        if oversized:
            error = ServiceError('ATTACHMENT_TOO_LARGE' if multipart else 'REQUEST_TOO_LARGE', 413,
                                 'حجم الطلب يتجاوز الحد المسموح.')
            return await JSONResponse(status_code=413, content={'detail': error.detail(request_id())})(scope, receive, send)
        # Buffered bytes are bounded (6 MiB or 256 KiB); the parser never sees
        # an oversized body, including chunked requests or extra file parts.
        body = bytes(buffer)
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()
        await self.app(scope, replay, send)
