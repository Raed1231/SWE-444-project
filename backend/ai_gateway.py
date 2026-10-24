"""Shared Gemini connection, schema validation and strictly bounded async attempts."""
import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from observability import add_attempt, logger, measure, request_id
from request_deadline import REQUEST_TIMEOUT_SECONDS, remaining_seconds, request_budget
from service_errors import ServiceError

load_dotenv(Path(__file__).with_name('.env'))
MODEL = os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')
MAX_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.5
_client = None
_http_client = None


def get_client():
    global _client, _http_client
    if _client is None:
        key = os.getenv('GEMINI_API_KEY')
        if not key:
            raise ServiceError('AI_NOT_CONFIGURED', 503, 'خدمة التحليل غير مهيأة على الخادم.')
        _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)
        _client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
            httpx_async_client=_http_client,
        ))
        logger.info('ai_client_ready request_id=%s model=%s timeout_s=%.1f max_attempts=%d transport=httpx',
                    request_id(), MODEL, REQUEST_TIMEOUT_SECONDS, MAX_ATTEMPTS)
    return _client


async def close_client():
    global _client, _http_client
    client, http_client = _client, _http_client
    _client, _http_client = None, None
    try:
        if client is not None:
            await client.aio.aclose()
    finally:
        # The SDK deliberately leaves a caller-provided HTTP client open.
        if http_client is not None:
            await http_client.aclose()


def classify_error(error):
    code = getattr(error, 'code', None)
    if isinstance(error, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)) or code in (408, 504):
        return ServiceError('AI_TIMEOUT', 504, 'انتهت مهلة التحليل. يمكنك إعادة المحاولة.', True)
    if code == 503:
        return ServiceError('AI_OVERLOADED', 503, 'خدمة الذكاء الاصطناعي مزدحمة مؤقتًا. حاول بعد قليل.', True)
    if code == 429:
        return ServiceError('AI_RATE_LIMITED', 429, 'تم بلوغ حد طلبات خدمة التحليل. حاول لاحقًا.', True)
    if isinstance(error, httpx.TransportError):
        return ServiceError('AI_CONNECTION_ERROR', 502, 'تعذّر اتصال الخادم بخدمة التحليل.', True)
    return ServiceError('AI_UPSTREAM_ERROR', 502, 'تعذّر إكمال الطلب لدى خدمة التحليل.', code in (500, 502))


async def generate(schema, system_instruction, contents, stage):
    # HTTP requests inherit a shared deadline across all AI stages. Direct
    # callers receive one bounded budget covering this call and its retries.
    with request_budget():
        return await _generate(schema, system_instruction, contents, stage)


async def _generate(schema, system_instruction, contents, stage):
    client = get_client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type='application/json',
        # SDK 2.30 converts Pydantic extra='forbid' into an unsupported legacy
        # response_schema.additional_properties field. Send standard JSON Schema
        # intact, then independently enforce the strict model on the server.
        response_json_schema=schema.model_json_schema(),
        temperature=0,
        max_output_tokens=2300,
    )
    for attempt in range(1, MAX_ATTEMPTS + 1):
        remaining = remaining_seconds()
        if remaining <= 0:
            raise ServiceError('AI_TIMEOUT', 504, 'انتهت مهلة الطلب. يمكنك إعادة المحاولة.', True)
        config.http_options = types.HttpOptions(timeout=max(1, int(remaining * 1000)))
        add_attempt(MODEL)
        start = time.perf_counter()
        try:
            with measure(stage):
                response = await asyncio.wait_for(client.aio.models.generate_content(
                    model=MODEL, contents=contents, config=config,
                ), timeout=remaining)
        except ServiceError:
            raise
        except (errors.APIError, httpx.HTTPError, asyncio.TimeoutError, TimeoutError) as error:
            problem = classify_error(error)
            source = ('provider' if isinstance(error, errors.APIError) else
                      'transport' if isinstance(error, httpx.HTTPError) else 'local_deadline')
            # Do not log str(error): SDK messages may contain request data or URLs.
            logger.warning('ai_failure request_id=%s stage=%s model=%s attempt=%d exception=%s provider_code=%s code=%s source=%s timeout_s=%.1f elapsed_ms=%.1f',
                           request_id(), stage, MODEL, attempt, type(error).__name__, getattr(error, 'code', None), problem.code,
                           source, remaining, (time.perf_counter() - start) * 1000)
            # Retry only explicit transient provider responses; no silent replay on
            # timeout, network ambiguity, malformed output or quota exhaustion.
            if getattr(error, 'code', None) in (500, 502, 503) and attempt < MAX_ATTEMPTS:
                with measure('retry_wait'):
                    await asyncio.sleep(min(RETRY_DELAY_SECONDS, remaining_seconds()))
                continue
            raise problem from None
        try:
            if not response.text:
                raise ValueError('Empty structured response')
            result = schema.model_validate_json(response.text)
        except (ValidationError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            logger.warning('ai_invalid_response request_id=%s stage=%s', request_id(), stage)
            raise ServiceError('AI_INVALID_RESPONSE', 502, 'وصلت نتيجة تحليل غير مكتملة. حاول مجددًا.', True) from None
        return result
    raise AssertionError('Unreachable retry state')
