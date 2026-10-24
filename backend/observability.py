"""Request-local timing only: never log complaints, filenames, keys or provider URLs."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import time

logger = logging.getLogger('wajhah')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
logger.propagate = False
current_request: ContextVar[dict | None] = ContextVar('wajhah_request', default=None)


def request_id():
    context = current_request.get()
    return context['request_id'] if context else 'background'


@contextmanager
def measure(stage):
    start = time.perf_counter()
    try:
        yield
    finally:
        context = current_request.get()
        if context is not None:
            context['timings'][stage] = context['timings'].get(stage, 0) + (time.perf_counter() - start) * 1000


def add_attempt(model=None):
    context = current_request.get()
    if context is not None:
        context['ai_attempts'] += 1
        if model is not None:
            context['ai_model'] = model
