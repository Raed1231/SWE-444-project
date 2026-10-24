"""Opt-in live benchmark. Generates no complaints and uses synthetic fixtures only."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-live', action='store_true', help='Explicitly permit real Gemini calls through the local API')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--samples', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--label', default='current-default')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports' / 'live.json')
    args = parser.parse_args()
    if not args.run_live:
        parser.error('No requests sent. Pass --run-live to permit real provider usage.')
    url = urlsplit(args.base_url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password:
        parser.error('Use a local HTTP API URL without credentials.')
    fixture_dir = Path(__file__).parent / 'fixtures'
    message = 'هذه بيانات اختبار وهمية: المدرسة الأهلية رفعت الرسوم من 10000 إلى 12000 ريال، وأريد تقديم شكوى عن الرسوم الدراسية.'
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.base_url, timeout=65) as client:
        for sample in range(1, args.samples + 1):
            for name in ['text', 'image.png', 'text-based.pdf', 'image-only.pdf']:
                started = time.perf_counter()
                row = {'label': args.label, 'case': name, 'sample': sample}
                try:
                    if name == 'text':
                        response = client.post('/route', json={'message': message, 'history': []})
                    else:
                        mime = 'image/png' if name.endswith('.png') else 'application/pdf'
                        response = client.post('/route-with-attachment', data={'complaint': message}, files={'file': (name, (fixture_dir / name).read_bytes(), mime)})
                    payload = response.json()
                    result = payload.get('routing', payload)
                    row.update(http_status=response.status_code, routing_status=result.get('status'), service_id=result.get('service_id'),
                               error_code=payload.get('detail', {}).get('code'), server_timing=response.headers.get('server-timing'), ai_attempts=response.headers.get('x-ai-attempts'))
                except (httpx.HTTPError, ValueError) as error:
                    row['exception'] = type(error).__name__
                row['seconds'] = round(time.perf_counter() - started, 3)
                rows.append(row)
                args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
