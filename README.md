# Wajhah — Sprint 1 Source Code

Wajhah is an Arabic/English prototype that helps users identify a documented Saudi government complaint service. Users describe their issue, optionally attach evidence, and receive a supported service suggestion, a clarification question, or an explanation that the current knowledge base does not cover the case.

The application uses React 19 and Vite on the frontend, FastAPI on the backend, and Gemini for analysis. Suggested routes must pass a strict local knowledge-base (KB) guard before a service is displayed. Wajhah provides guidance; it does not submit complaints or operate government portals.

## Sprint 1 scope

| User story | Implemented behavior |
| --- | --- |
| US-12 | Submit a free-text complaint with input validation. |
| US-13 | Optionally upload one attachment. |
| US-14 | Validate attachment format, size, and parsed contents. |
| US-16 | Analyze Arabic/English complaints and readable image/PDF evidence. |
| US-17 | Ask clarification questions and retain relevant complaint facts. |
| US-19 | Identify a supported KB service and reject unverifiable routes. |
| US-21 | Display the registered official submission link for a matched service. |

The existing post-result conversation also supports questions, fact corrections, and safe reassessment without uploading the original attachment again. These descriptions explain implemented functionality; they are not a guarantee of every live AI answer.

Only the services and scenarios in `backend/wajhah_routing_kb.json` are supported. `NOT_SUPPORTED` means the available KB cannot support a route; it does not establish official ineligibility. Accounts, official complaint drafting, government login, automated submission, portal navigation, and complaint-status tracking are outside this version.

## Project structure

```text
Wajhah-Sprint1-Final/
├── README.md
├── .gitignore
├── backend/
│   ├── main.py                         # FastAPI endpoints
│   ├── ai_gateway.py                   # Gemini calls and transient-error handling
│   ├── conversation.py                 # Contextual follow-up and reassessment
│   ├── kb_routing.py, contracts.py      # KB guard and structured contracts
│   ├── attachments.py, attachment_analysis.py
│   ├── body_limits.py, request_deadline.py
│   ├── observability.py, service_errors.py
│   ├── wajhah_routing_kb.json
│   ├── requirements.txt, requirements.lock, requirements-dev.txt
│   ├── .env.example
│   └── tests/                          # Automated tests and synthetic fixtures
└── frontend/
    ├── src/                            # Interface, API client, state, translations
    ├── public/wajhah.svg                # Application branding
    ├── tests/
    ├── .gitignore
    ├── index.html, package.json, package-lock.json
    └── vite.config.js, eslint.config.js
```

## Prerequisites

- Python 3.12, Node.js 24, and npm.
- Internet access to install dependencies; a Gemini API key and provider connectivity for live AI requests.
- No API key, external fonts, Poppler, or ReportLab are needed for the normal offline tests.

Commands below are for macOS/Linux and start from the extracted project root unless stated otherwise. On Windows, use `py -3.12` to create the environment and `.venv\Scripts\python.exe` instead of `.venv/bin/python`. Windows has not been verified in this delivery.

## Installation and configuration

From the project root:

```sh
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip check
```

`requirements.txt` records direct dependencies; `requirements.lock` pins the complete tested environment. Use the lockfile for reproducible installation.

For live analysis, copy `backend/.env.example` to `backend/.env` only if no local `.env` already exists. Enter your own key in that file. The example contains:

```dotenv
GEMINI_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
WAJHAH_COMBINED_ATTACHMENTS=false
```

The model and default attachment pipeline are preserved from the working application. Keep the key on the backend: never put it in React, a `VITE_` variable, source code, command history, or Git. Offline tests require no `.env` file.

In a second terminal, from the project root:

```sh
cd frontend
npm ci --include=dev --include=optional
```

## Run locally

In the backend terminal, from `backend/`:

```sh
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In the frontend terminal, from `frontend/`:

```sh
npm run dev
```

Open [Wajhah](http://127.0.0.1:5173/). The backend exposes [health](http://127.0.0.1:8000/health) and [API documentation](http://127.0.0.1:8000/docs). Use one server per port. `/health` tests API liveness, not Gemini availability.

Vite forwards `/api/*` to the backend on port 8000. A production frontend build needs a running backend and an equivalent same-origin `/api` proxy; `npm run preview` alone does not run the full application. No deployment is included.

## Basic usage

1. Choose Arabic or English; the interface uses RTL or LTR accordingly.
2. Enter a complaint and optionally select one supported attachment.
3. Submit once and wait for the response. Loading begins immediately; additional messages appear after 10 and 30 seconds without percentage progress.
4. Read `MATCHED`, `NEEDS_CLARIFICATION`, or `NOT_SUPPORTED`. For a match, use the displayed official link; for clarification, answer in the conversation.
5. Ask related questions, correct facts, or request reassessment. Use the new-complaint control to reset the current context. Failed requests retain the draft, attachment, and conversation for a manual retry.

## Routing, attachments, and conversation

Service IDs, names, and official links are resolved from the local KB rather than trusted from model-generated text. A schema-valid response can still be rejected by the KB guard. Unsupported or invalid decisions are not silently accepted. Link records include their verification status; they are not a promise that every portal or procedure is currently unchanged.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness without an AI call. |
| `POST /route` | JSON `{message, history}` for initial routing and legacy clarification. |
| `POST /attachments` | Multipart `file` validation without AI. |
| `POST /analyze-attachment` | Multipart `complaint, file` for attachment analysis. |
| `POST /route-with-attachment` | Multipart `complaint, file`; returns routing, attachment analysis, and conversation context. |
| `POST /follow-up` | JSON `{message, conversation_token, language}` for contextual follow-up. |

The frontend uses `/route` or `/route-with-attachment` initially, then `/follow-up`, including clarification replies. Extracted facts remain available without re-uploading. Once follow-up starts, keep using its conversation token instead of switching back to legacy history, which may omit fact updates. A different attachment requires a new complaint.

Ordinary questions do not automatically reroute a complaint. Fact updates and reassessment pass the same KB guard. Answers distinguish documented information from general preparation advice and unavailable information; they must not invent official documents, fees, eligibility rules, or portal procedures. The initial routing answer or official service names may remain Arabic; contextual follow-up supports both languages.

- Complaint text: 1–6,000 characters.
- Attachment: one non-empty JPG/JPEG/PNG/PDF, at most 5 MiB. PDFs must be unencrypted and contain 1–20 pages; static images are limited to 20 million pixels. Parsed contents and signatures are checked, not only extensions.
- Conversation: at most 40 follow-ups and 12 fact updates, with bounded recent history. Resource limits reject excessive context instead of silently dropping important facts.
- The signed conversation token is not encrypted and contains complaint context. Do not share or log it. It expires after two hours and is invalidated by a backend restart. This prototype uses one backend process, not independently keyed workers.
- Conversation state lives in the current page. Refreshing or resetting clears it; only the language preference is stored locally.

## Reliability, privacy, and limitations

The backend has a **60-second overall deadline**, including AI stages, retries, and backoff. This is a reliability fallback, not a normal performance target or a per-attempt timeout. Successful responses return immediately. Each AI stage allows at most one retry for explicit provider HTTP 500/502/503, with all stages and retries sharing the same request budget; timeouts, 429 responses, invalid output, and ambiguous connection failures are not automatically retried. The development proxy allows 65 seconds and the HTTP client allows 70 seconds.

Errors use a safe structure with a reason code, retryability, and request ID. Logs record timing separately from outcome and avoid complaint text, attachments, tokens, keys, and raw provider errors. Cancelling a browser request does not guarantee that provider processing stops.

Live use sends complaint content, attachments, and relevant context to Gemini. Use fictional data for demonstrations. The package includes only synthetic fixtures, no real complaints or user documents. Local secrets, installed environments, caches, build output, logs, and Git history are excluded.

Gemini can be slow or unavailable, and intermittent provider/validation failures remain possible. Offline success does not establish live semantic accuracy or eliminate that risk. KB coverage and detailed official requirements are limited; consult the registered official service for current requirements. Wajhah is a local prototype, not an approved production deployment.

## Tests and reproduction

From `backend/`, run all tests, explicitly including the approximately 25- and 60-second simulated deadline cases:

```sh
WAJHAH_RUN_SLOW_BUDGET_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

Omitting the environment flag skips the two slow cases. Tests use real image/PDF parsers, synthetic fixtures, and mocked Gemini responses, including SDK tests with mocked HTTP transport. No live AI key or server is required.

From `frontend/`:

```sh
npm test
npm run lint
npm run build
```

The tests cover validation, KB rejection, clarification, context retention, conversational grounding, deadlines, safe errors, reset handling, and bilingual client behavior. `tests/backend_contract.py` in the frontend is a retained optional launcher for the backend suite, not extra independent tests.

The six saved attachment fixtures have a provenance README and hash manifest. Only regeneration needs development dependencies: from `backend/`, install `requirements-dev.txt`, then run `python tests/fixtures/generate_fixtures.py`. Do not regenerate fixtures just to run tests.

Optional live benchmarking, from `backend/` with a configured running backend:

```sh
.venv/bin/python tests/live_benchmark.py --run-live --samples 1 --output tests/reports/manual-live.json
```

This uses synthetic data and consumes provider quota. It measures initial requests, not the complete conversation, and is never part of offline test discovery.

## Final-copy verification

<!-- FINAL_VERIFICATION_START -->
The final copy was independently installed and tested on **24 October 2026**, using fresh environments on macOS with Python 3.12.10, Node.js 24.22.0, and npm 11.19.1. Application code and KB contents are byte-identical to the current working project and the prior clean English package.

| Check executed against this final copy | Result |
| --- | --- |
| Fresh lockfile installations and `pip check` | Passed; no broken Python requirements. |
| Local imports, dependency provenance, KB load, and expected API contracts | Passed; 11 application modules, 10 dependency modules, and all six API operations. |
| Unmodified FastAPI startup, `/health`, and `/openapi.json` | Passed; HTTP 200. |
| Complete backend suite with slow-test flag | **196/196 passed**, no skips; about 88 seconds, including the 25- and 60-second mocked deadline tests. |
| Frontend tests | **37/37 passed**, no skips. |
| ESLint and production build | Passed. |
| Real local HTTP through Vite to the final backend | Passed: text routing, PDF validation/analysis, clarification with retained attachment facts, Arabic/English follow-up, and invalid-input rejection. Only the Gemini SDK client was scripted. |
| Actual desktop browser | Passed: Arabic RTL display, initial submission and official link, English LTR display and contextual follow-up, and new-complaint reset. |
| Git-ignore behavior | Passed: 24 checks covering excluded secrets/generated metadata and retained source/configuration/fixtures. |

Integration used isolated ports 15173/18080 to avoid existing servers. Vite's test port and proxy target were overridden at runtime only; the submitted configuration still uses 5173/8000 with the original rewrite and 65-second proxy timeout.

The npm installation reported an `allowScripts` warning for optional `fsevents@2.3.3`. No script-approval setting was changed; installation, development startup, tests, lint, and build succeeded afterward.

**No live Gemini calls were executed.** Offline results do not establish live classification accuracy, provider availability, or the currency of government websites. Mobile viewport behavior, Windows, production deployment, and a full browser test of every attachment/error branch were not verified in this cleanup. Relevant server/client branches are covered by the automated suite.

Verification environments and build output were removed afterward. The delivery folder and ZIP contain the same source files, without secrets or personal documents. Reinstall dependencies before running. **Status: ready for GitHub source submission**, with the above live-integration and production limitations; nothing was pushed or deployed.
<!-- FINAL_VERIFICATION_END -->
