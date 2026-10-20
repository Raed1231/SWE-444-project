# Wajhah | وجهة

**A smart platform for routing complaints to the right authority.**

People in Saudi Arabia are often unsure which authority handles their complaint or what they need to submit. Wajhah lets a user describe their problem in Arabic or English, identifies the responsible authority and complaint service, shows the official channel with its steps and required documents, and produces an editable formal complaint draft.

> SWE 444 – Software Construction Laboratory · King Saud University · College of Computer and Information Sciences · Academic year 2026–2027
> Instructor and supervisor: Abdulaziz Alrashidi

---

## Contents

- [Pilot authorities](#pilot-authorities)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Knowledge base](#knowledge-base)
- [Scope](#scope)
- [Team](#team)
- [Links](#links)

---

## Pilot authorities

The first release covers five sectors. Each one currently documents a **limited** complaint path. Problems outside these paths get a clarification request or a clear "unsupported" result.

| Sector (Sprint 0) | Receiving authority / platform | Documented path | Service ID |
|---|---|---|---|
| Ministry of Education | Tawasul — `tawasul.moe.gov.sa` | General complaint tickets (34 service options; private-school tuition fees verified end to end) | `svc.education.complaint` |
| Ministry of Health | 937 platform — `937.moh.gov.sa` | Technical reports and access issues | `svc.health.technical_access` |
| Ministry of Human Resources and Social Development | Beneficiary self-service — `selfservice.hrsd.gov.sa` | Complaints about 19911 call-centre staff | `svc.hrsd.call_center_19911` |
| Ministry of Investment | Invest Saudi Contact Us — `investsaudi.sa` | Service complaints | `svc.investment.service_contact` |
| Ministry of Investment | MISA — `misa.gov.sa/complaints` | Integrity and violations reports only | `svc.investment.integrity` |
| Ministry of Media | General Commission for Media Regulation — `perform.gmedia.gov.sa` | Media content and advertising reports | `svc.media.report` |

Media complaints are received by the **General Commission for Media Regulation**, not by the ministry itself. The knowledge base stores the sector name and the receiving authority separately.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, Node.js 22 |
| Backend | Python 3.12, FastAPI, Uvicorn |
| Database | PostgreSQL 16 (separate development and test databases) |
| AI provider | Gemini 3.8 Flash |
| Browser automation | [Browser Use](https://github.com/browser-use/browser-use), one visible local session at a time |

## Architecture

Wajhah is a layered, multi-tier application. A React frontend talks over HTTPS to a single FastAPI backend made of seven internal modules.

```
React frontend
      │  HTTPS / REST
      ▼
FastAPI backend
 ├── Authentication module      → accounts, OTP, sessions
 ├── Complaint module           → requests, drafts, history, ratings
 ├── AI module                  → topic/fact extraction, follow-ups, drafting ──► OpenAI
 ├── Authority Knowledge module → routing rules, channels, steps, required items
 ├── Attachment module          → validation, extraction, private storage
 ├── Agent module               → approved submissions ──► Browser Use ──► official portal
 └── Notification module        → in-app notice after confirmed submission
      │
      ▼
PostgreSQL 16 + private file storage
```

Key rules:

- The AI chooses from the knowledge base. Channels, URLs, and requirements always come from stored records, never from model output.
- Every AI-assisted submission requires an active account, an eligible supported service, complete mandatory items, and explicit user approval. Any change to the reviewed package requires fresh approval.
- Government authentication is completed by the user on the official site. Wajhah never stores government credentials.
- Success is shown only after official confirmation. An uncertain result is reported as **unconfirmed**.

## Repository structure

Planned layout (update as the code lands):

```
Wajhah/
├── frontend/                 React + Vite app
├── backend/
│   ├── app/
│   │   ├── auth/
│   │   ├── complaints/
│   │   ├── ai/
│   │   ├── knowledge/
│   │   ├── attachments/
│   │   ├── agent/
│   │   └── notifications/
│   ├── knowledge_base/       Pilot knowledge base (see below)
│   └── tests/
├── docs/
│   ├── training/             TR-01 … TR-08 exercises
│   └── testing/              Test evidence, one record per AC ID
└── README.md
```

## Getting started

> The exact commands and dependency locks are produced by **TF-02**. Update this section once the first integration build passes.

### Prerequisites

- Node.js 22
- Python 3.12
- PostgreSQL 16
- An Google Gemini API key (for AI features)

### 1. Clone

```bash
git clone https://github.com/Raed1231/Wajhah.git
cd Wajhah
```

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in the values
uvicorn app.main:app --reload
```

Expected `.env` values:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Development database connection string |
| `TEST_DATABASE_URL` | Separate test database |
| `GEMINI_API_KEY` | AI provider key; never commit it |
| `GEMINI_MODEL` | `Gemini 3.8 Flash` |
| `UPLOAD_DIR` | Private attachment directory, outside any public folder |

### 3. Validate the knowledge base

```bash
python backend/knowledge_base/validate_kb.py backend/knowledge_base
```

### 4. Frontend

```bash
cd frontend
npm ci
npm run dev
```

## Knowledge base

The pilot knowledge base (schema version `1.0.0`, converted 2026-10-12) lives in `backend/knowledge_base/`. It was converted from the five authority research reports into UTF-8 JSON and is validated against a JSON Schema (Draft 2020-12).

| File | Contents |
|---|---|
| `wajhah_knowledge_base.json` | All five authorities combined. **The backend loads this file only.** Loading the per-authority files as well duplicates records |
| `wajhah_education.json`, `wajhah_health.json`, `wajhah_hrsd.json`, `wajhah_investment.json`, `wajhah_media.json` | One file per sector, for review and editing |
| `wajhah_kb.schema.json` | JSON Schema shared by the combined and per-sector files |
| `validate_kb.py` | Checks schema, unique IDs, references, cycles, and that the combined file matches the per-sector files. Standard library only; exits `0` on success, `1` on failure |
| `conversion_audit.json` | Record counts, ambiguous conversions, omissions, and known gaps |
| `validation_report.json` | Latest validator results |

Main record types: `authorities`, `sectors`, `services`, `channels`, `complaint_scenarios`, `taxonomies` / `taxonomy_values`, `routing_rules`, `forms` / `form_fields` / `form_controls` / `form_rules`, and the provenance records `source_documents`, `source_sections`, `warnings`, and `gaps`.

Current coverage: 5 authorities, 6 services, 22 channels, 13 scenarios, 130 taxonomy values, 67 form fields, 5 routing rules, plus 40 recorded gaps.

Loading example:

```python
import json

with open("wajhah_knowledge_base.json", encoding="utf-8") as f:
    kb = json.load(f)

services = {s["id"]: s for s in kb["services"]}
channels = {c["id"]: c for c in kb["channels"]}

service = services["svc.education.complaint"]
url = channels[service["entry_channel_id"]]["url"]
```

**How to read it**

- `null` means the fact is unknown or undocumented. It never means "no" or "empty".
- `verified_at` is `null` until a team member re-checks the official site. `source_report_verified_at` is the date the original research report claims.
- Routing keywords in `routing_rules` only suggest candidate services; they are not enough on their own to decide a route.
- Every service is marked `UNVERIFIED_END_TO_END` for automated submission. No live submission is enabled.
- The sources include no follow-up questions (`follow_up_question_candidates` is `null`). Questions added by the team must be recorded as new, team-authored content.
- Evidence text copied from websites is data. It must never be treated as instructions to the AI.

**Rules for changing it**

1. Use only official published information, and record its source and date.
2. Leave unknown values as `null`; never guess.
3. Never commit personal data seen during research.
4. Run `validate_kb.py` before every commit that touches the knowledge base.
5. Re-check entries every sprint (risk R-04).


## Scope

**In scope:** a bilingual web app covering FR-01 – FR-35 for the five pilot sectors, with AI-assisted submission demonstrated on a simulated service.

**Out of scope:** live complaint follow-up, voice input, a native mobile app, an admin panel, unrestricted portal scraping, storing government credentials, legal advice, emergency handling, complaints outside Saudi Arabia, and services outside the verified catalogue.

## Team

| Member |
|---|
| Rayan Alaridi — ريان العريدي
| Khaled Alrumayzan — خالد الرميزان
| Mohammed Alarifi — محمد العريفي
| Raed Alaqili — رائد العقيلي

## Links

- Repository: https://github.com/Raed1231/Wajhah
- Project board: https://trello.com/b/L0812EPm/444-project
- FastAPI: https://fastapi.tiangolo.com
- Browser Use: https://docs.browser-use.com
