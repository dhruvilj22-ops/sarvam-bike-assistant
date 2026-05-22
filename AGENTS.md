# Bike Troubleshooting Assistant — AI-Native Application

## What This Is

An AI-native troubleshooting assistant for two-wheeler owners, roadside mechanics, and DIY enthusiasts
in India. Users describe bike issues via text, voice, or image. The assistant answers strictly from
indexed bike manuals — owner's manual, service manual, user guide. It never answers from general
knowledge. It refuses clearly when the answer is not in the manual.

This is a Sarvam assignment submission. Every major technical decision in this file has a one-line
justification. Simplicity and correctness are prioritized over completeness.

---

## User Context — Why This Exists

Three personas. One hard constraint.

**Personas**
- The owner — tier-2/3 city, low English comfort, far from service center, doesn't know if their
  issue is serious
- The roadside mechanic — high volume, unfamiliar variants, needs fast specific answers on someone
  else's bike
- The DIY enthusiast — technically inclined, wants to query the manual conversationally without
  hunting through 300 pages

**The hard constraint**
A wrong answer is worse than no answer. If the assistant hallucinates a torque spec or dismisses a
warning symptom, the user acts on it. That is a safety risk. Refusal is always safer than a confident
wrong answer.

---

## Anti-Hallucination Contract

This is the topmost priority of the entire system. Enforced at four layers — not just the prompt.

**Layer 1 — Retrieval confidence gate**
Before the LLM sees anything, the re-ranker confidence score is checked. If the top chunk scores
below the defined threshold, the query is flagged as potentially out-of-scope. This flag is passed
into the prompt. The LLM is told the context confidence is low before it generates.

**Layer 2 — Prompt-level hard instruction**
System prompt contains: "Answer only from the provided context sections. If the context does not
contain sufficient information, respond with: I couldn't find this in your [bike] manual. For this
issue, I'd recommend visiting an authorised service center. Do not use general automotive knowledge
under any circumstances."

**Layer 3 — Output validation**
Post-generation check: does the response contain a citation? If not, flag and re-generate. This
catches cases where the LLM answered from training data instead of retrieved context.

**Layer 4 — UX transparency**
Every answer shows the source section to the user. Users who can see the manual section are
implicitly verifying the answer themselves. This is both a trust feature and a passive guardrail.

---

## Technical Decisions

### Stack
- Frontend: Next.js
- Backend: Python FastAPI — serves static Next.js build at /
- Database: SQLite — session state, issue threads, history
- Vector store: Qdrant — namespace-isolated per bike manual
- Package manager: uv
- Containerization: Docker — target runtime, not a development blocker (see Docker Policy below)
- Environment: .env in project root

### AI and API Decisions

| Component | Choice | Why |
|---|---|---|
| STT Indic languages | Sarvam STT API | Meaningfully better accuracy for Hindi and regional Indian languages — the primary user demographic |
| STT English | OpenAI Whisper | Strong English accuracy, widely supported |
| Language routing | Detect on first query, store in session | Route to Sarvam for Indic, Whisper for English |
| TTS Indic | Sarvam TTS API | Target users speak Hindi, Tamil, Telugu — Sarvam is purpose-built for this |
| TTS English | OpenAI TTS | Consistent quality for English output |
| LLM | GPT-4o via OpenRouter | Strong instruction following, structured output support, vision capability |
| Embeddings | OpenAI text-embedding-3-small | Cost-effective, strong on technical text |
| PDF parsing prose | PyMuPDF | Fast, reliable for text and structure extraction |
| PDF parsing tables | LlamaParse | Table-aware extraction — PyMuPDF flattens tables into garbled text |
| Image understanding | GPT-4o vision | Converts diagram content to searchable text descriptions at ingestion and inference |
| Re-ranking | Cohere Rerank | Cross-encoder quality gate before LLM — single highest-impact retrieval improvement |
| Retrieval method | Hybrid BM25 + dense vector | Part codes and spec numbers need exact match; semantic handles natural language |

---

## API Keys Master List

All keys live in .env at project root. Never commit .env.

```
OPENROUTER_API_KEY       # LLM calls via OpenRouter
SARVAM_API_KEY           # STT and TTS for Indic languages
OPENAI_API_KEY           # Embeddings, Whisper STT, TTS English, GPT-4o vision
COHERE_API_KEY           # Rerank cross-encoder
QDRANT_URL               # Vector store (use in-memory mock if not available)
QDRANT_API_KEY           # Vector store auth (not needed for in-memory)
LLAMAPARSE_API_KEY       # Table-aware PDF parsing
USE_MOCKS=false          # Set to true to mock all external APIs
```

---

## Mock Strategy — All External APIs

Every external API has a drop-in mock. Switching mock on or off is a single env flag: USE_MOCKS=true.
Mocks have the same input/output contract as the real API. No pipeline code changes when switching.

| API | Mock Behavior |
|---|---|
| Sarvam STT | Returns hardcoded transcript: "white smoke coming from exhaust" |
| Sarvam TTS | Skips audio generation, returns text confirmation only |
| Whisper STT | Returns same hardcoded transcript as Sarvam mock |
| OpenAI embeddings | Returns a fixed random vector of correct dimensions — pipeline runs, retrieval not meaningful |
| Cohere Rerank | Returns input chunks in original order with mock scores 0.9, 0.7, 0.5 |
| LlamaParse | Returns pre-saved JSON fixture from tests/fixtures/sample_table.json |
| GPT-4o vision | Returns hardcoded: "Image shows white smoke from rear exhaust pipe, indicating possible oil burning or coolant leak" |
| OpenRouter LLM | Returns hardcoded structured response fixture from tests/fixtures/sample_response.json |
| Qdrant external | Uses qdrant-client in-memory mode — no external service, no API key needed |

Mock fixtures live in tests/fixtures/. Each fixture is a real API response captured during initial
integration testing so the shape is accurate.

---

## Docker Policy

Docker is the target runtime for final submission. It is never a development blocker.

Rules:
- Every part must run and pass tests locally without Docker first
- Docker is added in Part 12 as the final packaging step
- If any step says "run in Docker", interpret as "run locally first, Docker later"
- start.sh and stop.sh exist for both local (no Docker) and Docker modes

---

## Test Gate Policy — Mandatory Before Every Part Transition

This rule is non-negotiable. Claude must not proceed to the next part until the current part passes.

Rules:
- After completing each part, run the full test suite for that part
- Report results as: "X/Y tests passing (N%). Proceeding." or "X/Y tests passing (N%). Fixing before proceeding."
- 90% pass rate is the minimum threshold to proceed
- If below 90%, fix the failures before moving on — do not skip, do not defer
- Every part has a "Local verification" section — confirm this manually before marking the part done
- Test files live in tests/ and are named test_part_N.py or test_part_N.test.ts

---

## Architecture — Three Layers

**Layer 1 — Ingestion** (offline, runs once per manual)
Processes a manual PDF into a retrieval-ready indexed knowledge base. Most important engineering
decision in the product. Poor ingestion cannot be rescued by prompt engineering.

**Layer 2 — Inference** (real-time, runs on every query)
Takes multimodal input, retrieves grounded context, generates a strictly cited response.

**Layer 3 — Application** (what the user touches)
Session state, bike selection, issue threading, history, language routing.

---

## Ingestion Pipeline

### Step 1 — Content type classification pass
Before any chunking, run a classification pass over the entire parsed document. Label every block:
- prose
- table
- image/diagram
- warning/caution
- procedure

Chunking strategy is applied per content type. This pre-processing step is what separates a
retrieval system that works on technical manuals from one that technically runs but gives poor answers.

### Step 2 — Extraction per content type

**Prose** — PyMuPDF. Preserves headings, section numbers, chapter structure.

**Tables** — LlamaParse. Extract as structured key-value pairs, not flattened text. Each row or
logical group of rows becomes its own chunk. Example output: "Front axle nut torque: 65 Nm" not a
garbled flattened string.

**Images and diagrams** — PyMuPDF extracts image position and page number. At ingestion time, send
each image to GPT-4o vision with prompt: "Describe this technical diagram from a bike service manual.
List all labeled components, their positions, and any specifications shown." Store description as a
text chunk. Store reference to original image file path for display alongside answers.

**Warnings and cautions** — Rule-based detector identifies WARNING/CAUTION/NOTE callout blocks.
Always isolated as their own chunks. Never absorbed into adjacent prose. Safety-critical content must
never be buried inside a larger chunk.

**Procedures** — Numbered step lists detected and kept whole. A procedure chunk never splits
mid-step. Full procedure is the parent chunk. Individual steps are child chunks. Chunk boundary rule:
if a boundary falls inside a numbered list or mid-table, extend to the next natural break regardless
of token count.

### Step 3 — Chunking parameters

| Content type | Chunk size | Overlap | Special rule |
|---|---|---|---|
| Prose | 400-600 tokens | 80-100 tokens | None |
| Table | One row or logical group | None | No token limit applied |
| Warning/caution | One callout block | None | Never split |
| Procedure | Full procedure as parent, one step per child | None | Never split mid-step |
| Diagram | One description per image | None | Store image_path reference |

### Step 4 — Metadata schema

Every chunk receives the following metadata fields.

**Document-level** (identical for all chunks from one manual)
```
bike_brand
bike_model
bike_year
manual_type          # owner_manual / service_manual / user_guide
manual_source        # library / user_uploaded
document_id          # hash of source PDF — used as Qdrant namespace key
```

**Chunk-level**
```
chunk_id
content_type         # prose / specification / diagram / warning / procedure
chapter_number
chapter_title
section_number
section_title
page_number
parent_chunk_id      # populated for child chunks, null for parents
is_parent            # true / false
```

**Content-type-specific**
```
# specification chunks (tables)
table_title
spec_unit            # Nm / L / mm / rpm

# diagram chunks
image_path           # relative path to original image file
diagram_type         # exploded_view / wiring / location / cross_section

# warning chunks
severity             # WARNING / CAUTION / NOTE
related_procedure    # section_number of the associated procedure

# procedure chunks
procedure_step_count
tools_required       # list of tools mentioned in the procedure
```

### Step 5 — Document index

After ingestion, generate a document-level index JSON per manual:
```json
{
  "document_id": "",
  "bike_brand": "",
  "bike_model": "",
  "bike_year": "",
  "manual_type": "",
  "chapters": [{ "number": "", "title": "", "page_start": 0, "page_end": 0 }],
  "content_type_counts": {
    "prose": 0, "specification": 0, "diagram": 0, "warning": 0, "procedure": 0
  },
  "total_chunks": 0,
  "confidence_threshold": 0.0,
  "ingestion_timestamp": ""
}
```

Used for ingestion validation, query routing hints, and showing manual coverage to the user.

### Step 6 — Embedding and indexing

- Embed every chunk with OpenAI text-embedding-3-small
- Store in Qdrant under a namespace keyed to document_id
- Maintain a BM25 keyword index alongside for each namespace
- Namespace isolation ensures zero cross-model contamination at retrieval time

---

## Inference Pipeline

### Step 1 — Multimodal input handling

Three input types arrive at a unified endpoint:
- Voice → Sarvam STT (Indic) or Whisper (English) → text transcript
- Text → passes through directly
- Image → GPT-4o vision → structured text description → feeds retrieval
- Combinations → merge all into one unified query string

STT confidence check: if STT returns low confidence score, prompt user —
"I didn't catch that clearly — could you type it or try again?"

### Step 2 — Query understanding and expansion

One lightweight LLM call before retrieval:
- Detect language
- Extract core symptom or question
- Expand with synonyms — "thak thak awaaz" maps to "knocking sound engine"
- Classify query intent: diagnostic / specification / procedure / out-of-scope

### Step 3 — Query routing

Based on intent classification, adjust retrieval weight:
- Diagnostic → semantic retrieval dominant
- Specification lookup → BM25 + specification chunk filter dominant
- Procedure lookup → parent chunk retrieval dominant

### Step 4 — Hybrid retrieval

- Pre-filter Qdrant by document_id namespace
- Run expanded query against both vector index and BM25 simultaneously
- Merge using Reciprocal Rank Fusion
- Retrieve top 5-7 chunks

### Step 5 — Re-ranking

- Pass top chunks through Cohere Rerank cross-encoder
- Keep top 3 chunks after re-ranking
- 3 high-quality chunks outperform 10 mediocre ones

### Step 6 — Confidence gate

- Check re-ranker confidence scores against threshold set during ingestion testing
- If top chunk below threshold, flag query as out-of-scope
- Pass flag into prompt as context_confidence: low

### Step 7 — Response generation

LLM prompt has three components:
- System prompt — persona, hard constraints, citation format, severity label format, refusal language
- Retrieved context — top 3 chunks labeled with source section
- Conversation history — last 4-6 turns of current issue thread only

Structured output returned:
```json
{
  "answer_text": "",
  "spoken_summary": "",
  "citations": [{ "section_number": "", "section_title": "", "page_number": 0 }],
  "severity_label": "",
  "confidence": "",
  "suggested_followups": []
}
```

### Step 8 — TTS output

- If query was voice-initiated, send spoken_summary to Sarvam TTS (Indic) or OpenAI TTS (English)
- Never send full answer_text to TTS — too long for voice
- Full answer_text always displayed as text regardless of input mode

---

## Multi-Bike Support

### Library bikes
2-3 bikes pre-indexed at setup: Royal Enfield Meteor 350, TVS Apache RTR 160, Bajaj Pulsar NS200.
Each indexed under its own document_id namespace.

### Dynamic upload
User uploads PDF. Ingestion runs async. Progress shown. Conversation unlocks after ingestion completes.

### Precedence rule
User-uploaded manual always takes precedence over library manual for the same bike.

---

## Session and State Management

**Issue threading** — each issue is a separate conversation object with its own retrieval and message
history. New issue detected via LLM intent classification. Multiple threads open simultaneously.
Context never bleeds between threads.

**History summarization** — after turn 4 in a thread, summarize earlier turns into a compressed
summary. Use summary plus recent turns instead of full raw history.

**Thread resolution** — async LLM call at resolution generates: one-line issue summary, manual
section referenced, severity, status (Resolved / Ongoing / Go to service center), date. Not in
critical path.

**Language state** — detected language stored in session. Re-detect on each turn. No hard lock.
Language switching handled gracefully.

---

## Coding Standards

1. Use latest versions of libraries and idiomatic approaches
2. Never over-engineer. Always simplify. No unnecessary defensive programming. No extra features
3. Be concise. Keep README minimal. No emojis ever
4. When hitting issues, always identify root cause before trying a fix. Do not guess. Prove with
   evidence then fix root cause
5. Every AI and API decision has a one-line justification comment in code
6. Retrieval quality over UI polish
7. Hard refusal is always preferable to a hedged or hallucinated answer
8. Mocks must have identical input/output contracts to real APIs — no special casing in pipeline code

---

## Working Documentation

All planning and execution documents are in docs/.
Review docs/PLAN.md before proceeding with any part.
