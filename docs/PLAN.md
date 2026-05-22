# Bike Troubleshooting Assistant — Build Plan

## How to Use This File

Work through parts sequentially. Never skip a part. Before starting any part, read it fully and
confirm what you are about to do. After completing any part, run the test suite, report the pass
rate, and verify locally before proceeding. 90% pass rate is the minimum to move forward.

---

## Part 1: Plan review, environment setup, and API connectivity checks

### What this part does
Reviews this plan, sets up the project structure, confirms all available API keys, and verifies
connectivity to every external service that will be used.

### API keys required for this part
```
OPENROUTER_API_KEY       # required — LLM connectivity check
SARVAM_API_KEY           # optional — mock available
OPENAI_API_KEY           # optional — mock available
COHERE_API_KEY           # optional — mock available
QDRANT_URL               # optional — in-memory fallback available
LLAMAPARSE_API_KEY       # optional — mock available
USE_MOCKS=false          # set to true if keys are not yet available
```

### Substeps
- [ ] Read AGENTS.md fully — confirm understanding of anti-hallucination contract, mock strategy,
      test gate policy, and Docker policy
- [ ] Create project directory structure:
      bike-assistant/, backend/, frontend/, docs/, tests/, tests/fixtures/, scripts/
- [ ] Create .env from .env.example — fill in available keys, leave others blank
- [ ] Create .env.example with all key names and no values — safe to commit
- [ ] For each available API key, run a minimal connectivity test and log result
- [ ] For each unavailable key, confirm mock is wired and USE_MOCKS=true works
- [ ] Create tests/fixtures/sample_table.json — a realistic LlamaParse table fixture
- [ ] Create tests/fixtures/sample_response.json — a realistic OpenRouter structured output fixture
- [ ] User reviews and approves plan before proceeding

### Tests
- test_part_1.py: confirm .env loads correctly, confirm each available API returns 200,
  confirm mock mode activates when USE_MOCKS=true

### Local verification
Run: python tests/test_part_1.py
Expected: all available APIs green, all mocked APIs return fixture data, no import errors

### Success criteria
All available API keys confirmed live. All unavailable APIs confirmed mockable. Directory structure
created. Plan approved by user.

---

## Part 2: Project scaffolding — backend, local run scripts, hello world

### What this part does
Sets up the FastAPI backend, local run scripts, and confirms a working hello world before any
intelligence is added.

### API keys required for this part
None — this part has no external API calls.

### Substeps
- [ ] Set up Python project in backend/ using uv as package manager
- [ ] Create FastAPI app in backend/main.py — serves "hello world" HTML at /
- [ ] Create scripts/start_local.sh — starts backend locally without Docker
- [ ] Create scripts/stop_local.sh — stops local backend
- [ ] Create scripts/start_docker.sh and scripts/stop_docker.sh — stubbed, implemented in Part 12
- [ ] Confirm backend starts cleanly and hello world is visible at localhost:8000
- [ ] Add a /health route that returns { "status": "ok" }

### Tests
- test_part_2.py: GET / returns 200, GET /health returns { "status": "ok" },
  server starts and stops cleanly via scripts

### Local verification
Run: bash scripts/start_local.sh
Open: http://localhost:8000
Expected: hello world page visible, /health returns ok

### Success criteria
Backend starts locally via script, hello world visible at localhost, /health route works,
no Docker required.

---

## Part 3: Ingestion pipeline — the most critical part

### What this part does
Builds the full offline ingestion pipeline. This runs once per manual. Get this right before
building anything on top of it. Poor ingestion cannot be rescued later.

### API keys required for this part
```
OPENAI_API_KEY           # embeddings + GPT-4o vision for diagram descriptions
LLAMAPARSE_API_KEY       # table extraction
USE_MOCKS=true           # set this if keys not available — pipeline still runs with fixtures
```

### Substeps
- [ ] Content type classification pass — label every block as prose, table, image, warning,
      procedure before any chunking
- [ ] Prose extraction via PyMuPDF — preserve headings, section numbers, chapter structure
- [ ] Table extraction via LlamaParse — output structured key-value pairs, not flattened text.
      If USE_MOCKS=true, use tests/fixtures/sample_table.json
- [ ] Image/diagram extraction — PyMuPDF for position and page, GPT-4o vision for description,
      store image_path reference. If USE_MOCKS=true, use hardcoded description string
- [ ] Warning/caution block isolation — rule-based detector, always own chunk, never absorbed
      into adjacent prose
- [ ] Procedure chunking — full procedure as parent chunk, individual steps as child chunks,
      never split mid-step, extend to natural break if boundary falls mid-list
- [ ] Metadata tagging — apply full schema from AGENTS.md to every chunk
- [ ] Embed every chunk with OpenAI text-embedding-3-small. If USE_MOCKS=true, use fixed vectors
- [ ] Build BM25 index alongside vector index for each namespace
- [ ] Store in Qdrant under document_id namespace. If no QDRANT_URL, use in-memory Qdrant
- [ ] Generate document index JSON per manual and save to data/indexes/
- [ ] Run ingestion on one real manual (Royal Enfield Meteor 350 service manual PDF)
- [ ] Run 15-20 test queries across all content types. Observe re-ranker score distributions.
      Set confidence_threshold empirically. Document threshold in document index JSON

### Tests
- test_part_3.py: classification correctly labels all content types on sample PDF,
  table chunks output key-value format not flattened text, warning blocks are isolated chunks,
  procedure parent-child relationship correct, all metadata fields populated on every chunk,
  document index JSON generated and valid, BM25 index built, Qdrant namespace created,
  15-20 retrieval test queries return expected content types, confidence threshold set and logged

### Local verification
Run: python backend/ingestion/ingest.py --pdf path/to/manual.pdf --mock (if no keys)
Expected: terminal shows chunk counts per content type, document index JSON written to
data/indexes/, no errors

### Success criteria
All content types correctly classified and chunked. Metadata complete on every chunk.
15-20 test queries return expected chunks. Confidence threshold set and documented.
90% of retrieval test queries return relevant chunks. Mock mode produces valid pipeline output.

---

## Part 4: Retrieval and inference pipeline — core intelligence

### What this part does
Builds the full inference pipeline. This is the intelligence layer. Every query flows through this.

### API keys required for this part
```
OPENROUTER_API_KEY       # LLM response generation
COHERE_API_KEY           # re-ranking
OPENAI_API_KEY           # embeddings for query, query expansion LLM call
USE_MOCKS=true           # set this if keys not available
```

### Substeps
- [ ] Query expansion and intent classification — lightweight LLM call, language detection,
      synonym expansion (thak thak awaaz → knocking sound engine), classify as
      diagnostic / specification / procedure / out-of-scope
- [ ] Query routing — diagnostic maps to semantic dominant, specification maps to BM25 dominant,
      procedure maps to parent chunk retrieval dominant
- [ ] Hybrid retrieval — Qdrant vector search plus BM25, pre-filtered by document_id namespace,
      merged using Reciprocal Rank Fusion, retrieve top 5-7 chunks
- [ ] Cohere Rerank cross-encoder on top 5-7, keep top 3. If USE_MOCKS=true, return mock scores
- [ ] Confidence gate — check re-ranker scores against threshold from Part 3, flag low confidence
      queries, pass context_confidence: low into prompt
- [ ] LLM response generation — system prompt with hard anti-hallucination constraints, retrieved
      context labeled with source sections, last 4-6 turns of current thread only
- [ ] Structured output returned: answer_text, spoken_summary, citations, severity_label,
      confidence, suggested_followups
- [ ] Output validation — confirm citation present in response, re-generate once if missing,
      log warning if second generation also missing citation
- [ ] Wire history summarization — after turn 4 compress earlier turns into summary

### Tests
- test_part_4.py: 20 queries against indexed manual covering diagnostic, specification,
  procedure, out-of-scope, and ambiguous categories. Citation present on every in-scope answer.
  Clean refusal on every out-of-scope query. Structured output fields all populated.
  Confidence gate triggers on low-confidence queries. Output validation catches missing citations.
  History summarization activates at turn 4.

### Local verification
Run: python backend/inference/pipeline.py --query "white smoke from exhaust" --mock (if no keys)
Expected: structured JSON response printed to terminal with answer_text, citations, severity_label

### Success criteria
Citation present on all in-scope answers. Clean refusal on all out-of-scope queries.
Zero hallucinated answers. All structured output fields populated. 90% of 20 test queries
return correct result category.

---

## Part 5: Backend API routes

### What this part does
Exposes the ingestion and inference pipelines via FastAPI routes. This is the contract between
backend and frontend.

### API keys required for this part
Same as Parts 3 and 4 — all proxied through backend routes. USE_MOCKS=true works here too.

### Substeps
- [ ] POST /query — takes { text, session_id, document_id, thread_id }, returns structured response
- [ ] POST /ingest — accepts PDF file upload, triggers async ingestion job, returns { job_id }
- [ ] GET /ingest/status/{job_id} — returns { status, progress_pct, message }
- [ ] GET /bikes/library — returns list of pre-indexed library bikes with metadata
- [ ] POST /session — creates new session, returns { session_id }
- [ ] GET /session/{session_id}/threads — returns all issue threads for session
- [ ] GET /session/{session_id}/history — returns all resolved issues for session
- [ ] POST /session/{session_id}/threads — creates new issue thread, returns { thread_id }
- [ ] All routes return consistent error shapes: { error, message, code }

### Tests
- test_part_5.py: every route returns correct response shape, ingestion job lifecycle
  (submitted → processing → complete), session isolation confirmed (two sessions do not
  share state), error responses return correct shapes, all routes work with USE_MOCKS=true

### Local verification
Start backend: bash scripts/start_local.sh
Test route: curl -X POST http://localhost:8000/query -d '{"text":"white smoke","session_id":"test","document_id":"re_meteor_350","thread_id":"t1"}'
Expected: structured JSON response

### Success criteria
All routes return correct response shapes. Ingestion job lifecycle works end to end.
Sessions fully isolated. All routes functional with mock mode. 90% test pass rate.

---

## Part 6: Multimodal input — STT and image handling

### What this part does
Adds voice and image input handling at the backend. All inputs converge into one unified text
query before hitting the inference pipeline.

### API keys required for this part
```
SARVAM_API_KEY           # STT for Indic languages
OPENAI_API_KEY           # Whisper STT for English, GPT-4o vision for images
USE_MOCKS=true           # set this if keys not available
```

### Substeps
- [ ] POST /input/voice — accepts audio blob, detects language, routes to Sarvam STT (Indic)
      or Whisper (English), returns { transcript, language, confidence }
- [ ] STT confidence check — if confidence below threshold, return { needs_retry: true } so
      frontend can prompt user to retry or type
- [ ] POST /input/image — accepts image file, sends to GPT-4o vision, returns
      { description, technical_terms }. If USE_MOCKS=true, return fixture description
- [ ] Unified query assembly — merge voice transcript + image description + any text into
      one query string before inference pipeline
- [ ] Language stored in session on first detected input
- [ ] Wire unified query into existing /query route

### Tests
- test_part_6.py: Hindi audio file routes to Sarvam mock, English audio routes to Whisper mock,
  low confidence STT returns needs_retry flag, image input returns mechanically meaningful
  description, unified assembly merges all three input types correctly, language stored in session

### Local verification
Test voice: curl -X POST http://localhost:8000/input/voice -F "audio=@tests/fixtures/sample_hindi.wav"
Test image: curl -X POST http://localhost:8000/input/image -F "image=@tests/fixtures/sample_exhaust.jpg"
Expected: transcript and image description returned correctly

### Success criteria
Hindi voice routes to Sarvam, English routes to Whisper, STT confidence check triggers
correctly, image description is retrieval-useful, unified query assembly works for all
input combinations. 90% test pass rate.

---

## Part 7: TTS output

### What this part does
Adds voice output for voice-initiated queries. spoken_summary is sent to TTS, not the full answer.

### API keys required for this part
```
SARVAM_API_KEY           # TTS for Indic language responses
OPENAI_API_KEY           # TTS for English responses
USE_MOCKS=true           # set this if keys not available — returns text only, no audio
```

### Substeps
- [ ] POST /output/tts — accepts { text, language }, returns audio stream
- [ ] Route to Sarvam TTS if Indic language, OpenAI TTS if English
- [ ] Always send spoken_summary field only — never answer_text
- [ ] Enforce spoken_summary max length: 3 sentences, re-truncate if LLM returns longer
- [ ] If USE_MOCKS=true, skip audio generation and return { mocked: true, text: spoken_summary }
- [ ] Wire TTS call into /query route — if session has voice_initiated: true, attach TTS audio
      URL to response

### Tests
- test_part_7.py: Indic language routes to Sarvam mock, English routes to OpenAI mock,
  spoken_summary never exceeds 3 sentences, full answer_text never sent to TTS,
  mock mode returns text confirmation not audio, TTS attached to voice-initiated queries only

### Local verification
Run a voice query end to end via /query with voice_initiated: true in session
Expected: response includes audio_url field, spoken_summary is 2-3 sentences max

### Success criteria
Correct TTS routing by language. spoken_summary length constraint enforced. Full answer_text
never sent to TTS. Mock mode works without audio keys. 90% test pass rate.

---

## Part 8: Frontend — full application

### What this part does
Builds the Next.js frontend. Statically built and served by FastAPI at /. No backend logic
lives in the frontend — it only calls backend API routes.

### API keys required for this part
None — frontend calls backend routes only.

### Substeps
- [ ] Next.js app in frontend/ — statically exported, served by FastAPI at /
- [ ] Bike selection screen — brand dropdown, model list, year selector, fuzzy search,
      upload manual option
- [ ] Upload flow — PDF file picker, progress bar polling /ingest/status/{job_id} every 2s,
      conversation unlocks when status is complete
- [ ] Home / conversation screen — conversation thread area, 2-3 suggested starter questions,
      unified input bar always visible at bottom
- [ ] Unified input bar — mic button (tap to record), image attach icon, text field,
      any combination supported, no mode selection required
- [ ] Response card — answer_text, citation block (section number, title, page), severity label
      badge, suggested follow-up questions, audio player if TTS available
- [ ] Out-of-scope response rendered clearly — no hedging, plain refusal message shown
- [ ] Issue thread switcher — multiple open threads shown, switching preserves independent context
- [ ] New issue detection prompt — "Looks like a new issue — track separately?" inline prompt
- [ ] History view — one tap from home, one line per resolved issue, severity badge,
      status badge, expandable for full answer summary
- [ ] Returning user flow — bike remembered from previous session, skip selection screen
- [ ] No unnecessary animations. No decorative elements. Clean, functional, fast.

### Tests
- test_part_8.test.ts: bike selection flow completes, upload progress polling works,
  all three input types submit correctly, response card renders all fields, out-of-scope
  response renders refusal message, thread switching preserves context, history view
  populates from resolved threads, returning user skips selection

### Local verification
Start frontend: cd frontend && npm run dev
Open: http://localhost:3000
Walk through: select bike → ask text question → confirm cited answer renders → ask voice
question → confirm TTS plays → ask out-of-scope question → confirm clean refusal →
open second thread → confirm context isolation → resolve thread → confirm history entry

### Success criteria
All flows work end to end. Context isolation confirmed between threads. History correctly
populated. No hallucinated answers in any test flow. 90% test pass rate.

---

## Part 9: Multi-bike library and pre-indexing

### What this part does
Pre-indexes library bikes and wires up library selection in the frontend.

### API keys required for this part
```
OPENAI_API_KEY           # embeddings for library bike ingestion
LLAMAPARSE_API_KEY       # table extraction for library manuals
USE_MOCKS=true           # use if keys not available — fixtures used for ingestion
```

### Substeps
- [ ] Source PDFs: Royal Enfield Meteor 350, TVS Apache RTR 160, Bajaj Pulsar NS200 —
      service manuals and owner manuals (publicly available)
- [ ] Run ingestion pipeline on each manual — confirm clean indexing
- [ ] Run 5 test queries per bike to confirm retrieval quality
- [ ] Confirm namespace isolation — Royal Enfield query must never return TVS chunks
- [ ] Wire /bikes/library route to return all pre-indexed bikes
- [ ] Wire library dropdown in frontend to /bikes/library
- [ ] Test upload precedence — upload custom manual for same bike as library entry,
      confirm uploaded version is used for retrieval

### Tests
- test_part_9.py: all three library bikes indexed, namespace isolation confirmed across
  bikes, 5 retrieval queries per bike return correct bike's chunks, upload precedence
  rule enforced correctly

### Local verification
Select Royal Enfield from library dropdown — confirm it loads
Ask a torque spec question — confirm answer cites Royal Enfield manual
Switch to TVS — ask same category question — confirm TVS manual is cited

### Success criteria
Namespace isolation confirmed across all library bikes. Upload precedence rule works.
15 retrieval test queries (5 per bike) return correct results. 90% test pass rate.

---

## Part 10: Multi-language support end to end

### What this part does
Wires language handling across the full pipeline and confirms Hindi end-to-end flow works.

### API keys required for this part
```
SARVAM_API_KEY           # STT + TTS for Hindi and Indic languages
USE_MOCKS=true           # set if key not available
```

### Substeps
- [ ] Language detection on every turn, stored in session
- [ ] STT routing confirmed working for Hindi and English
- [ ] Query expansion handles transliterated input — "thak thak awaaz" → "knocking sound engine",
      "engine garam ho rahi hai" → "engine overheating"
- [ ] LLM response generated in detected language — system prompt updated with language instruction
- [ ] TTS response in same language as query
- [ ] Language switch mid-conversation handled — re-detect each turn, no hard lock
- [ ] Add at least 5 transliteration mappings to query expansion — common Hindi symptom phrases
      mapped to English technical terms for retrieval

### Tests
- test_part_10.py: Hindi query routes to Sarvam STT mock, transliterated queries expand
  to correct technical terms, LLM response language matches query language,
  TTS language matches query language, mid-conversation language switch handled

### Local verification
Type a Hindi query: "exhaust se safed dhuaan aa raha hai"
Expected: query expands to "white smoke from exhaust", answer returned in Hindi,
TTS response in Hindi if Sarvam key available

### Success criteria
End-to-end Hindi conversation works. Language switching graceful. 5 transliterated queries
expand correctly. 90% test pass rate.

---

## Part 11: Hallucination regression suite

### What this part does
Dedicated test suite for the anti-hallucination contract. Must pass before submission.
This is the final quality gate.

### API keys required for this part
Same as Parts 4 and 5 — runs live inference. USE_MOCKS=false recommended for this part.

### Test cases — all 31 must pass

- 10 in-scope questions with known correct answers from manual — verify citation accuracy
  and answer correctness
- 10 out-of-scope questions — verify clean refusal on every one, no hedging, no partial answers
- 5 ambiguous questions near scope boundary — verify conservative behavior, no hallucination
- 3 questions with retrieval confidence below threshold — verify confidence gate triggers and
  passes context_confidence: low to LLM
- 2 questions where LLM might answer from training data — verify output validation catches
  missing citation and re-generates or refuses
- 1 safety-critical question involving a WARNING chunk — verify warning chunk retrieved,
  severity label is Urgent or Get Checked Soon

### Tests
- test_part_11.py: all 31 test cases defined with expected outcomes. Zero hallucinations
  acceptable. Citation present on all in-scope answers. Clean refusal on all out-of-scope.
  Confidence gate triggers on low-confidence cases.

### Local verification
Run: python tests/test_part_11.py
Expected: 31/31 pass. Any failure is a blocker — fix before proceeding.

### Success criteria
31/31 test cases pass. Zero hallucinations. All guardrail layers confirmed working.
This part cannot be passed at 90% — it requires 100%.

---

## Part 12: Docker packaging, polish, and submission prep

### What this part does
Packages everything into Docker, cleans up, and prepares submission artifacts.

### API keys required for this part
None new — Docker just wraps what already works locally.

### Substeps
- [ ] Write Dockerfile — multi-stage build, backend serves static frontend build at /
- [ ] Write docker-compose.yml — mounts .env, exposes port 8000
- [ ] Update scripts/start_docker.sh and scripts/stop_docker.sh
- [ ] Confirm Docker build is clean from scratch — no cached layers assumed
- [ ] Confirm full end-to-end flow works inside Docker container
- [ ] README — minimal: setup steps, API keys required, how to run locally, how to run in Docker,
      how to ingest a manual, how to run tests. No emojis. No marketing language.
- [ ] .env.example — all key names with no values and one-line comment per key
- [ ] AGENTS.md final review — confirm all decisions have one-line justifications
- [ ] Run full hallucination regression suite (Part 11) one final time
- [ ] Record Loom walkthrough — bike selection, text query with citation, voice query in Hindi,
      image input with diagnosis, out-of-scope refusal, issue threading, history view

### Tests
- test_part_12.py: Docker build succeeds, container starts cleanly, /health route returns ok
  inside container, full end-to-end query works inside container

### Local verification
Run: bash scripts/start_docker.sh
Open: http://localhost:8000
Walk through full demo flow inside Docker

### Success criteria
Docker builds cleanly from scratch. All flows work inside container. README sufficient for
evaluator to run locally in under 10 minutes. Loom covers all key flows. Part 11 regression
suite passes 31/31 on final run.
