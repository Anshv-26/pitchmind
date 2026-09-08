# PitchMind

Agentic Premier League intelligence. Deterministic Python/ML produces every
number; the agent layer (not yet built) will only interpret and explain them.

## Running the backend

```bash
pip install -r requirements.txt
uvicorn backend.app.api.main:app --reload
```

The API starts with **no credentials**. `/health`, the live (replay-backed)
routes, and both prediction routes work offline. The current-data routes need
`PITCHMIND_FOOTBALL_DATA_ORG_KEY` (see `.env.example`) and return `503` with an
actionable message without it.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Network-free liveness check |
| GET | `/api/v1/standings` | Current PL table (needs API key) |
| GET | `/api/v1/fixtures` | Fixtures **and** results; filters: `team`, `status`, `date_from`, `date_to`, `limit` |
| GET | `/api/v1/live/matches` | All in-play matches |
| GET | `/api/v1/live/matches/{provider_fixture_id}` | One match's live state |
| POST | `/api/v1/predict` | H/D/A probabilities from the frozen model |
| POST | `/api/v1/predict/explain` | Exact contribution breakdown for that prediction |

```bash
curl localhost:8000/health
curl "localhost:8000/api/v1/fixtures?team=Arsenal&status=FINISHED&limit=5"
curl localhost:8000/api/v1/live/matches
curl -X POST localhost:8000/api/v1/predict \
     -H 'content-type: application/json' \
     -d '{"elo_diff": 85.0, "diff_ewma_ppg": 0.42, "diff_ewma_sot_diff": 1.3}'
```

Interactive docs at `http://localhost:8000/docs` once running.

## Two things to know when reading responses

**Live data is replay data.** API-Football's free tier was verified not to
expose the current Premier League season, so live routes are served by a
deterministic `ReplayProvider`. Every response says so:
`provenance.source_kind == "REPLAY"`. Nothing presents scripted data as real.

**Predictions are not production-validated.** The frozen strength-trio model
was trained through 2024/25 and the sealed 2025/26 final test has not been run.
Every prediction response carries `model_provenance.sealed_final_test_completed
= false`. The endpoint scores a caller-supplied pre-kickoff feature snapshot; it
never builds current-season features and never retrains.

## Architecture

```
HTTP route  ->  deterministic tool  ->  football-data service / frozen model
```

No LLM sits in that path. `backend/app/tools/` is the boundary a future Claude
Agent SDK will call — agents get small typed functions, never provider payloads,
pandas frames, or model internals.

## Tests

```bash
pytest -q
```

Network-free by design; providers are mocked and no API key is required.
