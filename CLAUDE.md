# PitchMind — Engineering Instructions

PitchMind is an agentic AI Premier League intelligence platform, built as
a 12-day portfolio project. It combines historical and current Premier
League data, traditional ML, statistical modelling, explainable ML, and
agentic AI (Anthropic's Claude Agent SDK) behind a FastAPI backend and a
Next.js frontend.

The implementation must remain understandable enough that the developer
can explain every major architectural and ML decision in a technical
interview. Favor reliable, understandable, reproducible, portfolio-quality
solutions over unnecessary enterprise complexity — but do not remove
planned features solely to save time or simplify.


## Repository Architecture

```text
pitchmind/
|
|-- backend/
|   `-- app/
|       |-- agents/
|       |-- api/
|       |-- core/
|       |-- ml/
|       |-- services/
|       `-- tools/
|
|-- frontend/
|
|-- data/
|   |-- raw/
|   `-- processed/
|
|-- models/
|-- notebooks/
|-- scripts/
|-- tests/
|-- docs/
|
|-- CLAUDE.md
|-- README.md
`-- .gitignore
```

Maintain these boundaries unless there is a clear technical reason to
change them.


## Technology Stack

**Backend:** Python, FastAPI

**Data / ML:** pandas, NumPy, scikit-learn, XGBoost, SHAP, Poisson
regression

**Database:** PostgreSQL, once persistence becomes necessary

**Frontend:** Next.js, React, TypeScript, Tailwind CSS

**Agentic AI:** Anthropic Claude Agent SDK, exclusively

**Do NOT introduce** LangChain, LangGraph, CrewAI, AutoGen, or any other
agent framework unless explicitly requested by the developer.


## Planned Features (retain all of these)

1. Historical Premier League data pipeline
2. Current/live Premier League data
3. Match-result probability prediction
4. Poisson-based goal/score prediction
5. Explainable ML using SHAP
6. Statistics Agent
7. ML Agent
8. Research Agent
9. Tactical Analysis Agent
10. Critic / Verification Agent
11. Orchestrator Agent
12. Conditional/cost-aware agent routing
13. What-if match simulator
14. Player similarity/scouting system
15. Caching
16. FastAPI backend
17. Next.js frontend
18. Structured outputs between application components
19. Testing and evaluation
20. Portfolio-quality documentation


## Fundamental Architecture: Two Intelligence Layers

**A. Deterministic / ML layer.**

Normal Python, statistics, and trained models are responsible for:

- calculations
- aggregations
- rolling statistics
- feature engineering
- match probabilities
- goal predictions
- SHAP values
- similarity calculations
- scenario calculations
- database queries


**B. Claude / agent layer.**

Claude is responsible for:

- understanding user intent
- planning
- deciding which tools are required
- selecting specialist agents
- interpreting structured evidence
- current-context research when required
- tactical synthesis
- verification
- natural-language explanation

Claude must **never** replace deterministic computation when normal
Python or a trained model can perform the task reliably.


## Prediction Integrity

LLMs must never invent:

- match probabilities
- expected goals
- predicted score probabilities
- SHAP feature contributions
- player similarity scores
- calculated football statistics

These values must originate from deterministic tools or trained models.

- **Bad:** Claude decides Arsenal has a 63% win probability.
- **Good:** XGBoost returns a 0.63 Arsenal win probability and Claude
  explains what that value means.


## ML Models

PitchMind builds and evaluates its own models.

- **Match-outcome models:**
  - Multinomial Logistic Regression
  - Random Forest
  - XGBoost

- **Goal modelling:**
  - Poisson-based model

- **Explainability:**
  - SHAP

- **Player scouting:**
  - normalized player feature vectors
  - deterministic similarity calculation such as cosine similarity

Claude models are not substitutes for these statistical/ML models.


## ML Data-Leakage Rules (critical)

Every model feature must represent information available **before
kickoff** of the match being predicted.

Never use statistics from the current match as prediction inputs.

For rolling statistics, shift historical observations before calculating
the rolling window — the current match must never contribute to its own
features.

- **Correct:** `shift(1).rolling(...)`
- **Incorrect:** `rolling(...)` when it includes the current observation

Additional rules:

- Season-to-date cumulative statistics must exclude the current match.
- No future match may influence an earlier match's features.
- Never randomly shuffle matches for final football-model evaluation.
- Use chronological/time-based train/validation/test splits.
- Preprocessing such as scaling, imputation, or encoding must be fitted
  using training data only.
- Set random seeds wherever randomness is used.
- Preserve `predict_proba` outputs.
- Evaluate probability quality, not only classification accuracy.
- Track at minimum where appropriate:
  - accuracy
  - log loss
  - confusion matrix
  - precision
  - recall
  - F1
  - probability calibration
  - Brier-style metrics


## Betting Data

Football-Data and similar source files may contain bookmaker odds and
other betting-market-derived columns.

For PitchMind's primary prediction models:

- Do NOT use bookmaker odds as model inputs.
- Do NOT use betting-market probabilities as model inputs.
- Do NOT use derived bookmaker consensus features.
- Betting-related columns should be excluded during data processing.

The primary PitchMind models should learn from football-performance data,
not simply reproduce bookmaker expectations.

Betting data may only be used later in a clearly separated benchmark or
comparison experiment if explicitly requested by the developer.


## Data Rules

- Raw source data (`data/raw`) is immutable.
- Never manually edit raw source datasets.
- Processed datasets (`data/processed`) must be reproducible from scripts.
- Normalize team names consistently.
- Parse dates explicitly.
- Sort matches chronologically before calculating historical features.
- Validate:
  - duplicates
  - invalid results
  - unexpected missing values
  - inconsistent team names
  - unexpected schema changes
  - chronological ordering
- Generated datasets and trained model artifacts should not be manually
  modified.


## Agent Architecture

Target runtime system:

```text
User
  |
  v
FastAPI
  |
  v
Orchestrator Agent
  |
  +-- Statistics Agent
  |
  +-- ML Agent
  |
  +-- Research Agent
  |
  +-- Tactical Analysis Agent
  |
  `-- Critic Agent
  |
  v
Structured Response
  |
  v
Frontend
```

Agents communicate using compact structured outputs wherever practical.

Do not pass giant raw datasets into Claude context.

Instead expose narrow deterministic tools such as:

- `get_team_form()`
- `get_recent_matches()`
- `get_standings()`
- `compare_teams()`
- `predict_match()`
- `predict_score()`
- `explain_prediction()`
- `simulate_scenario()`
- `find_similar_players()`

The Orchestrator must call **only** the minimum specialist agents
necessary for the current request.

Not every request should invoke every agent.


### What-If Simulator

Scenario calculations must be deterministic wherever possible.

The ML/statistical layer calculates baseline and modified probabilities.

Claude explains differences **after** the calculations are produced and
must never fabricate the numerical effect of a scenario.


### Player Scouting

Player similarity must be based on actual player features and
deterministic similarity calculations.

Claude may:

- interpret player profiles
- explain similarities
- discuss tactical fit

Claude may not fabricate player similarity scores.


### Research Agent

Current injuries, suspensions, and team news require current evidence.

Do not treat model memory as authoritative for current football
information.

Research output must clearly distinguish:

- verified facts
- uncertain information
- interpretation


### Critic Agent

The Critic Agent should verify at minimum:

- prediction numbers correspond exactly to tool/model output
- claims are supported by available evidence
- numerical values have not been invented
- contradictory specialist outputs are identified
- uncertainty is represented appropriately

The Critic should return compact structured verification rather than
rewriting the entire analysis unnecessarily.


## Claude Model Routing

Always use the least expensive / lowest-resource model that can perform
the task reliably.


### Runtime Target — PitchMind Agents

**Haiku**

Use for lightweight, narrow, structured specialist work such as:

- simple Statistics Agent interpretation
- routine ML tool-result interpretation
- straightforward Research Agent extraction
- structured verification by the Critic Agent


**Sonnet**

Use for:

- normal orchestration
- multi-source synthesis
- tactical interpretation
- moderately complex reasoning
- difficult specialist tasks where Haiku is insufficient


**Opus**

Opus must be exceptional, not routine.

Do **not** use Opus by default in the PitchMind runtime.

Only consider Opus for an exceptionally difficult reasoning request when
all of the following hold:

1. Sonnet has demonstrably failed or is inadequate.
2. Deterministic tools cannot solve the reasoning problem.
3. The extra reasoning quality materially improves the result.
4. Using Opus is explicitly permitted by configuration/developer policy.

There must never be an automatic escalation that silently sends every
complex request to Opus.


### During Development — Claude Code

Opus 5 may be selected temporarily for genuinely difficult tasks such as:

- subtle architecture problems
- very difficult bugs
- complex concurrency/state-management failures
- adversarial ML leakage review when Sonnet cannot resolve an issue
- large repo-wide refactoring decisions
- difficult multi-agent orchestration design

Normal coding should use Sonnet.

After the difficult task is solved, return to Sonnet.

Fable or other highest-cost/heaviest models should not be introduced
without explicit developer approval.


## Model Configuration

Do not scatter Claude model identifiers throughout the codebase.

Centralize model configuration so:

- Haiku
- Sonnet
- Opus

versions can be changed without modifying agent logic.

Agent implementations should reference centralized configuration rather
than hardcoded model IDs wherever practical.

This configuration should also make it possible to:

- disable Opus runtime usage entirely
- change models without rewriting agents
- enforce cost-aware model routing
- configure development and production behavior separately


## Cost / Subscription Constraints

This is a personal portfolio project with a strict cost constraint.

**Development target:**

- Claude Pro subscription
- ChatGPT Plus subscription
- zero additional Anthropic API spending

**Do NOT:**

- create or require an `ANTHROPIC_API_KEY` without explicit approval
- switch Claude Code to Console/API billing
- enable paid usage credits automatically
- introduce paid LLM providers
- introduce paid external APIs without explicit approval
- introduce infrastructure that requires payment without first
  identifying the cost and asking for approval

If Claude subscription usage is exhausted, prefer waiting for the usage
window to reset rather than silently enabling paid usage.

Use free/open data sources and free development tiers wherever practical.


## Public Deployment Authentication

Do not assume that personal Claude subscription authentication may be
used to serve unrestricted public users.

If a deployment architecture would require:

- separately billed Anthropic API authentication
- paid Claude API credits
- another paid LLM provider
- another paid external service

stop and inform the developer before implementing it.

Do not silently convert the application from subscription-backed
development into paid API usage.

Keep Claude provider/model/authentication configuration isolated from
business logic so authentication can be changed later without rewriting
the agent architecture.

The project should remain fully usable for local development and
controlled portfolio demonstration under the project's current cost
constraints.


## Cost-Efficient Agent Design

Minimize LLM usage without reducing functionality.

1. **Conditional routing**
   - Only invoke agents actually needed for the request.

2. **Deterministic computation**
   - Python performs arithmetic, statistics, feature computation, and data
     processing whenever possible.

3. **Compact structured agent outputs**
   - Avoid long essays between agents.

4. **Caching**
   - Cache reusable statistics, predictions, research results, and analyses.

5. **Small tool responses**
   - Retrieve only data relevant to the current request.

6. **Limited agent turns**
   - Prevent unnecessary agent loops.

7. **Context discipline**
   - Do not repeatedly send complete datasets or irrelevant conversation
     history to Claude.

8. **Model routing**
   - Haiku for simple work
   - Sonnet for substantial reasoning
   - Opus for exceptional cases only


## Code Quality

- Use clear module boundaries.
- Use Python type hints where practical.
- Use docstrings for important public functions.
- Prefer small, testable functions.
- Avoid unnecessary abstraction.
- Avoid premature optimization.
- Avoid unnecessary dependencies.
- Never hardcode secrets.
- Never commit `.env` files.
- Never commit API credentials.
- Never modify unrelated modules for a focused task.
- Run relevant tests after changes.
- Explain significant architectural changes before implementing them.


## Development Workflow

For every non-trivial task:

1. Read `CLAUDE.md`.
2. Inspect relevant existing code.
3. Understand current behavior.
4. Propose a focused implementation plan.
5. Wait for approval when architectural changes are involved.
6. Implement the smallest coherent change.
7. Run relevant tests.
8. Inspect failures rather than hiding them.
9. Fix genuine problems.
10. Summarize exactly what changed.

Do not generate dozens of unrelated files from one broad prompt.

Prefer incremental development.


## Git Rules

Keep commits focused and meaningful.

Never commit:

- secrets
- API keys
- `.env`
- virtual environments
- generated caches
- large generated datasets unless intentionally approved
- unnecessary model binaries

Do not rewrite Git history or perform destructive Git operations unless
explicitly requested.


## 12-Day Constraint

PitchMind must be completed within a 12-day sprint.

Favor:

- reliable
- understandable
- reproducible
- portfolio-quality

solutions over unnecessary enterprise complexity.

Do not remove planned PitchMind features solely to save time.


## Final Operating Principle

PitchMind should use:

```text
Claude = reasoning, planning, interpretation, orchestration

Python / ML / statistics = calculations, predictions, similarity,
                           feature engineering, numerical analysis
```

When deciding whether Claude or deterministic code should perform a task,
prefer deterministic code whenever the task can be reliably expressed as
normal computation.

The agentic layer should make the application smarter — it should not
replace conventional software engineering or statistical modelling.