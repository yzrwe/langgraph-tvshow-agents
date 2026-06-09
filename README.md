# TVShowAgents — Multi-Agent Renewal Decisions (LangGraph + Claude)

A LangGraph multi-agent system that decides whether a TV show should be
renewed, renewed conditionally, given a final season, or canceled — using
live data from the TMDB API.

Originally built as a Colab group project for UT Dallas FIN 6327; ported
here as a runnable script.

## Architecture

```
START
  │
  ▼
5 tool-calling analyst agents (sequential, Claude Haiku)
  performance → audience → market → financial → critical
  each with conditional routing: analyst ──(tool_calls?)──► ToolNode ──► next
  │
  ▼
Structured debate (renew advocate vs. cancel advocate)
  │
  ▼
Programming Director (Claude Sonnet) — weighs evidence, recommends
  │
  ▼
Network Executive (Claude Sonnet) — final call:
  RENEW / RENEW_CONDITIONAL / FINAL_SEASON / CANCEL
```

What it demonstrates:

- **Typed graph state** — `AgentState(MessagesState)` with per-analyst report
  fields and a nested `RenewalDebateState` TypedDict.
- **Conditional edges on tool calls** — each analyst routes through its own
  scoped `ToolNode` only when the model actually requested tools.
- **Model tiering** — cheap/fast model (Haiku) for the five data-gathering
  analysts; deep model (Sonnet) reserved for the two decision nodes.
- **Adversarial synthesis** — a renew advocate argues from
  performance/audience evidence while a cancel advocate argues from
  financial/critical evidence, before a judge decides.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # console.anthropic.com
export TMDB_API_KEY=...        # themoviedb.org read access token
python tvshow_agents.py "Stranger Things"
```

Example output (real run): for *Stranger Things*, the system returned
**FINAL_SEASON** — protect the franchise's legacy, control escalating cast
costs, and pivot to spin-offs while the brand is at its peak.

---

*Regis Yizerwe — yizerwer@gmail.com*
