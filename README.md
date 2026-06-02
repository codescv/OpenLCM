# OpenLCM — Lossless Context Management for LLM Agents

**Unbounded memory. Bounded context.**

OpenLCM is a framework-agnostic Python SDK that gives your AI agents a permanent, lossless memory — without ever hitting the context limit. Every message is persisted verbatim in SQLite and compressed into a hierarchical DAG of summaries. Nothing is ever lost. Any past moment is recoverable.

```bash
pip install openlcm
```

---

## The problem

LLMs have a hard token limit. As conversations grow, agents either crash or replace old turns with a flat, irreversible summary. Details fall out permanently — decisions, constraints, file paths, tool results.

## How LCM works

OpenLCM maintains two layers:

1. **Immutable message store** — every message written verbatim to SQLite with a stable `store_id`. FTS5-indexed. Never modified.
2. **Summary DAG** — older messages are compressed into D0 leaf nodes → D1 session arcs → D2 durable history. Each node points back to its source messages for exact recovery.

The model always sees: `system + highest DAG node + recent D0 nodes + fresh tail (last N raw messages)`.

---

## Quick start

```python
from openlcm import LCMEngine

engine = LCMEngine(model="anthropic/claude-haiku-4-5-20251001")
engine.bind_session("my-session", context_length=200_000)

# Call before every LLM turn — compresses only when needed
messages = await engine.compress(messages)
```

Pass any [LiteLLM](https://github.com/BerriAI/litellm) model string: `openai/gpt-4o`, `gemini/gemini-2.0-flash`, `azure/gpt-4o`, `bedrock/...`, `ollama/llama3`, etc.

---

## Framework adapters

All adapters are included — one install, no extras needed.

### LangGraph

```python
from openlcm.adapters.langgraph import LCMCheckpointer

graph = StateGraph(MyState).compile(
    checkpointer=LCMCheckpointer(llm=my_llm)
)
```

### Google ADK

```python
from openlcm.adapters.google_adk import LCMSessionService, lcm_compress_callback

agent = LlmAgent(
    name="assistant",
    model="gemini-2.0-flash",
    tools=[...],
    before_model_callback=lcm_compress_callback(engine),
)
runner = Runner(agent=agent, session_service=LCMSessionService(engine))
```

### AutoGen

```python
from openlcm.adapters.autogen import LCMContext

agent = AssistantAgent(
    "assistant",
    model_client=client,
    model_context=LCMContext(llm=client),
)
```

### CrewAI

```python
from openlcm.adapters.crewai import LCMStorage

crew = Crew(
    memory=True,
    long_term_memory=LongTermMemory(storage=LCMStorage(engine))
)
```

### OpenAI / Groq / Mistral / Ollama

```python
from openlcm.adapters.openai import OpenAIMessages

lcm = OpenAIMessages.to_lcm(messages)
if engine.should_compress_preflight(lcm):
    lcm      = await engine.compress(lcm)
    messages = OpenAIMessages.from_lcm(lcm)
```

### Anthropic

```python
from openlcm.adapters.anthropic import AnthropicMessages

lcm = AnthropicMessages.to_lcm(messages, system=system_prompt)
lcm = await engine.compress(lcm)
system_out, anthropic_msgs = AnthropicMessages.from_lcm(lcm)
```

### LlamaIndex / Haystack / Gemini

```python
from openlcm.adapters.llamaindex import LlamaIndexMessages
from openlcm.adapters.haystack   import HaystackMessages
from openlcm.adapters.gemini     import GeminiMessages
```

All follow the same `to_lcm()` / `from_lcm()` interface.

---

## Configuration

```python
from openlcm.core.config import LCMConfig

config = LCMConfig.from_env()
config.context_threshold  = 0.75   # compress at 75% of context window
config.fresh_tail_count   = 64     # protect last 64 messages from compression
config.leaf_chunk_tokens  = 20_000 # tokens per D0 leaf summary
config.condensation_fanin = 4      # D0 nodes before a D1 arc is created

engine = LCMEngine(model="...", config=config)
```

| Env var | Default | Description |
|---|---|---|
| `LCM_CONTEXT_THRESHOLD` | `0.75` | Compression trigger as fraction of context window |
| `LCM_FRESH_TAIL_COUNT` | `64` | Messages protected from compression at tail |
| `LCM_LEAF_CHUNK_TOKENS` | `20000` | Tokens per D0 summary chunk |
| `LCM_CONDENSATION_FANIN` | `4` | D0 nodes required before D1 arc is created |

---

## Live dashboard

```python
import threading
from openlcm.viz.server import create_app, serve as viz_serve

threading.Thread(
    target=lambda: viz_serve(create_app(engine), port=7842, open_browser=True),
    daemon=True
).start()
```

Or from the CLI:

```bash
openlcm viz          # opens http://localhost:7842
openlcm grep "query" # full-text search across all sessions
openlcm status       # session stats
```

The dashboard shows token pressure, DAG graph, SQLite message store, and a live event log — all updating in real time.

---

## Guarantees

- **Lossless** — every message persisted with stable `store_id`. Recoverable even after 100 compactions.
- **Deterministic** — summarization always terminates. L1 → L2 → L3 escalation with circuit breaker.
- **Zero-cost** — compression fires only when the threshold is exceeded. Short conversations pay zero overhead.

---

## License

MIT — see [LICENSE](LICENSE).

Built on the LCM paper by Ehrlich & Blackman (Voltropy, 2026).
