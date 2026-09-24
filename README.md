# AnonAgent

A local privacy layer between sensitive data and cloud LLMs.

Sensitive text is analysed on this machine, every private value is swapped for
a placeholder, and only the placeholder version is sent to the model. The
answer comes back, the real values are restored locally, and the user reads a
normal answer. The provider never sees a name, an account number or a key.

```
Check whether James Bond has funds for 50,000 RSD.     <- stays local
Check whether <PERSON_001> has funds for 50,000 RSD.   <- sent to cloud
Check whether James Bond has funds for 50,000 RSD.     <- restored locally
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 — not 3.13, where
the spaCy build Presidio needs does not install.

```bash
uv sync    # also fetches the 560 MB spaCy model
```

To reach a cloud model, put a `.env` beside `pyproject.toml`. It is gitignored
and must stay that way:

```bash
OPENAI_API_KEY=sk-...           # or GROQ_API_KEY, for --provider groq
LLM_MODEL=gpt-4o-mini           # optional, this is the default
ANONAGENT_SECRET=any-long-string  # optional: keeps fingerprints stable
```

Without it, everything except `--ask` and `--chat` still works.

## Use

Everything except `--ask` and `--chat` runs offline, with no API key:

```bash
uv run streamlit run app.py             # three-column interface

uv run anonagent                        # built-in sample: banking, medical, code
uv run anonagent "Call Maria at +1 415 555 0182"
echo "..." | uv run anonagent -         # "-" reads stdin

uv run anonagent --file record.txt      # writes record.masked.txt
uv run anonagent --ask                  # send the masked text to a model
uv run anonagent --chat                 # follow-up questions in one session
uv run anonagent --review               # rule on uncertain detections yourself
uv run anonagent --score-threshold 0.9  # watch the guard come apart
```

Exit codes: `0` fine, `1` bad usage, `2` a leak was blocked, `3` the model
could not be reached.

## Tests

```bash
uv run pytest                  # everything
uv run pytest -m "not slow"    # skip the tests that load spaCy
```

## How it works

| Module                                                           | Responsibility                                                    |
| ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`privacy/detector.py`](src/anonagent/privacy/detector.py)       | Finds sensitive spans; resolves overlapping claims                |
| [`privacy/recognizers.py`](src/anonagent/privacy/recognizers.py) | Identity numbers, labelled fields, employee IDs, API keys, money  |
| [`privacy/vault.py`](src/anonagent/privacy/vault.py)             | The two-way mapping. The only thing that can undo a placeholder   |
| [`privacy/masker.py`](src/anonagent/privacy/masker.py)           | `mask()` and `unmask()`, plus the sweep and the leak check        |
| [`documents.py`](src/anonagent/documents.py)                     | Splits long text into chunks without cutting through a value      |
| [`review.py`](src/anonagent/review.py)                           | Puts uncertain detections to the user, failing safe at every turn |
| [`events.py`](src/anonagent/events.py)                           | Structured trace of each step, for the console and the screen     |
| [`agent/llm.py`](src/anonagent/agent/llm.py)                     | Builds the chat model: OpenAI, Groq or local Ollama               |
| [`agent/pipeline.py`](src/anonagent/agent/pipeline.py)           | mask, ask, unmask, recording every step                           |
| [`app.py`](app.py)                                               | The three-column interface                                        |
