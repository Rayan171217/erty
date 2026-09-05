# JARVIS - Personal AI Assistant

A LangGraph-based personal assistant powered by Claude (Anthropic API). It
routes each request through a supervisor to specialized agents — scheduling,
research, and communication — with a risk-manager gate and human approval
before any sensitive action is executed.

## How it works

1. **Supervisor** classifies the request into `schedule`, `research`,
   `communication`, or `general`.
2. The matching agent node produces a proposal or answer:
   - **Scheduler** drafts a meeting/reminder proposal (enforces a hard daily
     meeting limit).
   - **Research** answers factual questions without inventing facts, and
     flags requests that need a live API (weather, stocks, news).
   - **Communication** drafts emails/messages for review.
3. **Risk manager** scans the output for sensitive patterns (passwords,
   SSNs, etc.) and blocks the response if any are found.
4. **Finalizer** formats the final reply. Actions that need approval are
   only "executed" (logged, in paper mode) after the user confirms.

State is checkpointed per conversation thread via LangGraph's in-memory
saver, so a single run of `run_jarvis()` keeps context across turns.

## Setup

```bash
cd jarvis
pip install -r requirements.txt
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Run

```bash
python jarvis.py
```

Type a request (e.g. "schedule a call with Sam tomorrow at 3pm") and follow
the prompts. Type `exit` or `quit` to stop.

## Notes

- This assistant runs in **paper mode**: approved actions are only logged,
  not executed against real calendars/inboxes.
- Swap the model in `get_llm()` (e.g. `claude-3-haiku-20240307`) for a
  cheaper/faster alternative.
