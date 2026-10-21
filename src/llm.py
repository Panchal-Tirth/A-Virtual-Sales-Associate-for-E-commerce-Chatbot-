import json, time
from groq import Groq
from config import GROQ_API_KEY, LLM_FAST, LLM_SMART

_client = Groq(api_key=GROQ_API_KEY)

def _call(**kw):
    kw.setdefault("reasoning_effort", "low")
    for attempt in range(4):
        try:
            return _client.chat.completions.create(**kw)
        except Exception as e:
            msg = str(e).lower()
            if "reasoning_effort" in msg and "reasoning_effort" in kw:   # model doesn't support it
                kw.pop("reasoning_effort"); continue
            retryable = any(x in msg for x in ("rate", "429", "timeout", "503", "overloaded"))
            if attempt == 3 or not retryable:
                raise
            time.sleep(2 ** attempt)

def chat_json(system, messages, model=LLM_FAST):
    r = _call(model=model, temperature=0, max_completion_tokens=2000,
              response_format={"type": "json_object"},
              messages=[{"role": "system", "content": system}] + messages)
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except Exception:
        return {}

def chat_text(system, messages, model=LLM_SMART, temperature=0.4):
    r = _call(model=model, temperature=temperature, max_completion_tokens=2500,
              messages=[{"role": "system", "content": system}] + messages)
    return (r.choices[0].message.content or "").strip()