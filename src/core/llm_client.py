from __future__ import annotations
import os
from typing import List, Dict
import httpx  # not required using langchain
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

PROVIDER = (os.getenv("PROVIDER") or "ollama").strip().lower()
MODEL = (os.getenv("MODEL") or "mistral:7b").strip()
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S") or "600")
# No temperature handling for Day-1: keep payloads simple and compatible.
LLM_TEMPERATURE = None

Message = Dict[str, str]

# below method(chat) is not rewuired for langchain usage

def chat(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
    """Send `messages` to the configured LLM provider and return assistant text.

    This thin, provider-agnostic helper keeps the interface simple for Day-1
    teaching: callers pass OpenAI-style `messages` and get back the assistant's
    `content` string. Validation is intentionally minimal to keep code readable.

    Args:
        messages: List of message dicts with `role` and `content`.
        timeout: Request timeout in seconds.

    Returns:
        str: Assistant text returned by the selected provider.

    Raises:
        ValueError: If `messages` is empty or not a list.
        RuntimeError: For provider-specific failures (missing keys, empty replies).
        NotImplementedError: If `PROVIDER` is not supported.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            "messages must be a non-empty list of {'role','content'} dicts."
        )

    if PROVIDER == "ollama":
        url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
        payload = {"model": MODEL, "messages": messages, "stream": False}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = (data.get("message") or {}).get("content")
            if not msg:
                raise RuntimeError(
                    "Ollama returned empty content. Check model and host."
                )
            return msg

    elif PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing but PROVIDER=openai.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {"model": MODEL, "messages": messages, "temperature": 0}
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("OpenAI returned no choices. Check model and key.")
            return (choices[0].get("message") or {}).get("content") or ""

    else:
        raise NotImplementedError("Unsupported PROVIDER. Use 'ollama' or 'openai'.")


# langchain method 
def _to_lc_messages(messages: List[Message]):
    """Convert [{'role','content'}] into LangChain BaseMessages."""
    lc_msgs = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            # treat 'user' and anything else as human input
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs

# langchain method for calling LLM
def _make_llm():
    """
    Create the LangChain chat model according to PROVIDER/MODEL envs.

    Note: We do NOT pass a `timeout` kwarg here for maximum compatibility
    across LangChain versions/backends (e.g., ChatOllama often has no such arg).
    """
    if PROVIDER == "ollama":
        # LangChain's Ollama wrapper reads OLLAMA_HOST from env.
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        return ChatOllama(model=MODEL)
    elif PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing but PROVIDER=openai.")
        # Keep temperature=0 for deterministic teaching runs
        return ChatOpenAI(model=MODEL, temperature=0)
    else:
        raise NotImplementedError("Unsupported PROVIDER. Use 'ollama' or 'openai'.")

def chat_lc(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
    """
    Send OpenAI-style messages and return assistant text (string).
    Keeps the exact caller contract your agents already use.

    `timeout` is currently advisory (not enforced for all backends uniformly).
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {'role','content'} dicts.")

    llm = _make_llm()
    lc_msgs = _to_lc_messages(messages)

    # Call the model; avoid per-call config to keep compatibility wide.
    resp = llm.invoke(lc_msgs)
    return getattr(resp, "content", "") or ""