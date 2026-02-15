"""
Unified LLM client - supports Gemini, Anthropic (Claude), and OpenAI.
Updated to support temperature parameter for deterministic behavior.
"""

import os
from typing import List, Dict, Any


def _get_provider() -> str:
    """Return 'gemini', 'anthropic', or 'openai'. LLM_PROVIDER overrides if set."""
    forced = os.getenv("LLM_PROVIDER", "").lower()
    if forced in ("openai", "gemini", "anthropic"):
        key = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[forced]
        if os.getenv(key):
            return forced
        raise ValueError(f"LLM_PROVIDER={forced} but {key} is not set in .env")
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise ValueError(
        "Set one of GEMINI_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env. "
        "Get keys at: Gemini https://aistudio.google.com/apikey | "
        "Anthropic https://console.anthropic.com | "
        "OpenAI https://platform.openai.com"
    )


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,  # Added temperature parameter
) -> str:
    """
    Send chat messages to LLM and return the assistant's reply.
    messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
    temperature: 0.0 for deterministic, higher for creative (default 0.0 for consistency)
    """
    provider = _get_provider()

    if provider == "gemini":
        return _gemini_chat(messages, model or "gemini-2.0-flash", max_tokens, temperature)
    elif provider == "anthropic":
        return _anthropic_chat(messages, model or "claude-3-5-haiku-20241022", max_tokens, temperature)
    else:
        return _openai_chat(messages, model or "gpt-4o-mini", max_tokens, temperature)


def _anthropic_chat(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Anthropic: system is separate, messages are user/assistant only
    system = ""
    chat_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system = msg["content"]
        else:
            chat_messages.append({"role": msg["role"], "content": msg["content"]})

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or None,
        messages=chat_messages,
    )
    return response.content[0].text


def _openai_chat(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def _gemini_chat(messages: List[Dict[str, str]], model: str, max_tokens: int, temperature: float) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model_map = {
        "gpt-4o": "gemini-2.5-pro",
        "gpt-4o-mini": "gemini-2.0-flash",
        "gpt-4": "gemini-2.5-pro",
        "gemini-1.5-flash": "gemini-2.0-flash",
        "gemini-1.5-pro": "gemini-2.5-pro",
    }
    gemini_model = model_map.get(model, model)
    gen_model = genai.GenerativeModel(
        gemini_model,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    # Build Gemini chat history from (user, assistant) pairs
    history = []
    i = 0
    system_content = ""
    last_assistant = ""
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "system":
            system_content = msg["content"]
            i += 1
            continue
        if msg["role"] == "user":
            content = (system_content + "\n\n" + msg["content"]) if system_content else msg["content"]
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                history.append({"role": "user", "parts": [content]})
                history.append({"role": "model", "parts": [messages[i + 1]["content"]]})
                last_assistant = messages[i + 1]["content"]
                i += 2
                system_content = ""
            else:
                chat = gen_model.start_chat(history=history)
                response = chat.send_message(content)
                return response.text if response.text else ""
        else:
            i += 1
    return last_assistant


# Convenience for scripts that need model names
def get_default_models() -> tuple:
    """Return (main_model, recursive_model) for current provider."""
    provider = _get_provider()
    if provider == "gemini":
        return ("gemini-2.5-pro", "gemini-2.0-flash")
    if provider == "anthropic":
        return ("claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022")
    return ("gpt-4o", "gpt-4o-mini")