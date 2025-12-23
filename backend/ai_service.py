"""AI service powered by OpenAI GPT models."""
import json
import logging
import os
from typing import Dict, List, Optional
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not _OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env file.")

_MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano-2025-08-07")
_client = OpenAI(api_key=_OPENAI_API_KEY)
_logger = logging.getLogger(__name__)


def _chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int = 500,
    response_format: Dict[str, str] | None = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    active_logger = logger or _logger
    try:
        response = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            max_completion_tokens=max_tokens,
            response_format=response_format,
        )
        # st.write(response)
    except OpenAIError as exc:  # pragma: no cover - depends on API availability
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

    active_logger.debug(
        "OpenAI response received (model=%s, format=%s, messages=%d): %s",
        _MODEL_NAME,
        response_format,
        len(messages),
        response,
    )

    if not response.choices:
        raise RuntimeError("OpenAI API returned an empty response.")

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI API returned an empty response.")

    return content.strip()


def generate_topic_explanation(topic: str) -> str:
    prompt = (
        f"Explain the topic '{topic}' in under 120 words using approachable language. "
        "Focus on clarity and practical intuition."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI tutor who explains topics clearly "
                "and concisely for beginners. Answer immediately with the final "
                "answer only—no step-by-step reasoning."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    response = _chat_completion(messages)

    return response


def _dialogue_messages(topic: str) -> List[Dict[str, str]]:
    instructions = (
        "You are orchestrating a duo teaching skit. Answer immediately with the final "
        "lines—no step-by-step reasoning. "
        "Produce EXACTLY three dialogue turns in JSON for CARTOON_DAD, JOHN, then CARTOON_DAD. "
        "Total word count across lines should stay near 120 words. "
        "CARTOON_DAD asks a funny question related to the topic and includes a specific example request. "
        "JOHN responds with calm, reassuring clarity, explains the topic, and ties it directly to the specific example CARTOON_DAD asked for. "
        "The final CARTOON_DAD line quickly summarizes the takeaway and thanks John. "
        "Respond ONLY with valid JSON shaped like "
        '{"dialogue":[{"speaker":"CARTOON_DAD","line":"..."}]}'
    )
    user_prompt = (
        f"Topic: {topic}. Keep the dialogue approachable and helpful while honoring the persona rules."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]


def _parse_dialogue_payload(payload: str) -> List[Dict[str, str]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload from OpenAI.") from exc

    dialogue = data.get("dialogue")
    if not isinstance(dialogue, list):
        raise ValueError("Dialogue payload missing 'dialogue' list.")

    expected_order = ["CARTOON_DAD", "JOHN", "CARTOON_DAD"]
    if len(dialogue) != len(expected_order):
        raise ValueError("Dialogue must contain exactly three turns.")

    parsed: List[Dict[str, str]] = []
    for idx, expected in enumerate(expected_order):
        entry = dialogue[idx]
        speaker = str(entry.get("speaker", "")).strip().upper()
        line = str(entry.get("line", "")).strip()
        if speaker != expected:
            raise ValueError(f"Dialogue turn {idx + 1} must be spoken by {expected}.")
        if not line:
            raise ValueError("Dialogue line text is required.")
        parsed.append({"speaker": expected, "line": line})

    return parsed


def generate_dialogue(topic: str, logger: Optional[logging.Logger] = None) -> List[Dict[str, str]]:
    """Generate a structured three-turn dialogue for Duo Mode."""
    active_logger = logger or _logger
    active_logger.info("Generating dialogue for topic: %s", topic)
    base_messages = _dialogue_messages(topic)
    response_format = {"type": "json_object"}
    last_error: ValueError | None = None

    for attempt in range(2):
        messages = list(base_messages)
        if attempt == 1:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Reminder: The reply MUST be valid JSON matching "
                        '{"dialogue":[{"speaker":"CARTOON_DAD","line":"..."},{"speaker":"JOHN","line":"..."},{"speaker":"CARTOON_DAD","line":"..."}]}. '
                        "Do not add commentary."
                    ),
                }
            )
        raw = _chat_completion(
            messages,
            max_tokens=10000,
            response_format=response_format,
            logger=active_logger,
        )
        try:
            parsed = _parse_dialogue_payload(raw)
            active_logger.info("Dialogue generation succeeded on attempt %s", attempt + 1)
            return parsed
        except ValueError as exc:
            active_logger.warning("Dialogue parse attempt %s failed: %s", attempt + 1, exc)
            last_error = exc
            continue

    raise RuntimeError(f"Failed to produce valid dialogue: {last_error}")
