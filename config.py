import os
from dataclasses import dataclass

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is listed in requirements.txt
    find_dotenv = None
    load_dotenv = None


DEFAULT_LLM_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_LLM_MODEL = "deepseek-chat"

PLACEHOLDER_API_KEYS = {
    "",
    "your-api-key-here",
    "your-deepseek-api-key",
    "your-dashscope-key-here",
}


def _should_load_dotenv():
    value = os.environ.get("KGQA_LOAD_DOTENV", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _load_local_dotenv():
    if not load_dotenv or not _should_load_dotenv():
        return

    if find_dotenv:
        dotenv_path = find_dotenv(usecwd=True)
        if dotenv_path:
            load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    load_dotenv(override=False)


_load_local_dotenv()


def _clean(value):
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    url: str
    model: str


def get_llm_config(api_key=None, api_url=None, model=None):
    _load_local_dotenv()

    return LLMConfig(
        api_key=(
            _clean(api_key)
            if api_key is not None
            else _clean(os.environ.get("LLM_API_KEY"))
        ),
        url=(
            _clean(api_url)
            if api_url is not None
            else _clean(os.environ.get("LLM_URL"))
        ) or DEFAULT_LLM_URL,
        model=(
            _clean(model)
            if model is not None
            else _clean(os.environ.get("LLM_MODEL"))
        ) or DEFAULT_LLM_MODEL,
    )


def has_llm_api_key(api_key):
    return _clean(api_key).lower() not in PLACEHOLDER_API_KEYS


def get_llm_status():
    llm_config = get_llm_config()
    return {
        "configured": has_llm_api_key(llm_config.api_key),
        "url": llm_config.url,
        "model": llm_config.model,
    }
