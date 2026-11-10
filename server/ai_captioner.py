"""AI-powered Alt-Text Generator for Images and Videos.

Supports:
1. Ollama (Local vision models: LLaVA, Llama 3.2 Vision, Moondream, Qwen-VL, Gemma 3, etc.)
2. Google Gemini API (Native image & video understanding)
3. OpenAI & OpenAI-compatible APIs (GPT-4o, GPT-4o-mini, LM Studio, llama.cpp, vLLM)
4. Anthropic Claude API (Claude 3.7 / 3.5 Sonnet & Haiku vision)
"""

import base64
import json
import shutil
import ssl
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VISION_MODEL_KEYWORDS = (
    "llava",
    "vision",
    "moondream",
    "qwen",
    "vl",
    "minicpm",
    "bakllava",
    "pixtral",
    "gemma3",
    "paligemma",
    "internvl",
    "cogvlm",
)

MASTODON_ALT_TEXT_MAX_CHARS = 1500


class AICaptionerError(Exception):
    """Raised when an AI provider fails to generate a caption."""


def _create_ssl_context() -> ssl.SSLContext:
    """Create a default SSL context for HTTPS requests."""
    ctx = ssl.create_default_context()
    return ctx


def _strip_data_url(data_str: str) -> Tuple[str, str]:
    """Strip data URL header if present and return (mime_type, raw_base64)."""
    data_str = data_str.strip()
    if data_str.startswith("data:"):
        header, _, b64_part = data_str.partition(",")
        mime = "image/jpeg"
        if ";" in header:
            mime = header[5 : header.index(";")]
        elif len(header) > 5:
            mime = header[5:]
        return mime, b64_part.strip()
    return "image/jpeg", data_str


def build_accessibility_prompt(
    media_type: str = "image",
    detail_level: str = "descriptive",
    post_context: str = "",
    language: str = "English",
    custom_instructions: str = "",
    frame_count: int = 1,
) -> str:
    """Build a specialized system/user prompt for blind-accessible alt-text."""
    is_video = media_type == "video" or frame_count > 1

    detail_guidelines = {
        "concise": (
            "Write a concise, clear alt-text description in 1 to 2 sentences "
            "(roughly 120-250 characters). Focus on the main subject, action, and any critical text."
        ),
        "descriptive": (
            "Write a helpful, vivid alt-text description in 2 to 4 sentences "
            "(roughly 250-600 characters). Describe the main subject, setting, actions, "
            "expressions/mood, and transcribe any important visible text."
        ),
        "detailed": (
            "Write a comprehensive, richly detailed alt-text description in 4 to 7 sentences "
            "(up to 1,200 characters). Describe the composition, subjects, clothing/appearance, "
            "actions, background setting, lighting/colors, and verbatim transcribe all visible text."
        ),
    }
    level_instruction = detail_guidelines.get(
        detail_level, detail_guidelines["descriptive"]
    )

    if is_video:
        media_instruction = (
            "You are describing a VIDEO for blind and low-vision Mastodon users who use screen readers. "
            f"You are given {'extracted keyframes in chronological order from the video' if frame_count > 1 else 'the video'}. "
            "Describe what happens across the clip from beginning to end: the subjects, their actions and movement over time, "
            "the setting, camera movement or scene changes, and any on-screen text or captions. "
            "Synthesize the sequence into a single cohesive description of the video (do NOT list 'Frame 1, Frame 2' separately)."
        )
    else:
        media_instruction = (
            "You are describing an IMAGE for blind and low-vision Mastodon users who use screen readers."
        )

    prompt_parts = [
        media_instruction,
        "",
        "Accessibility Rules (W3C WAI & Perkins School for the Blind Guidelines):",
        "1. Do NOT begin with 'Image of', 'Photo of', 'Picture of', 'Video of', or 'This shows'. Start immediately with the subject or scene.",
        "2. However, DO mention the medium at the end or naturally if relevant (e.g., 'Screenshot of a terminal window...', 'Black-and-white illustration of...', 'Animated loop of...').",
        "3. Transcribe any visible text, signs, labels, code, or meme captions verbatim inside quotation marks.",
        "4. Be objective, accurate, and natural to listen to via a screen reader.",
        "5. Output ONLY the final alt-text caption itself — no introductory commentary, no markdown headers, and no surrounding quotes.",
        "6. Keep the total length under 1,450 characters (Mastodon's 1,500-character limit).",
        "",
        f"Detail Level: {level_instruction}",
        f"Output Language: {language or 'English'}",
    ]

    if post_context and post_context.strip():
        prompt_parts.append(
            f"\nAuthor's accompanying post text for context (avoid repeating it verbatim, use it to understand relevance/names): \"{post_context.strip()}\""
        )

    if custom_instructions and custom_instructions.strip():
        prompt_parts.append(
            f"\nAdditional user instructions: {custom_instructions.strip()}"
        )

    return "\n".join(prompt_parts)


def extract_frames_with_ffmpeg(
    video_bytes: bytes, suffix: str = ".mp4", max_frames: int = 4
) -> List[str]:
    """Optional server-side fallback to extract base64 JPEG keyframes using ffmpeg if installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return []

    frames_b64: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_file = tmp_path / f"input{suffix}"
        video_file.write_bytes(video_bytes)

        # Extract up to max_frames evenly spaced or at 1 fps
        out_pattern = str(tmp_path / "frame_%02d.jpg")
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_file),
            "-vf",
            "fps=1,scale='min(1024,iw)':-2",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "4",
            out_pattern,
        ]
        try:
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=False,
            )
            for frame_file in sorted(tmp_path.glob("frame_*.jpg"))[:max_frames]:
                b64 = base64.b64encode(frame_file.read_bytes()).decode("ascii")
                frames_b64.append(b64)
        except Exception:
            return []

    return frames_b64


AUTO_DESCRIPTION_PREFIX = "AI generated description: "


def clean_caption_output(text: str) -> str:
    """Clean model output and prefix with 'AI generated description: '."""
    cleaned = text.strip()

    # Strip surrounding quotes if the model wrapped the entire response in quotes
    if (
        len(cleaned) >= 2
        and cleaned[0] in ('"', "“")
        and cleaned[-1] in ('"', "”")
        and cleaned.count('"') == 2
    ):
        cleaned = cleaned[1:-1].strip()

    # Strip common unwanted prefixes
    prefixes_to_strip = [
        "AI generated description:",
        "AI-generated description:",
        "Automatically generated description:",
        "Alt-text:",
        "Alt text:",
        "Caption:",
        "Description:",
        "Here is the alt-text:",
        "Here's the alt-text:",
    ]
    lower = cleaned.lower()
    for prefix in prefixes_to_strip:
        if lower.startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break

    if not cleaned:
        return ""

    cleaned = f"{AUTO_DESCRIPTION_PREFIX}{cleaned}"

    if len(cleaned) > MASTODON_ALT_TEXT_MAX_CHARS:
        cleaned = cleaned[: MASTODON_ALT_TEXT_MAX_CHARS - 1].rsplit(" ", 1)[0] + "…"

    return cleaned


def list_ollama_models(base_url: str = "http://localhost:11434") -> Dict[str, Any]:
    """Query local Ollama server for available models."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models_raw = data.get("models", [])
            models = []
            for m in models_raw:
                name = m.get("name", "")
                families = (
                    m.get("details", {}).get("families") or []
                )
                is_vision = any(
                    kw in name.lower() for kw in VISION_MODEL_KEYWORDS
                ) or "clip" in [f.lower() for f in families] or "mllama" in [f.lower() for f in families]
                models.append(
                    {
                        "name": name,
                        "size": m.get("size", 0),
                        "family": m.get("details", {}).get("family", ""),
                        "parameter_size": m.get("details", {}).get("parameter_size", ""),
                        "is_vision_likely": is_vision,
                    }
                )
            # Sort vision-likely models first
            models.sort(key=lambda x: (not x["is_vision_likely"], x["name"]))
            return {
                "reachable": True,
                "provider": "ollama",
                "models": models,
                "base_url": base_url,
            }
    except Exception as exc:
        return {
            "reachable": False,
            "provider": "ollama",
            "models": [],
            "base_url": base_url,
            "error": str(exc),
        }


DEFAULT_GEMINI_MODELS = [
    {"name": "gemini-2.5-flash", "display_name": "Gemini 2.5 Flash", "is_vision_likely": True},
    {"name": "gemini-2.5-pro", "display_name": "Gemini 2.5 Pro", "is_vision_likely": True},
    {"name": "gemini-2.0-flash", "display_name": "Gemini 2.0 Flash", "is_vision_likely": True},
    {"name": "gemini-1.5-pro", "display_name": "Gemini 1.5 Pro", "is_vision_likely": True},
    {"name": "gemini-1.5-flash", "display_name": "Gemini 1.5 Flash", "is_vision_likely": True},
]

DEFAULT_OPENAI_MODELS = [
    {"name": "gpt-4o-mini", "display_name": "gpt-4o-mini", "is_vision_likely": True},
    {"name": "gpt-4o", "display_name": "gpt-4o", "is_vision_likely": True},
    {"name": "gpt-4.1-mini", "display_name": "gpt-4.1-mini", "is_vision_likely": True},
    {"name": "gpt-4.1", "display_name": "gpt-4.1", "is_vision_likely": True},
    {"name": "gpt-4-turbo", "display_name": "gpt-4-turbo", "is_vision_likely": True},
]

DEFAULT_ANTHROPIC_MODELS = [
    {"name": "claude-3-7-sonnet-latest", "display_name": "Claude 3.7 Sonnet (latest)", "is_vision_likely": True},
    {"name": "claude-3-5-sonnet-latest", "display_name": "Claude 3.5 Sonnet (latest)", "is_vision_likely": True},
    {"name": "claude-3-5-haiku-latest", "display_name": "Claude 3.5 Haiku (latest)", "is_vision_likely": True},
    {"name": "claude-3-opus-latest", "display_name": "Claude 3 Opus (latest)", "is_vision_likely": True},
]


def list_gemini_models(api_key: str) -> Dict[str, Any]:
    """Query Google Gemini API for available generative models."""
    api_key = (api_key or "").strip()
    if not api_key:
        return {
            "reachable": False,
            "provider": "gemini",
            "models": list(DEFAULT_GEMINI_MODELS),
            "error": "No Gemini API key configured yet; showing default models.",
        }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={urllib.parse.quote(api_key)}"
    )
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(
            req, timeout=8, context=_create_ssl_context()
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_models = data.get("models", [])
            models = []
            excluded_substrings = (
                "embedding",
                "aqa",
                "text-bison",
                "chat-bison",
                "imagen",
                "veo",
                "tts",
            )
            for m in raw_models:
                raw_name = m.get("name", "")
                name = (
                    raw_name[len("models/") :]
                    if raw_name.startswith("models/")
                    else raw_name
                )
                if not name:
                    continue
                lower_name = name.lower()
                if any(ex in lower_name for ex in excluded_substrings):
                    continue
                methods = m.get("supportedGenerationMethods") or []
                if methods and "generateContent" not in methods:
                    continue
                is_vision = any(
                    prefix in lower_name
                    for prefix in ("gemini-2.5", "gemini-2.0", "gemini-1.5", "gemini-", "gemma-3")
                )
                models.append(
                    {
                        "name": name,
                        "display_name": m.get("displayName") or name,
                        "is_vision_likely": is_vision,
                    }
                )
            models.sort(
                key=lambda x: (
                    not x["is_vision_likely"],
                    not x["name"].startswith("gemini-2.5"),
                    not x["name"].startswith("gemini-2.0"),
                    x["name"],
                )
            )
            return {
                "reachable": True,
                "provider": "gemini",
                "models": models or list(DEFAULT_GEMINI_MODELS),
            }
    except Exception as exc:
        return {
            "reachable": False,
            "provider": "gemini",
            "models": list(DEFAULT_GEMINI_MODELS),
            "error": str(exc),
        }


def list_openai_models(
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
) -> Dict[str, Any]:
    """Query OpenAI or an OpenAI-compatible server for available models."""
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    api_key = (api_key or "").strip()
    is_local = any(
        host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", ".local")
    )
    if not api_key and not is_local:
        return {
            "reachable": False,
            "provider": "openai",
            "base_url": base_url,
            "models": list(DEFAULT_OPENAI_MODELS),
            "error": "No OpenAI API key configured yet; showing default models.",
        }

    url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(
            req, timeout=8, context=_create_ssl_context()
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_list = data.get("data") if isinstance(data, dict) else None
            if raw_list is None and isinstance(data, dict):
                raw_list = data.get("models", [])
            if not isinstance(raw_list, list):
                raw_list = []

            is_official_openai = "api.openai.com" in base_url
            excluded_openai_prefixes = (
                "whisper",
                "tts",
                "dall-e",
                "text-embedding",
                "embedding",
                "babbage",
                "davinci",
                "omni-moderation",
                "text-moderation",
                "gpt-3.5",
                "canary",
            )
            excluded_openai_substrings = (
                "-audio-",
                "-realtime-",
                "-transcribe",
                "-tts",
                "-search-",
            )
            vision_keywords = VISION_MODEL_KEYWORDS + (
                "gpt-4o",
                "gpt-4.1",
                "gpt-4.5",
                "gpt-4-turbo",
                "chatgpt-4o",
                "o1",
                "o3",
                "o4",
                "claude",
                "gemini",
            )

            models = []
            seen = set()
            for m in raw_list:
                if isinstance(m, str):
                    name = m
                elif isinstance(m, dict):
                    name = m.get("id") or m.get("name") or ""
                else:
                    continue
                name = name.strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                lower_name = name.lower()
                if is_official_openai:
                    if lower_name.startswith(excluded_openai_prefixes):
                        continue
                    if any(sub in lower_name for sub in excluded_openai_substrings):
                        continue

                is_vision = any(kw in lower_name for kw in vision_keywords)
                models.append(
                    {
                        "name": name,
                        "display_name": name,
                        "is_vision_likely": is_vision,
                    }
                )

            models.sort(key=lambda x: (not x["is_vision_likely"], x["name"]))
            return {
                "reachable": True,
                "provider": "openai",
                "base_url": base_url,
                "models": models or (list(DEFAULT_OPENAI_MODELS) if is_official_openai else []),
            }
    except Exception as exc:
        return {
            "reachable": False,
            "provider": "openai",
            "base_url": base_url,
            "models": list(DEFAULT_OPENAI_MODELS),
            "error": str(exc),
        }


def list_anthropic_models(api_key: str) -> Dict[str, Any]:
    """Query Anthropic API for available Claude models."""
    api_key = (api_key or "").strip()
    if not api_key:
        return {
            "reachable": False,
            "provider": "anthropic",
            "models": list(DEFAULT_ANTHROPIC_MODELS),
            "error": "No Anthropic API key configured yet; showing default models.",
        }

    url = "https://api.anthropic.com/v1/models"
    headers = {
        "Accept": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(
            req, timeout=8, context=_create_ssl_context()
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            raw_list = data.get("data", [])
            models = []
            seen = set()
            for m in raw_list:
                name = (m.get("id") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                display_name = m.get("display_name") or name
                is_vision = "claude-3" in name.lower() or "claude-4" in name.lower() or "sonnet" in name.lower() or "opus" in name.lower() or "haiku" in name.lower()
                models.append(
                    {
                        "name": name,
                        "display_name": display_name,
                        "is_vision_likely": is_vision,
                    }
                )
            # Also include convenient "-latest" aliases if not returned by /v1/models
            for alias in DEFAULT_ANTHROPIC_MODELS:
                if alias["name"] not in seen:
                    models.append(dict(alias))
            return {
                "reachable": True,
                "provider": "anthropic",
                "models": models or list(DEFAULT_ANTHROPIC_MODELS),
            }
    except Exception as exc:
        return {
            "reachable": False,
            "provider": "anthropic",
            "models": list(DEFAULT_ANTHROPIC_MODELS),
            "error": str(exc),
        }


def list_provider_models(
    provider: str,
    ai_config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Query the specified AI provider for its list of available models."""
    provider = (provider or "ollama").strip().lower()
    ai_cfg = ai_config or {}
    ovr = overrides or {}

    if provider == "ollama":
        base_url = (
            ovr.get("base_url")
            or ai_cfg.get("ollama", {}).get("base_url")
            or "http://localhost:11434"
        )
        return list_ollama_models(base_url=base_url)

    if provider == "gemini":
        api_key = (
            ovr.get("api_key")
            if "api_key" in ovr and ovr.get("api_key")
            else ai_cfg.get("gemini", {}).get("api_key", "")
        )
        return list_gemini_models(api_key=api_key)

    if provider == "openai":
        base_url = (
            ovr.get("base_url")
            or ai_cfg.get("openai", {}).get("base_url")
            or "https://api.openai.com/v1"
        )
        api_key = (
            ovr.get("api_key")
            if "api_key" in ovr and ovr.get("api_key")
            else ai_cfg.get("openai", {}).get("api_key", "")
        )
        return list_openai_models(base_url=base_url, api_key=api_key)

    if provider == "anthropic":
        api_key = (
            ovr.get("api_key")
            if "api_key" in ovr and ovr.get("api_key")
            else ai_cfg.get("anthropic", {}).get("api_key", "")
        )
        return list_anthropic_models(api_key=api_key)

    return {
        "reachable": False,
        "provider": provider,
        "models": [],
        "error": f"Unknown AI provider: {provider}",
    }


def generate_with_ollama(
    images_b64: List[str],
    prompt: str,
    base_url: str = "http://localhost:11434",
    model: str = "llava:latest",
) -> str:
    """Generate caption using a local or remote Ollama server."""
    if not images_b64:
        raise AICaptionerError(
            "No image or video keyframes were provided for Ollama vision analysis."
        )

    base_url = base_url.rstrip("/")
    model = (model or "llava:latest").strip()
    raw_images = [_strip_data_url(img)[1] for img in images_b64]

    chat_url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": raw_images,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
        },
    }

    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "")
            if not content:
                raise AICaptionerError("Ollama returned an empty caption.")
            return clean_caption_output(content)
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(err_body)
            err_msg = err_json.get("error", err_body)
        except Exception:
            err_msg = err_body
        raise AICaptionerError(
            f"Ollama error ({http_err.code}) with model '{model}': {err_msg}. "
            f"Make sure the vision model is pulled (e.g. `ollama pull {model}`)."
        ) from http_err
    except urllib.error.URLError as url_err:
        raise AICaptionerError(
            f"Could not connect to Ollama at {base_url} ({url_err.reason}). "
            "Ensure Ollama is running (`ollama serve` or open the Ollama app)."
        ) from url_err


def generate_with_gemini(
    images_b64: List[str],
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    raw_media_b64: Optional[str] = None,
    raw_media_mime: Optional[str] = None,
) -> str:
    """Generate caption using Google Gemini API (supports both native video and image frames)."""
    api_key = (api_key or "").strip()
    if not api_key:
        raise AICaptionerError(
            "Google Gemini API key is not configured. Add your key in AI Settings."
        )

    model = (model or "gemini-2.5-flash").strip()
    parts: List[Dict[str, Any]] = []

    # If native video bytes were passed and are reasonably sized (<18MB base64), include video inline
    if (
        raw_media_b64
        and raw_media_mime
        and raw_media_mime.startswith("video/")
        and len(raw_media_b64) < 24_000_000
    ):
        _, clean_b64 = _strip_data_url(raw_media_b64)
        parts.append(
            {
                "inline_data": {
                    "mime_type": raw_media_mime,
                    "data": clean_b64,
                }
            }
        )
    elif images_b64:
        for img_str in images_b64:
            mime, clean_b64 = _strip_data_url(img_str)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": mime or "image/jpeg",
                        "data": clean_b64,
                    }
                }
            )
    else:
        raise AICaptionerError("No media data provided for Gemini caption generation.")

    parts.append({"text": prompt})

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}"
    )
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 600,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=90, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if not candidates:
                raise AICaptionerError("Gemini API returned no candidates.")
            content_parts = (
                candidates[0].get("content", {}).get("parts", [])
            )
            text = "".join(p.get("text", "") for p in content_parts).strip()
            if not text:
                raise AICaptionerError("Gemini API returned an empty caption.")
            return clean_caption_output(text)
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_msg = json.loads(err_body).get("error", {}).get("message", err_body)
        except Exception:
            err_msg = err_body
        raise AICaptionerError(
            f"Google Gemini API error ({http_err.code}): {err_msg}"
        ) from http_err
    except urllib.error.URLError as url_err:
        raise AICaptionerError(
            f"Network error contacting Google Gemini API: {url_err.reason}"
        ) from url_err


def generate_with_openai(
    images_b64: List[str],
    prompt: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> str:
    """Generate caption using OpenAI or an OpenAI-compatible vision endpoint."""
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    api_key = (api_key or "").strip()
    is_local = any(
        host in base_url for host in ("localhost", "127.0.0.1", "0.0.0.0", ".local")
    )
    if not api_key and not is_local:
        raise AICaptionerError(
            "OpenAI API key is not configured. Add your API key in AI Settings."
        )

    if not images_b64:
        raise AICaptionerError(
            "No image or video keyframes were provided for OpenAI vision analysis."
        )

    content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_str in images_b64:
        mime, clean_b64 = _strip_data_url(img_str)
        data_url = f"data:{mime};base64,{clean_b64}"
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "auto"},
            }
        )

    url = f"{base_url}/chat/completions"
    payload = {
        "model": (model or "gpt-4o-mini").strip(),
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=90, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            if not choices:
                raise AICaptionerError("OpenAI endpoint returned no choices.")
            text = choices[0].get("message", {}).get("content", "")
            if not text:
                raise AICaptionerError("OpenAI endpoint returned an empty caption.")
            return clean_caption_output(text)
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_msg = json.loads(err_body).get("error", {}).get("message", err_body)
        except Exception:
            err_msg = err_body
        raise AICaptionerError(
            f"OpenAI API error ({http_err.code}): {err_msg}"
        ) from http_err
    except urllib.error.URLError as url_err:
        raise AICaptionerError(
            f"Could not connect to OpenAI endpoint at {base_url}: {url_err.reason}"
        ) from url_err


def generate_with_anthropic(
    images_b64: List[str],
    prompt: str,
    api_key: str,
    model: str = "claude-3-7-sonnet-latest",
) -> str:
    """Generate caption using Anthropic Claude Vision API."""
    api_key = (api_key or "").strip()
    if not api_key:
        raise AICaptionerError(
            "Anthropic API key is not configured. Add your API key in AI Settings."
        )

    if not images_b64:
        raise AICaptionerError(
            "No image or video keyframes were provided for Anthropic Claude vision analysis."
        )

    content_blocks: List[Dict[str, Any]] = []
    for img_str in images_b64:
        mime, clean_b64 = _strip_data_url(img_str)
        if mime not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            mime = "image/jpeg"
        content_blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": clean_b64,
                },
            }
        )
    content_blocks.append({"type": "text", "text": prompt})

    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": (model or "claude-3-7-sonnet-latest").strip(),
        "max_tokens": 600,
        "temperature": 0.3,
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=90, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            blocks = result.get("content", [])
            text = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            ).strip()
            if not text:
                raise AICaptionerError("Anthropic API returned an empty caption.")
            return clean_caption_output(text)
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="replace")
        try:
            err_msg = json.loads(err_body).get("error", {}).get("message", err_body)
        except Exception:
            err_msg = err_body
        raise AICaptionerError(
            f"Anthropic API error ({http_err.code}): {err_msg}"
        ) from http_err
    except urllib.error.URLError as url_err:
        raise AICaptionerError(
            f"Network error contacting Anthropic API: {url_err.reason}"
        ) from url_err


def generate_alt_text(
    ai_config: Dict[str, Any],
    media_type: str = "image",
    images_b64: Optional[List[str]] = None,
    raw_media_b64: Optional[str] = None,
    raw_media_mime: Optional[str] = None,
    post_context: str = "",
    detail_level: Optional[str] = None,
    custom_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified entry point to generate accessible alt-text for an image or video."""
    images_b64 = list(images_b64 or [])
    provider = (ai_config.get("provider") or "ollama").lower()
    effective_detail = detail_level or ai_config.get("detail_level", "descriptive")
    language = ai_config.get("language", "English")
    effective_custom = (
        custom_instructions
        if custom_instructions is not None
        else ai_config.get("custom_instructions", "")
    )

    # If single image was sent via raw_media_b64 and images_b64 is empty, use it
    if not images_b64 and raw_media_b64:
        if (raw_media_mime or "").startswith("image/"):
            images_b64 = [raw_media_b64]
        elif (raw_media_mime or "").startswith("video/"):
            # Try server-side ffmpeg extraction if client didn't send keyframes
            _, clean_b64 = _strip_data_url(raw_media_b64)
            video_bytes = base64.b64decode(clean_b64)
            images_b64 = extract_frames_with_ffmpeg(video_bytes)

    prompt = build_accessibility_prompt(
        media_type=media_type,
        detail_level=effective_detail,
        post_context=post_context if ai_config.get("include_post_context", True) else "",
        language=language,
        custom_instructions=effective_custom,
        frame_count=max(len(images_b64), 1),
    )

    if provider == "ollama":
        ollama_cfg = ai_config.get("ollama", {})
        model_used = ollama_cfg.get("model", "llava:latest")
        caption = generate_with_ollama(
            images_b64=images_b64,
            prompt=prompt,
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            model=model_used,
        )
    elif provider == "gemini":
        gemini_cfg = ai_config.get("gemini", {})
        model_used = gemini_cfg.get("model", "gemini-2.5-flash")
        caption = generate_with_gemini(
            images_b64=images_b64,
            prompt=prompt,
            api_key=gemini_cfg.get("api_key", ""),
            model=model_used,
            raw_media_b64=raw_media_b64,
            raw_media_mime=raw_media_mime,
        )
    elif provider == "openai":
        openai_cfg = ai_config.get("openai", {})
        model_used = openai_cfg.get("model", "gpt-4o-mini")
        caption = generate_with_openai(
            images_b64=images_b64,
            prompt=prompt,
            api_key=openai_cfg.get("api_key", ""),
            base_url=openai_cfg.get("base_url", "https://api.openai.com/v1"),
            model=model_used,
        )
    elif provider == "anthropic":
        anthropic_cfg = ai_config.get("anthropic", {})
        model_used = anthropic_cfg.get("model", "claude-3-7-sonnet-latest")
        caption = generate_with_anthropic(
            images_b64=images_b64,
            prompt=prompt,
            api_key=anthropic_cfg.get("api_key", ""),
            model=model_used,
        )
    else:
        raise AICaptionerError(f"Unsupported AI provider: {provider}")

    return {
        "caption": caption,
        "provider": provider,
        "model": model_used,
        "media_type": media_type,
        "frames_analyzed": len(images_b64),
    }
