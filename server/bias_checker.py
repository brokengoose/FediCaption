"""Pre-Post Racial & Gender Bias Analyzer for FediCaption.

Inspects all components of a Mastodon draft before publishing:
1. Main post text and Content Warning / Spoiler text
2. Attached images and videos (both visual frames and alt-text descriptions)
3. Linked web pages (fetching title, meta description, and text excerpt)
4. Referenced or replied-to Mastodon statuses

Uses the configured AI provider (Ollama, Google Gemini, OpenAI / compatible,
or Anthropic Claude) and falls back to a built-in deterministic heuristic
analyzer if the AI provider is offline or unconfigured.
"""

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from server.ai_captioner import (
    AICaptionerError,
    _create_ssl_context,
    _strip_data_url,
)

# Regex to extract HTTP/HTTPS links from draft post text
_URL_REGEX = re.compile(r'https?://[^\s<>"\')\]]+', re.IGNORECASE)

# Deterministic fallback rules for common gender-exclusive and racially loaded terms.
# Tuple structure: (compiled_regex, bias_category, explanation, inclusive_replacement)
_HEURISTIC_BIAS_PATTERNS: List[Tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"\bmankind\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses male-default term 'mankind' to refer to all people.",
        "humanity",
    ),
    (
        re.compile(r"\bmanpower\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses male-default term 'manpower' for workforce or staffing.",
        "workforce",
    ),
    (
        re.compile(r"\bman-hours\b|\bman hours\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses male-default term 'man-hours' for labor time.",
        "person-hours",
    ),
    (
        re.compile(r"\bmanmade\b|\bman-made\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses male-default term 'man-made' for artificial or synthetic items.",
        "human-made",
    ),
    (
        re.compile(r"\bchairman\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'chairman'.",
        "chairperson",
    ),
    (
        re.compile(r"\bpoliceman\b|\bpolicemen\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'policeman/policemen'.",
        "police officer",
    ),
    (
        re.compile(r"\bfireman\b|\bfiremen\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'fireman/firemen'.",
        "firefighter",
    ),
    (
        re.compile(r"\bmailman\b|\bmailmen\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'mailman'.",
        "mail carrier",
    ),
    (
        re.compile(r"\bsalesman\b|\bsalesmen\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'salesman'.",
        "salesperson",
    ),
    (
        re.compile(r"\bspokesman\b|\bspokesmen\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-exclusive title 'spokesman'.",
        "spokesperson",
    ),
    (
        re.compile(r"\bhey guys\b|\byou guys\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Addresses a general or mixed audience as 'guys', which defaults to male.",
        "everyone",
    ),
    (
        re.compile(r"\bhusbands and wives\b|\bwives and husbands\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-binary phrasing ('husbands and wives') when referring to spouses or partners.",
        "spouses and partners",
    ),
    (
        re.compile(r"\bboyfriend or girlfriend\b|\bgirlfriend or boyfriend\b", re.IGNORECASE),
        "Implicit Gender Bias",
        "Uses gender-binary phrasing ('boyfriend or girlfriend') instead of inclusive partner language.",
        "partner",
    ),
    (
        re.compile(r"\bwhitelist\b", re.IGNORECASE),
        "Implicit Racial Bias",
        "Uses 'whitelist' (equating 'white' with approved/good and 'black' with blocked/bad).",
        "allowlist",
    ),
    (
        re.compile(r"\bblacklist\b", re.IGNORECASE),
        "Implicit Racial Bias",
        "Uses 'blacklist' (equating 'black' with blocked/bad).",
        "blocklist",
    ),
    (
        re.compile(r"\bmaster[\s/-]+slave\b", re.IGNORECASE),
        "Implicit Racial Bias",
        "Uses 'master/slave' metaphor rooted in racial chattel slavery.",
        "primary/replica",
    ),
    (
        re.compile(r"\bgrandfathered\b|\bgrandfather clause\b", re.IGNORECASE),
        "Implicit Racial Bias",
        "Uses 'grandfathered', a term originating from Jim Crow voter disenfranchisement clauses.",
        "legacy-exempted",
    ),
    (
        re.compile(r"\buppity\b", re.IGNORECASE),
        "Implicit Racial Bias",
        "Uses 'uppity', a racially loaded term historically used against Black people.",
        "arrogant",
    ),
]


def extract_urls_from_text(text: str) -> List[str]:
    """Extract unique HTTP/HTTPS URLs from post text, stripping trailing punctuation."""
    if not text:
        return []
    found = _URL_REGEX.findall(text)
    cleaned: List[str] = []
    seen = set()
    for raw_url in found:
        url = raw_url.rstrip(".,;:!?)")
        if url and url not in seen:
            seen.add(url)
            cleaned.append(url)
    return cleaned


def fetch_link_preview_text(url: str, timeout: int = 5) -> Dict[str, str]:
    """Fetch a linked webpage and extract its title, meta description, and a short text excerpt."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"url": url, "title": "", "excerpt": "", "reachable": "false"}

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "FediCaption/1.0 (+https://github.com/brokengoose/FediCaption)",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.5",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            req, timeout=timeout, context=_create_ssl_context()
        ) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw_bytes = resp.read(131072)  # Read at most 128 KB
            if "application/json" in ctype:
                data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                if isinstance(data, dict):
                    title = str(data.get("title") or data.get("url") or "")
                    content = str(
                        data.get("content")
                        or data.get("description")
                        or data.get("text")
                        or ""
                    )
                    plain = re.sub(r"<[^>]+>", " ", content)
                    plain = html.unescape(re.sub(r"\s+", " ", plain)).strip()
                    return {
                        "url": url,
                        "title": title[:200],
                        "excerpt": plain[:600],
                        "reachable": "true",
                    }

            if "text/" not in ctype and "html" not in ctype and "xml" not in ctype:
                return {
                    "url": url,
                    "title": f"Media / binary link ({ctype.split(';')[0]})",
                    "excerpt": "",
                    "reachable": "true",
                }

            html_str = raw_bytes.decode("utf-8", errors="replace")
            title_match = re.search(
                r"<title[^>]*>(.*?)</title>", html_str, re.IGNORECASE | re.DOTALL
            )
            title = (
                html.unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
                if title_match
                else ""
            )

            meta_desc = ""
            meta_match = re.search(
                r'<meta[^>]+(?:name|property)=["\'](?:description|og:description|twitter:description)["\'][^>]+content=["\']([^"\']+)["\']',
                html_str,
                re.IGNORECASE,
            )
            if not meta_match:
                meta_match = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'](?:description|og:description|twitter:description)["\']',
                    html_str,
                    re.IGNORECASE,
                )
            if meta_match:
                meta_desc = html.unescape(meta_match.group(1)).strip()

            body_clean = re.sub(
                r"<(script|style|noscript|svg|head)[^>]*>.*?</\1>",
                " ",
                html_str,
                flags=re.IGNORECASE | re.DOTALL,
            )
            body_clean = re.sub(r"<[^>]+>", " ", body_clean)
            body_clean = html.unescape(re.sub(r"\s+", " ", body_clean)).strip()
            combined_excerpt = (
                f"{meta_desc} — {body_clean}".strip(" —")
                if meta_desc
                else body_clean
            )
            return {
                "url": url,
                "title": title[:200],
                "excerpt": combined_excerpt[:600],
                "reachable": "true",
            }
    except Exception:
        return {"url": url, "title": "", "excerpt": "", "reachable": "false"}


def _heuristic_bias_check(
    status_text: str,
    spoiler_text: str,
    attachments: List[Dict[str, Any]],
    links: List[Dict[str, str]],
    referenced_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Perform a deterministic check for common racial or gender bias terms across all sources."""
    findings: List[Dict[str, str]] = []
    main_text_biased = False
    suggested_wording = status_text or ""

    def scan_text_block(text: str, source_label: str, is_main_text: bool = False) -> None:
        nonlocal main_text_biased, suggested_wording
        if not text:
            return
        for pattern, bias_type, explanation, replacement in _HEURISTIC_BIAS_PATTERNS:
            if pattern.search(text):
                findings.append(
                    {
                        "source": source_label,
                        "bias_type": bias_type,
                        "explanation": explanation,
                    }
                )
                if is_main_text:
                    main_text_biased = True
                    suggested_wording = pattern.sub(replacement, suggested_wording)

    scan_text_block(status_text, "Post Text", is_main_text=True)
    if spoiler_text:
        scan_text_block(spoiler_text, "Content Warning", is_main_text=False)

    for idx, att in enumerate(attachments):
        label = att.get("label") or f"Attachment {idx + 1}"
        alt_text = att.get("alt_text") or ""
        scan_text_block(alt_text, f"{label} (Alt-Text)")

    for link in links:
        url = link.get("url", "")
        combined = f"{url} {link.get('title', '')} {link.get('excerpt', '')}"
        scan_text_block(combined, f"Link ({url})")

    for ref in referenced_messages:
        author = ref.get("author") or "Referenced Post"
        ref_text = f"{ref.get('spoiler_text', '')} {ref.get('content', '')} {' '.join(ref.get('media_descriptions') or [])}"
        scan_text_block(ref_text, f"Referenced Message ({author})")

    checked_sources = ["Post Text"]
    if spoiler_text:
        checked_sources.append("Content Warning")
    for idx, att in enumerate(attachments):
        checked_sources.append(att.get("label") or f"Attachment {idx + 1}")
    for link in links:
        checked_sources.append(f"Link ({link.get('url', '')})")
    for ref in referenced_messages:
        checked_sources.append(
            f"Referenced Message ({ref.get('author') or 'Reply Target'})"
        )

    has_bias = len(findings) > 0
    if has_bias:
        locations = sorted({f["source"] for f in findings})
        summary = (
            f"Potential racial or gender bias was detected in: {', '.join(locations)}. "
            "Review the findings below before publishing."
        )
    else:
        summary = (
            f"Checked {', '.join(checked_sources)}. "
            "No explicit or implicit racial or gender bias was detected."
        )

    return {
        "has_bias": has_bias,
        "summary": summary,
        "checked_sources": checked_sources,
        "findings": findings,
        "main_text_biased": main_text_biased,
        "suggested_wording": (
            suggested_wording
            if main_text_biased and suggested_wording.strip() != (status_text or "").strip()
            else ""
        ),
    }


def build_bias_check_prompt(
    status_text: str,
    spoiler_text: str,
    attachments: List[Dict[str, Any]],
    links: List[Dict[str, str]],
    referenced_messages: List[Dict[str, Any]],
) -> str:
    """Build the structured prompt for the AI model to check for racial or gender bias."""
    sections = [
        "You are an inclusive communication and implicit-bias reviewer for a Mastodon client.",
        "Before the user publishes their post, analyze ALL of the following components for explicit or implicit racial, ethnic, or gender bias (including subtle stereotypes, microaggressions, exclusionary default-male/default-white assumptions, gender-exclusive language, or biased imagery/links):",
        "",
        f"1. MAIN POST TEXT:\n{status_text or '(empty)'}",
    ]
    if spoiler_text:
        sections.append(f"\n2. CONTENT WARNING / SPOILER TEXT:\n{spoiler_text}")

    if attachments:
        att_lines = []
        for idx, att in enumerate(attachments):
            label = att.get("label") or f"Attachment {idx + 1}"
            alt = att.get("alt_text") or "(no alt-text)"
            att_lines.append(f"- {label}: Alt-Text = {alt}")
        sections.append(
            "\n3. ATTACHMENTS (also inspect any attached images/video frames provided):\n"
            + "\n".join(att_lines)
        )
    else:
        sections.append("\n3. ATTACHMENTS:\n(none)")

    if links:
        link_lines = []
        for lk in links:
            url = lk.get("url", "")
            title = lk.get("title", "")
            excerpt = lk.get("excerpt", "")
            link_lines.append(
                f"- URL: {url}\n  Title: {title or '(unavailable)'}\n  Excerpt: {excerpt or '(unavailable)'}"
            )
        sections.append("\n4. LINKS IN POST:\n" + "\n".join(link_lines))
    else:
        sections.append("\n4. LINKS IN POST:\n(none)")

    if referenced_messages:
        ref_lines = []
        for ref in referenced_messages:
            author = ref.get("author") or "Unknown author"
            cw = ref.get("spoiler_text") or ""
            content = ref.get("content") or ""
            media_descs = ref.get("media_descriptions") or []
            ref_lines.append(
                f"- Referenced Message from {author}:\n"
                f"  CW: {cw or '(none)'}\n"
                f"  Text: {content}\n"
                f"  Media Alt-Text: {'; '.join(media_descs) if media_descs else '(none)'}"
            )
        sections.append("\n5. REFERENCED / REPLIED-TO MESSAGES:\n" + "\n".join(ref_lines))
    else:
        sections.append("\n5. REFERENCED / REPLIED-TO MESSAGES:\n(none)")

    sections.extend(
        [
            "",
            "Respond ONLY with a valid JSON object matching this exact schema (no markdown code fences, no extra commentary):",
            "{",
            '  "has_bias": true or false,',
            '  "summary": "A brief 1-2 sentence summary of what was checked and whether any racial or gender bias was found.",',
            '  "findings": [',
            "    {",
            '      "source": "Exact location where bias was found (e.g., \'Post Text\', \'Content Warning\', \'Attachment 1\', \'Link (https://...)\', or \'Referenced Message (@user)\')",',
            '      "bias_type": "Type of bias (e.g., \'Implicit Racial Bias\', \'Implicit Gender Bias\')",',
            '      "explanation": "Brief, constructive explanation of the potential bias in that specific source."',
            "    }",
            "  ],",
            '  "main_text_biased": true or false,',
            '  "suggested_wording": "If main_text_biased is true, provide complete alternate wording for the MAIN POST TEXT that conveys the user\'s intended message in a less biased, more inclusive way. If main_text_biased is false, use an empty string."',
            "}",
        ]
    )
    return "\n".join(sections)


def _call_provider_for_bias_check(
    ai_config: Dict[str, Any],
    prompt: str,
    images_b64: List[str],
) -> Tuple[str, str, str]:
    """Call the configured AI provider (Ollama, Gemini, OpenAI, or Anthropic) with text and optional images."""
    provider = (ai_config.get("provider") or "ollama").lower()
    raw_images = [_strip_data_url(img)[1] for img in images_b64 if img]

    if provider == "ollama":
        ollama_cfg = ai_config.get("ollama", {})
        base_url = (ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        model = (ollama_cfg.get("model") or "llava:latest").strip()
        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if raw_images:
            msg["images"] = raw_images
        payload = {
            "model": model,
            "messages": [msg],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result.get("message", {}).get("content", "")
            return content, provider, model

    if provider == "gemini":
        gemini_cfg = ai_config.get("gemini", {})
        api_key = (gemini_cfg.get("api_key") or "").strip()
        if not api_key:
            raise AICaptionerError("Gemini API key not configured.")
        model = (gemini_cfg.get("model") or "gemini-2.5-flash").strip()
        parts: List[Dict[str, Any]] = []
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
        parts.append({"text": prompt})
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}"
        )
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 800},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(
            req, timeout=60, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            content_parts = (
                candidates[0].get("content", {}).get("parts", [])
                if candidates
                else []
            )
            text = "".join(p.get("text", "") for p in content_parts).strip()
            return text, provider, model

    if provider == "openai":
        openai_cfg = ai_config.get("openai", {})
        base_url = (
            openai_cfg.get("base_url") or "https://api.openai.com/v1"
        ).rstrip("/")
        api_key = (openai_cfg.get("api_key") or "").strip()
        is_local = any(
            host in base_url
            for host in ("localhost", "127.0.0.1", "0.0.0.0", ".local")
        )
        if not api_key and not is_local:
            raise AICaptionerError("OpenAI API key not configured.")
        model = (openai_cfg.get("model") or "gpt-4o-mini").strip()
        if images_b64:
            content_blocks: Any = [{"type": "text", "text": prompt}]
            for img_str in images_b64:
                mime, clean_b64 = _strip_data_url(img_str)
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{clean_b64}",
                            "detail": "auto",
                        },
                    }
                )
        else:
            content_blocks = prompt
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content_blocks}],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            req, timeout=60, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            choices = result.get("choices", [])
            text = (
                choices[0].get("message", {}).get("content", "")
                if choices
                else ""
            )
            return text, provider, model

    if provider == "anthropic":
        anthropic_cfg = ai_config.get("anthropic", {})
        api_key = (anthropic_cfg.get("api_key") or "").strip()
        if not api_key:
            raise AICaptionerError("Anthropic API key not configured.")
        model = (
            anthropic_cfg.get("model") or "claude-3-7-sonnet-latest"
        ).strip()
        content_blocks = []
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
        payload = {
            "model": model,
            "max_tokens": 800,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            req, timeout=60, context=_create_ssl_context()
        ) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            blocks = result.get("content", [])
            text = "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            ).strip()
            return text, provider, model

    raise AICaptionerError(f"Unsupported AI provider: {provider}")


def _parse_bias_json_response(
    raw_text: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse JSON bias analysis output from an LLM, merging with fallback metadata."""
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate = json_match.group(0) if json_match else cleaned
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return fallback
    except Exception:
        return fallback

    raw_findings = parsed.get("findings")
    findings: List[Dict[str, str]] = []
    if isinstance(raw_findings, list):
        for f in raw_findings:
            if isinstance(f, dict):
                source = str(f.get("source") or "Post Text").strip()
                bias_type = str(
                    f.get("bias_type") or "Implicit Racial or Gender Bias"
                ).strip()
                explanation = str(f.get("explanation") or "").strip()
                if explanation:
                    findings.append(
                        {
                            "source": source,
                            "bias_type": bias_type,
                            "explanation": explanation,
                        }
                    )

    has_bias = bool(parsed.get("has_bias")) or len(findings) > 0
    main_text_biased = bool(parsed.get("main_text_biased")) or any(
        "post text" in f["source"].lower() or "main text" in f["source"].lower()
        for f in findings
    )
    suggested_wording = str(parsed.get("suggested_wording") or "").strip()
    if main_text_biased and not suggested_wording and fallback.get("suggested_wording"):
        suggested_wording = fallback["suggested_wording"]

    summary = str(parsed.get("summary") or "").strip() or fallback["summary"]
    return {
        "has_bias": has_bias,
        "summary": summary,
        "checked_sources": fallback["checked_sources"],
        "findings": findings,
        "main_text_biased": main_text_biased,
        "suggested_wording": suggested_wording if main_text_biased else "",
    }


def check_post_for_bias(
    ai_config: Dict[str, Any],
    status_text: str = "",
    spoiler_text: str = "",
    attachments: Optional[List[Dict[str, Any]]] = None,
    referenced_messages: Optional[List[Dict[str, Any]]] = None,
    links: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Check post text, attachments, links, and referenced messages for racial or gender bias."""
    attachments = list(attachments or [])
    referenced_messages = list(referenced_messages or [])

    if links is None:
        urls = extract_urls_from_text(status_text)
        links = [fetch_link_preview_text(u) for u in urls[:5]]
    else:
        links = list(links)

    heuristic_result = _heuristic_bias_check(
        status_text=status_text,
        spoiler_text=spoiler_text,
        attachments=attachments,
        links=links,
        referenced_messages=referenced_messages,
    )

    images_b64: List[str] = []
    for att in attachments:
        att_imgs = att.get("images_b64") or []
        if isinstance(att_imgs, list):
            images_b64.extend(att_imgs[:2])
        elif isinstance(att.get("image_b64"), str) and att["image_b64"]:
            images_b64.append(att["image_b64"])

    prompt = build_bias_check_prompt(
        status_text=status_text,
        spoiler_text=spoiler_text,
        attachments=attachments,
        links=links,
        referenced_messages=referenced_messages,
    )

    try:
        raw_resp, provider_used, model_used = _call_provider_for_bias_check(
            ai_config=ai_config,
            prompt=prompt,
            images_b64=images_b64,
        )
        parsed = _parse_bias_json_response(raw_resp, fallback=heuristic_result)
        parsed["provider"] = provider_used
        parsed["model"] = model_used
        parsed["links_checked"] = [lk.get("url", "") for lk in links]
        return parsed
    except Exception:
        heuristic_result["provider"] = "heuristic-fallback"
        heuristic_result["model"] = "built-in"
        heuristic_result["links_checked"] = [lk.get("url", "") for lk in links]
        return heuristic_result
