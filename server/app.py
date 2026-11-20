"""Main HTTP server and CLI entry point for FediCaption."""

import argparse
import base64
import json
import mimetypes
import os
import re
import ssl
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from server import __version__
from server.ai_captioner import (
    AICaptionerError,
    _create_ssl_context,
    _strip_data_url,
    generate_alt_text,
    list_ollama_models,
    list_provider_models,
)
from server.bias_checker import check_post_for_bias
from server.config import (
    _deep_merge,
    load_config,
    sanitize_config_for_client,
    save_config,
)
from server.mastodon_client import (
    MastodonAPIError,
    MastodonClient,
    fetch_remote_media_as_base64,
    normalize_instance_url,
)


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
GITHUB_REPO = "brokengoose/FediCaption"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"


def _parse_version_tuple(version_str: str) -> Tuple[int, ...]:
    """Parse a semantic version string (e.g., 'v1.2.3' or '1.2.3') into a comparable tuple of ints."""
    cleaned = (version_str or "").strip().lstrip("vV")
    parts = re.findall(r"\d+", cleaned)
    if not parts:
        return (0,)
    return tuple(int(p) for p in parts)


def check_github_update(current_version: str = __version__) -> Dict[str, Any]:
    """Check GitHub for a newer version of FediCaption via Releases or main branch __version__."""
    headers = {
        "User-Agent": f"FediCaption/{current_version} (+{GITHUB_REPO_URL})",
        "Accept": "application/vnd.github+json, text/plain;q=0.9, */*;q=0.8",
    }
    ssl_ctx = _create_ssl_context()
    latest_version = current_version
    release_url = GITHUB_REPO_URL
    checked = False

    # 1. Check latest GitHub Release first
    release_api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        req = urllib.request.Request(release_api_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5, context=ssl_ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            if isinstance(data, dict):
                tag = (
                    data.get("tag_name")
                    or data.get("name")
                    or data.get("version")
                    or ""
                ).strip()
                if tag:
                    latest_version = tag.lstrip("vV")
                    release_url = data.get("html_url") or GITHUB_REPO_URL
                    checked = True
    except Exception:
        pass

    # 2. Also check server/__init__.py on GitHub main branch
    raw_init_url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/server/__init__.py"
    )
    try:
        req = urllib.request.Request(raw_init_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5, context=ssl_ctx) as resp:
            raw_py = resp.read().decode("utf-8", errors="replace")
            match = re.search(
                r'__version__\s*=\s*["\']([^"\']+)["\']', raw_py
            )
            if match:
                branch_ver = match.group(1).strip().lstrip("vV")
                if _parse_version_tuple(branch_ver) > _parse_version_tuple(
                    latest_version
                ):
                    latest_version = branch_ver
                checked = True
    except Exception:
        pass

    update_available = _parse_version_tuple(latest_version) > _parse_version_tuple(
        current_version
    )
    return {
        "ok": True,
        "checked": checked,
        "current_version": current_version.lstrip("vV"),
        "latest_version": latest_version.lstrip("vV"),
        "update_available": update_available,
        "release_url": release_url,
    }


def _get_mastodon_client(config: Optional[Dict[str, Any]] = None) -> MastodonClient:
    """Create a MastodonClient initialized with the stored or provided instance config."""
    cfg = config or load_config()
    m_cfg = cfg.get("mastodon", {})
    return MastodonClient(
        instance_url=m_cfg.get("instance_url", "https://mastodon.social"),
        access_token=m_cfg.get("access_token", ""),
    )


def _resolve_effective_ai_config(body: Dict[str, Any]) -> Dict[str, Any]:
    """Load stored AI configuration and merge any per-request ai_override fields."""
    cfg = load_config()
    ai_cfg = cfg.get("ai", {})
    if isinstance(body.get("ai_override"), dict):
        ai_cfg = _deep_merge(ai_cfg, body["ai_override"])
    return ai_cfg


def _fetch_and_save_instance_limits(client: MastodonClient) -> Dict[str, int]:
    """Query the Mastodon server for its character/media limits and persist them to config."""
    try:
        limits = client.get_instance_limits()
        save_config({"mastodon": limits})
        return limits
    except Exception:
        cfg = load_config().get("mastodon", {})
        return {
            "max_post_chars": int(cfg.get("max_post_chars") or 500),
            "max_media_attachments": int(cfg.get("max_media_attachments") or 4),
            "alt_text_limit": int(cfg.get("alt_text_limit") or 1500),
        }


class FediCaptionHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for FediCaption API and static web interface."""

    server_version = "FediCaption/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet clean logging
        if os.environ.get("FEDICAPTION_QUIET") == "1":
            return
        sys.stderr.write(f"[FediCaption] {self.address_string()} - {format % args}\n")

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        # Allow up to 60MB for video/image base64 uploads
        if length > 60 * 1024 * 1024:
            raise ValueError("Payload exceeds 60MB limit.")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, rel_path: str) -> None:
        if not rel_path or rel_path == "/":
            rel_path = "index.html"
        rel_path = rel_path.lstrip("/")
        target = (STATIC_DIR / rel_path).resolve()

        # Prevent directory traversal
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self._send_error_json("Forbidden", HTTPStatus.FORBIDDEN)
            return

        if not target.is_file():
            self._send_error_json("Not found", HTTPStatus.NOT_FOUND)
            return

        mime_type, _ = mimetypes.guess_type(str(target))
        if not mime_type:
            mime_type = "application/octet-stream"
        if mime_type.startswith("text/") or mime_type in (
            "application/javascript",
            "application/json",
        ):
            mime_type = f"{mime_type}; charset=utf-8"

        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = dict(urllib.parse.parse_qsl(parsed.query))

        try:
            if path == "/api/health":
                self._send_json(
                    {
                        "ok": True,
                        "service": "FediCaption",
                        "version": __version__,
                    }
                )
                return

            if path == "/api/version":
                info = check_github_update()
                self._send_json(info)
                return

            if path == "/api/config":
                cfg = load_config()
                self._send_json({"ok": True, "config": sanitize_config_for_client(cfg)})
                return

            if path == "/api/ai/ollama-models":
                cfg = load_config()
                base_url = query.get("base_url") or cfg.get("ai", {}).get(
                    "ollama", {}
                ).get("base_url", "http://localhost:11434")
                info = list_ollama_models(base_url)
                self._send_json({"ok": True, **info})
                return

            if path == "/api/ai/models":
                cfg = load_config()
                ai_cfg = cfg.get("ai", {})
                provider = query.get("provider") or ai_cfg.get("provider", "ollama")
                info = list_provider_models(
                    provider=provider,
                    ai_config=ai_cfg,
                    overrides=query,
                )
                self._send_json({"ok": True, **info})
                return

            if path == "/api/mastodon/instance":
                cfg = load_config()
                instance_url = query.get("instance_url") or cfg.get("mastodon", {}).get(
                    "instance_url", "https://mastodon.social"
                )
                client = MastodonClient(
                    instance_url=instance_url,
                    access_token=cfg.get("mastodon", {}).get("access_token", ""),
                )
                limits = client.get_instance_limits()
                updated = save_config(
                    {
                        "mastodon": {
                            "instance_url": client.instance_url,
                            **limits,
                        }
                    }
                )
                self._send_json(
                    {
                        "ok": True,
                        "limits": limits,
                        "config": sanitize_config_for_client(updated),
                    }
                )
                return

            if path == "/api/mastodon/verify":
                client = _get_mastodon_client()
                account = client.verify_credentials()
                limits = _fetch_and_save_instance_limits(client)
                updated = save_config({"mastodon": {"account": account, **limits}})
                self._send_json(
                    {
                        "ok": True,
                        "account": account,
                        "limits": limits,
                        "config": sanitize_config_for_client(updated),
                    }
                )
                return

            if path == "/api/mastodon/timeline":
                client = _get_mastodon_client()
                cfg = load_config()
                acct = cfg.get("mastodon", {}).get("account") or {}
                timeline_type = query.get("type", "home")
                max_id = query.get("max_id")
                since_id = query.get("since_id")
                limit = int(query.get("limit", "25"))
                target_account_id = query.get("account_id") or acct.get("id")
                statuses = client.get_timeline(
                    timeline_type=timeline_type,
                    max_id=max_id,
                    since_id=since_id,
                    limit=limit,
                    account_id=target_account_id,
                )
                self._send_json({"ok": True, "statuses": statuses})
                return

            if path == "/api/mastodon/account":
                account_id = query.get("id", "").strip()
                if not account_id:
                    self._send_error_json("Missing account id", 400)
                    return
                client = _get_mastodon_client()
                account = client.get_account(account_id)
                relationship = client.get_account_relationship(account_id)
                self._send_json(
                    {
                        "ok": True,
                        "account": account,
                        "relationship": relationship,
                    }
                )
                return

            if path == "/api/mastodon/notifications":
                client = _get_mastodon_client()
                max_id = query.get("max_id")
                limit = int(query.get("limit", "25"))
                notifications = client.get_notifications(max_id=max_id, limit=limit)
                self._send_json({"ok": True, "notifications": notifications})
                return

            if path == "/api/mastodon/thread":
                status_id = query.get("id", "").strip()
                if not status_id:
                    self._send_error_json("Missing status id", 400)
                    return
                client = _get_mastodon_client()
                thread = client.get_status_thread(status_id)
                self._send_json({"ok": True, **thread})
                return

            if path == "/api/mastodon/status/source":
                status_id = query.get("id", "").strip()
                if not status_id:
                    self._send_error_json("Missing status id", 400)
                    return
                client = _get_mastodon_client()
                source = client.get_status_source(status_id)
                self._send_json({"ok": True, "source": source})
                return

            if path == "/api/mastodon/quote":
                quote_url = query.get("url", "").strip()
                if not quote_url:
                    self._send_error_json("Missing quoted post url", 400)
                    return
                client = _get_mastodon_client()
                quoted_status = client.resolve_quoted_status(quote_url)
                self._send_json({"ok": True, "status": quoted_status})
                return

            if path == "/api/mastodon/oauth/callback":
                code = query.get("code", "").strip()
                if not code:
                    self._send_error_json("Missing OAuth authorization code", 400)
                    return
                cfg = load_config()
                m_cfg = cfg.get("mastodon", {})
                client = MastodonClient(m_cfg.get("instance_url", ""))
                host = self.headers.get("Host", "127.0.0.1:8765")
                redirect_uri = f"http://{host}/api/mastodon/oauth/callback"
                token_data = client.exchange_oauth_code(
                    client_id=m_cfg.get("client_id", ""),
                    client_secret=m_cfg.get("client_secret", ""),
                    code=code,
                    redirect_uri=redirect_uri,
                )
                access_token = token_data.get("access_token", "")
                client.access_token = access_token
                account = client.verify_credentials()
                limits = _fetch_and_save_instance_limits(client)
                save_config(
                    {
                        "mastodon": {
                            "access_token": access_token,
                            "account": account,
                            **limits,
                        }
                    }
                )
                # Redirect back to main UI
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", "/?oauth=success")
                self.end_headers()
                return

            # Static files
            self._serve_static(path)

        except MastodonAPIError as m_err:
            self._send_error_json(str(m_err), status=m_err.status_code)
        except AICaptionerError as ai_err:
            self._send_error_json(str(ai_err), status=400)
        except Exception as exc:
            self._send_error_json(f"Server error: {exc}", status=500)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()

            if path == "/api/config":
                updated = save_config(body)
                if isinstance(body.get("mastodon"), dict) and body["mastodon"].get(
                    "instance_url"
                ):
                    client = _get_mastodon_client(updated)
                    limits = _fetch_and_save_instance_limits(client)
                    updated = save_config({"mastodon": limits})
                self._send_json(
                    {"ok": True, "config": sanitize_config_for_client(updated)}
                )
                return

            if path == "/api/ai/models":
                cfg = load_config()
                ai_cfg = cfg.get("ai", {})
                provider = body.get("provider") or ai_cfg.get("provider", "ollama")
                info = list_provider_models(
                    provider=provider,
                    ai_config=ai_cfg,
                    overrides=body,
                )
                self._send_json({"ok": True, **info})
                return

            if path == "/api/ai/caption":
                ai_cfg = _resolve_effective_ai_config(body)

                media_type = body.get("media_type", "image")
                images_b64 = body.get("images_b64") or []
                raw_media_b64 = body.get("raw_media_b64")
                raw_media_mime = body.get("raw_media_mime")
                remote_url = body.get("remote_url")

                if remote_url and not images_b64 and not raw_media_b64:
                    mime, b64 = fetch_remote_media_as_base64(remote_url)
                    raw_media_mime = mime
                    raw_media_b64 = b64
                    if mime.startswith("image/"):
                        images_b64 = [b64]

                result = generate_alt_text(
                    ai_config=ai_cfg,
                    media_type=media_type,
                    images_b64=images_b64,
                    raw_media_b64=raw_media_b64,
                    raw_media_mime=raw_media_mime,
                    post_context=body.get("post_context", ""),
                    detail_level=body.get("detail_level"),
                    custom_instructions=body.get("custom_instructions"),
                )
                self._send_json({"ok": True, **result})
                return

            if path == "/api/ai/bias-check":
                ai_cfg = _resolve_effective_ai_config(body)

                attachments = body.get("attachments") or []
                for att in attachments:
                    if (
                        isinstance(att, dict)
                        and att.get("remote_url")
                        and not att.get("images_b64")
                    ):
                        try:
                            mime, b64 = fetch_remote_media_as_base64(
                                att["remote_url"]
                            )
                            if mime.startswith("image/"):
                                att["images_b64"] = [b64]
                        except Exception:
                            pass

                bias_result = check_post_for_bias(
                    ai_config=ai_cfg,
                    status_text=body.get("status_text", ""),
                    spoiler_text=body.get("spoiler_text", ""),
                    attachments=attachments,
                    referenced_messages=body.get("referenced_messages") or [],
                )
                self._send_json({"ok": True, **bias_result})
                return

            if path == "/api/mastodon/oauth/start":
                instance_url = normalize_instance_url(
                    body.get("instance_url", "https://mastodon.social")
                )
                host = self.headers.get("Host", "127.0.0.1:8765")
                redirect_uri = body.get(
                    "redirect_uri", f"http://{host}/api/mastodon/oauth/callback"
                )
                client = MastodonClient(instance_url=instance_url)
                app_data = client.register_oauth_app(redirect_uri=redirect_uri)
                client_id = app_data.get("client_id", "")
                client_secret = app_data.get("client_secret", "")
                limits = _fetch_and_save_instance_limits(client)
                save_config(
                    {
                        "mastodon": {
                            "instance_url": instance_url,
                            "client_id": client_id,
                            "client_secret": client_secret,
                            **limits,
                        }
                    }
                )
                authorize_url = client.get_authorize_url(
                    client_id=client_id, redirect_uri=redirect_uri
                )
                self._send_json(
                    {
                        "ok": True,
                        "instance_url": instance_url,
                        "authorize_url": authorize_url,
                        "redirect_uri": redirect_uri,
                        "limits": limits,
                    }
                )
                return

            if path == "/api/mastodon/oauth/token":
                # Direct access token login or manual OAuth code exchange
                cfg = load_config()
                m_cfg = cfg.get("mastodon", {})
                instance_url = normalize_instance_url(
                    body.get("instance_url") or m_cfg.get("instance_url", "")
                )
                access_token = (body.get("access_token") or "").strip()
                code = (body.get("code") or "").strip()

                client = MastodonClient(instance_url=instance_url)
                if code and not access_token:
                    redirect_uri = body.get(
                        "redirect_uri", "urn:ietf:wg:oauth:2.0:oob"
                    )
                    token_data = client.exchange_oauth_code(
                        client_id=m_cfg.get("client_id", ""),
                        client_secret=m_cfg.get("client_secret", ""),
                        code=code,
                        redirect_uri=redirect_uri,
                    )
                    access_token = token_data.get("access_token", "")

                if not access_token:
                    self._send_error_json("Access token or authorization code required", 400)
                    return

                client.access_token = access_token
                account = client.verify_credentials()
                limits = _fetch_and_save_instance_limits(client)
                updated = save_config(
                    {
                        "mastodon": {
                            "instance_url": instance_url,
                            "access_token": access_token,
                            "account": account,
                            **limits,
                        }
                    }
                )
                self._send_json(
                    {
                        "ok": True,
                        "account": account,
                        "limits": limits,
                        "config": sanitize_config_for_client(updated),
                    }
                )
                return

            if path == "/api/mastodon/logout":
                updated = save_config(
                    {
                        "mastodon": {
                            "access_token": "",
                            "account": None,
                        }
                    }
                )
                self._send_json(
                    {"ok": True, "config": sanitize_config_for_client(updated)}
                )
                return

            if path == "/api/mastodon/media":
                client = _get_mastodon_client()
                data_b64 = body.get("file_b64", "")
                filename = body.get("filename", "upload.jpg")
                mime_type = body.get("mime_type", "")
                description = body.get("description", "")

                if not data_b64:
                    self._send_error_json("No media file data provided", 400)
                    return

                detected_mime, clean_b64 = _strip_data_url(data_b64)
                if not mime_type:
                    mime_type = detected_mime or "application/octet-stream"
                file_bytes = base64.b64decode(clean_b64)

                media_obj = client.upload_media(
                    file_bytes=file_bytes,
                    filename=filename,
                    mime_type=mime_type,
                    description=description,
                )
                self._send_json({"ok": True, "media": media_obj})
                return

            if path == "/api/mastodon/status":
                client = _get_mastodon_client()
                status_text = body.get("status", "")
                media_ids = body.get("media_ids") or []
                spoiler_text = body.get("spoiler_text", "")
                visibility = body.get("visibility", "public")
                in_reply_to_id = body.get("in_reply_to_id")
                sensitive = bool(body.get("sensitive", False))
                language = body.get("language", "en")

                if not status_text.strip() and not media_ids:
                    self._send_error_json(
                        "Cannot publish an empty post without text or media.", 400
                    )
                    return

                posted = client.post_status(
                    status_text=status_text,
                    media_ids=media_ids,
                    spoiler_text=spoiler_text,
                    visibility=visibility,
                    in_reply_to_id=in_reply_to_id,
                    sensitive=sensitive,
                    language=language,
                )
                self._send_json({"ok": True, "status": posted})
                return

            if path == "/api/mastodon/status/action":
                client = _get_mastodon_client()
                status_id = body.get("status_id", "").strip()
                action = body.get("action", "").strip()
                if not status_id or not action:
                    self._send_error_json("status_id and action are required", 400)
                    return
                result = client.status_action(status_id=status_id, action=action)
                self._send_json({"ok": True, "status": result})
                return

            if path == "/api/mastodon/account/subscribe":
                client = _get_mastodon_client()
                account_id = body.get("account_id", "").strip()
                action = body.get("action", "follow").strip()
                if not account_id:
                    self._send_error_json("account_id is required", 400)
                    return
                relationship = client.set_account_subscription(
                    account_id=account_id, action=action
                )
                self._send_json({"ok": True, "relationship": relationship})
                return

            self._send_error_json(f"Unknown endpoint: {path}", 404)

        except MastodonAPIError as m_err:
            self._send_error_json(str(m_err), status=m_err.status_code)
        except AICaptionerError as ai_err:
            self._send_error_json(str(ai_err), status=400)
        except Exception as exc:
            self._send_error_json(f"Server error: {exc}", status=500)

    def do_PUT(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
            if path == "/api/mastodon/media":
                client = _get_mastodon_client()
                media_id = body.get("media_id", "").strip()
                description = body.get("description", "")
                if not media_id:
                    self._send_error_json("media_id is required", 400)
                    return
                updated = client.update_media_description(media_id, description)
                self._send_json({"ok": True, "media": updated})
                return

            if path == "/api/mastodon/status":
                client = _get_mastodon_client()
                status_id = body.get("status_id", "").strip()
                if not status_id:
                    self._send_error_json("status_id is required", 400)
                    return
                status_text = body.get("status", "")
                media_ids = body.get("media_ids")
                media_attributes = body.get("media_attributes")
                spoiler_text = body.get("spoiler_text", "")
                sensitive = bool(body.get("sensitive", False))
                language = body.get("language", "en")

                if not status_text.strip() and not media_ids:
                    self._send_error_json(
                        "Cannot save an empty post without text or media.", 400
                    )
                    return

                updated_status = client.edit_status(
                    status_id=status_id,
                    status_text=status_text,
                    media_ids=media_ids,
                    media_attributes=media_attributes,
                    spoiler_text=spoiler_text,
                    sensitive=sensitive,
                    language=language,
                )
                self._send_json({"ok": True, "status": updated_status})
                return

            self._send_error_json(f"Unknown endpoint: {path}", 404)
        except MastodonAPIError as m_err:
            self._send_error_json(str(m_err), status=m_err.status_code)
        except Exception as exc:
            self._send_error_json(f"Server error: {exc}", status=500)


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create and return a configured ThreadingHTTPServer instance."""
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), FediCaptionHandler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FediCaption — Accessible Mastodon Client with AI Alt-Text for Images & Videos"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("FEDICAPTION_HOST", "127.0.0.1"),
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("FEDICAPTION_PORT", "8765")),
        help="Port to bind (default: 8765)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the web UI in the default browser",
    )
    args = parser.parse_args()

    # Try binding the requested port, or find the next available port
    port = args.port
    server: Optional[ThreadingHTTPServer] = None
    for candidate_port in range(port, port + 20):
        try:
            server = create_server(args.host, candidate_port)
            port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        print(f"Error: Could not bind to any port near {args.port}", file=sys.stderr)
        sys.exit(1)

    url = f"http://{args.host}:{port}"
    print("=" * 64)
    print("  FediCaption — Accessible Mastodon Client with AI Alt-Text")
    print(f"  Running at: {url}")
    print("  Press Ctrl+C to quit.")
    print("=" * 64)

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down FediCaption...")
        server.server_close()


if __name__ == "__main__":
    main()
