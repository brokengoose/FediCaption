"""Main HTTP server and CLI entry point for FediCaption."""

import argparse
import base64
import json
import mimetypes
import os
import sys
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from server.ai_captioner import (
    AICaptionerError,
    _strip_data_url,
    generate_alt_text,
    list_ollama_models,
)
from server.config import (
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


def _get_mastodon_client(config: Optional[Dict[str, Any]] = None) -> MastodonClient:
    cfg = config or load_config()
    m_cfg = cfg.get("mastodon", {})
    return MastodonClient(
        instance_url=m_cfg.get("instance_url", "https://mastodon.social"),
        access_token=m_cfg.get("access_token", ""),
    )


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
                self._send_json({"ok": True, "service": "FediCaption"})
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

            if path == "/api/ai/caption":
                cfg = load_config()
                ai_cfg = cfg.get("ai", {})
                # Allow per-request provider/model override from UI
                if "ai_override" in body and isinstance(body["ai_override"], dict):
                    from server.config import _deep_merge

                    ai_cfg = _deep_merge(ai_cfg, body["ai_override"])

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
