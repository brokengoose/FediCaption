"""Mastodon REST API & OAuth 2.0 Client using Python standard library."""

import base64
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_MAX_POST_CHARS = 500
DEFAULT_MAX_MEDIA_ATTACHMENTS = 4
DEFAULT_ALT_TEXT_LIMIT = 1500


class MastodonAPIError(Exception):
    """Raised when a Mastodon API request fails."""

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def extract_instance_limits(instance_data: Dict[str, Any]) -> Dict[str, int]:
    """Extract maximum post length, media attachment count, and alt-text limit from instance metadata."""
    if not isinstance(instance_data, dict):
        return {
            "max_post_chars": DEFAULT_MAX_POST_CHARS,
            "max_media_attachments": DEFAULT_MAX_MEDIA_ATTACHMENTS,
            "alt_text_limit": DEFAULT_ALT_TEXT_LIMIT,
        }

    config = instance_data.get("configuration") or {}
    statuses_cfg = config.get("statuses") or {}
    media_cfg = config.get("media_attachments") or {}

    # Check standard Mastodon v2/v1 configuration.statuses.max_characters,
    # then fallback to Glitch-soc / Hometown / Pleroma / Akkoma fields
    raw_max_chars = (
        statuses_cfg.get("max_characters")
        or instance_data.get("max_toot_chars")
        or instance_data.get("max_characters")
    )
    try:
        max_post_chars = (
            int(raw_max_chars)
            if raw_max_chars is not None
            else DEFAULT_MAX_POST_CHARS
        )
        if max_post_chars <= 0:
            max_post_chars = DEFAULT_MAX_POST_CHARS
    except (ValueError, TypeError):
        max_post_chars = DEFAULT_MAX_POST_CHARS

    raw_max_media = statuses_cfg.get("max_media_attachments")
    try:
        max_media_attachments = (
            int(raw_max_media)
            if raw_max_media is not None
            else DEFAULT_MAX_MEDIA_ATTACHMENTS
        )
        if max_media_attachments <= 0:
            max_media_attachments = DEFAULT_MAX_MEDIA_ATTACHMENTS
    except (ValueError, TypeError):
        max_media_attachments = DEFAULT_MAX_MEDIA_ATTACHMENTS

    raw_alt_limit = media_cfg.get("description_limit")
    try:
        alt_text_limit = (
            int(raw_alt_limit)
            if raw_alt_limit is not None
            else DEFAULT_ALT_TEXT_LIMIT
        )
        if alt_text_limit <= 0:
            alt_text_limit = DEFAULT_ALT_TEXT_LIMIT
    except (ValueError, TypeError):
        alt_text_limit = DEFAULT_ALT_TEXT_LIMIT

    return {
        "max_post_chars": max_post_chars,
        "max_media_attachments": max_media_attachments,
        "alt_text_limit": alt_text_limit,
    }


def normalize_instance_url(instance_url: str) -> str:
    """Normalize a Mastodon instance domain or URL into https://domain."""
    url = (instance_url or "").strip().rstrip("/")
    if not url:
        raise MastodonAPIError("Mastodon instance URL is required.", 400)
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc:
        raise MastodonAPIError(f"Invalid Mastodon instance URL: {instance_url}", 400)
    return f"{parsed.scheme}://{parsed.netloc}"


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context()


def _encode_multipart_formdata(
    fields: Dict[str, str],
    files: List[Tuple[str, str, str, bytes]],
) -> Tuple[bytes, str]:
    """Encode fields and files into a multipart/form-data payload.

    files is a list of (field_name, filename, content_type, file_bytes).
    Returns (body_bytes, content_type_header).
    """
    boundary = f"----FediCaptionBoundary{uuid.uuid4().hex}"
    lines: List[bytes] = []

    for name, value in fields.items():
        if value is None:
            continue
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        lines.append(f"{value}\r\n".encode("utf-8"))

    for field_name, filename, content_type, file_bytes in files:
        safe_filename = filename.replace('"', "_")
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(
            (
                f'Content-Disposition: form-data; name="{field_name}"; '
                f'filename="{safe_filename}"\r\n'
            ).encode("utf-8")
        )
        lines.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        lines.append(file_bytes)
        lines.append(b"\r\n")

    lines.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(lines)
    content_type_header = f"multipart/form-data; boundary={boundary}"
    return body, content_type_header


class MastodonClient:
    """Lightweight, full-featured Mastodon REST API client."""

    def __init__(self, instance_url: str, access_token: str = ""):
        self.instance_url = normalize_instance_url(instance_url)
        self.access_token = (access_token or "").strip()

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        raw_body: Optional[bytes] = None,
        content_type: Optional[str] = None,
        require_auth: bool = False,
        timeout: int = 45,
    ) -> Tuple[Any, int]:
        if require_auth and not self.access_token:
            raise MastodonAPIError(
                "Not signed in to Mastodon. Connect your account in Settings first.",
                401,
            )

        url = f"{self.instance_url}{path}"
        if params:
            query_items: List[Tuple[str, str]] = []
            for k, v in params.items():
                if v is None or v == "":
                    continue
                if isinstance(v, list):
                    for item in v:
                        query_items.append((f"{k}[]", str(item)))
                elif isinstance(v, bool):
                    query_items.append((k, "true" if v else "false"))
                else:
                    query_items.append((k, str(v)))
            if query_items:
                url = f"{url}?{urllib.parse.urlencode(query_items)}"

        headers: Dict[str, str] = {
            "User-Agent": "FediCaption/1.0 (Accessible Mastodon Client)",
            "Accept": "application/json",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"

        data: Optional[bytes] = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        elif raw_body is not None:
            data = raw_body
            if content_type:
                headers["Content-Type"] = content_type

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(
                req, timeout=timeout, context=_ssl_context()
            ) as resp:
                status_code = resp.getcode()
                raw = resp.read().decode("utf-8", errors="replace")
                if not raw.strip():
                    return {}, status_code
                return json.loads(raw), status_code
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
                err_msg = (
                    err_json.get("error_description")
                    or err_json.get("error")
                    or err_body
                )
            except Exception:
                err_msg = err_body or str(http_err)
            raise MastodonAPIError(
                f"Mastodon API error ({http_err.code}): {err_msg}",
                status_code=http_err.code,
            ) from http_err
        except urllib.error.URLError as url_err:
            raise MastodonAPIError(
                f"Could not connect to {self.instance_url}: {url_err.reason}",
                status_code=502,
            ) from url_err

    def register_oauth_app(self, redirect_uri: str) -> Dict[str, Any]:
        """Register FediCaption OAuth application on the Mastodon instance."""
        payload = {
            "client_name": "FediCaption (Accessible AI Alt-Text Client)",
            "redirect_uris": f"{redirect_uri}\nurn:ietf:wg:oauth:2.0:oob",
            "scopes": "read write follow push",
            "website": "https://github.com/brokengoose/FediCaption",
        }
        data, _ = self._request("POST", "/api/v1/apps", json_body=payload)
        return data

    def get_authorize_url(self, client_id: str, redirect_uri: str) -> str:
        """Build the browser OAuth 2.0 authorization URL."""
        params = {
            "client_id": client_id,
            "scope": "read write follow push",
            "redirect_uri": redirect_uri,
            "response_type": "code",
        }
        return f"{self.instance_url}/oauth/authorize?{urllib.parse.urlencode(params)}"

    def exchange_oauth_code(
        self,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """Exchange an OAuth authorization code for an access token."""
        payload = {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
            "code": code.strip(),
            "scope": "read write follow push",
        }
        data, _ = self._request("POST", "/oauth/token", json_body=payload)
        return data

    def verify_credentials(self) -> Dict[str, Any]:
        """Verify current access token and return the authenticated account."""
        data, _ = self._request(
            "GET", "/api/v1/accounts/verify_credentials", require_auth=True
        )
        return data

    def get_instance_info(self) -> Dict[str, Any]:
        """Fetch instance metadata (character limits, description, etc.)."""
        try:
            data, _ = self._request("GET", "/api/v2/instance")
            return data
        except MastodonAPIError:
            data, _ = self._request("GET", "/api/v1/instance")
            return data

    def get_instance_limits(self) -> Dict[str, int]:
        """Query the Mastodon instance for its maximum post length and media limits."""
        info = self.get_instance_info()
        return extract_instance_limits(info)

    def get_timeline(
        self,
        timeline_type: str = "home",
        max_id: Optional[str] = None,
        since_id: Optional[str] = None,
        limit: int = 25,
        account_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch statuses from the requested timeline."""
        params: Dict[str, Any] = {"limit": limit}
        if max_id:
            params["max_id"] = max_id
        if since_id:
            params["since_id"] = since_id

        if timeline_type == "home":
            data, _ = self._request(
                "GET", "/api/v1/timelines/home", params=params, require_auth=True
            )
        elif timeline_type == "local":
            params["local"] = True
            data, _ = self._request("GET", "/api/v1/timelines/public", params=params)
        elif timeline_type in ("federated", "public"):
            data, _ = self._request("GET", "/api/v1/timelines/public", params=params)
        elif timeline_type == "bookmarks":
            data, _ = self._request(
                "GET", "/api/v1/bookmarks", params=params, require_auth=True
            )
        elif timeline_type == "favourites":
            data, _ = self._request(
                "GET", "/api/v1/favourites", params=params, require_auth=True
            )
        elif timeline_type == "mine":
            if not account_id:
                acct = self.verify_credentials()
                account_id = acct.get("id")
            data, _ = self._request(
                "GET",
                f"/api/v1/accounts/{account_id}/statuses",
                params=params,
                require_auth=True,
            )
        elif timeline_type == "account":
            if not account_id:
                raise MastodonAPIError("account_id is required for user timeline", 400)
            data, _ = self._request(
                "GET",
                f"/api/v1/accounts/{account_id}/statuses",
                params=params,
                require_auth=False,
            )
        else:
            raise MastodonAPIError(f"Unknown timeline type: {timeline_type}", 400)

        return data if isinstance(data, list) else []

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """Fetch public profile metadata for a specific account."""
        data, _ = self._request("GET", f"/api/v1/accounts/{account_id}")
        return data if isinstance(data, dict) else {}

    def get_account_relationship(
        self, account_id: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch the authenticated user's relationship (following, notifying) with an account."""
        if not self.access_token:
            return None
        try:
            data, _ = self._request(
                "GET",
                "/api/v1/accounts/relationships",
                params={"id": [account_id]},
                require_auth=True,
            )
            if isinstance(data, list) and len(data) > 0:
                return data[0]
        except MastodonAPIError:
            pass
        return None

    def set_account_subscription(
        self, account_id: str, action: str
    ) -> Dict[str, Any]:
        """Follow/subscribe to or unfollow/unsubscribe from a user's posts."""
        if action in ("follow", "subscribe"):
            data, _ = self._request(
                "POST",
                f"/api/v1/accounts/{account_id}/follow",
                json_body={"reblogs": True},
                require_auth=True,
            )
            return data
        if action in ("unfollow", "unsubscribe"):
            data, _ = self._request(
                "POST",
                f"/api/v1/accounts/{account_id}/unfollow",
                require_auth=True,
            )
            return data
        if action == "notify_on":
            data, _ = self._request(
                "POST",
                f"/api/v1/accounts/{account_id}/follow",
                json_body={"reblogs": True, "notify": True},
                require_auth=True,
            )
            return data
        if action == "notify_off":
            data, _ = self._request(
                "POST",
                f"/api/v1/accounts/{account_id}/follow",
                json_body={"reblogs": True, "notify": False},
                require_auth=True,
            )
            return data
        raise MastodonAPIError(f"Unsupported account subscription action: {action}", 400)

    def get_notifications(
        self, max_id: Optional[str] = None, limit: int = 25
    ) -> List[Dict[str, Any]]:
        """Fetch recent notifications for the authenticated user."""
        params: Dict[str, Any] = {"limit": limit}
        if max_id:
            params["max_id"] = max_id
        data, _ = self._request(
            "GET", "/api/v1/notifications", params=params, require_auth=True
        )
        return data if isinstance(data, list) else []

    def get_status_thread(self, status_id: str) -> Dict[str, Any]:
        """Fetch a status and its ancestor/descendant thread context."""
        status, _ = self._request("GET", f"/api/v1/statuses/{status_id}")
        context, _ = self._request("GET", f"/api/v1/statuses/{status_id}/context")
        return {
            "status": status,
            "ancestors": context.get("ancestors", []),
            "descendants": context.get("descendants", []),
        }

    def upload_media(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        description: str = "",
        wait_for_processing: bool = True,
    ) -> Dict[str, Any]:
        """Upload an image or video attachment to Mastodon with alt-text description."""
        fields: Dict[str, str] = {}
        if description:
            fields["description"] = description

        body, content_type = _encode_multipart_formdata(
            fields=fields,
            files=[("file", filename, mime_type, file_bytes)],
        )

        try:
            data, status_code = self._request(
                "POST",
                "/api/v2/media",
                raw_body=body,
                content_type=content_type,
                require_auth=True,
                timeout=120,
            )
        except MastodonAPIError as exc:
            if exc.status_code == 404:
                data, status_code = self._request(
                    "POST",
                    "/api/v1/media",
                    raw_body=body,
                    content_type=content_type,
                    require_auth=True,
                    timeout=120,
                )
            else:
                raise

        media_id = data.get("id")
        # HTTP 202 means Mastodon is processing the media asynchronously (common for videos)
        if status_code == 202 and wait_for_processing and media_id:
            for _ in range(30):
                time.sleep(1.5)
                check_data, check_code = self._request(
                    "GET", f"/api/v1/media/{media_id}", require_auth=True
                )
                if check_code == 200 and check_data.get("url"):
                    data = check_data
                    break

        return data

    def update_media_description(
        self, media_id: str, description: str
    ) -> Dict[str, Any]:
        """Update the alt-text description of an already-uploaded media attachment."""
        data, _ = self._request(
            "PUT",
            f"/api/v1/media/{media_id}",
            json_body={"description": description},
            require_auth=True,
        )
        return data

    def post_status(
        self,
        status_text: str,
        media_ids: Optional[List[str]] = None,
        spoiler_text: str = "",
        visibility: str = "public",
        in_reply_to_id: Optional[str] = None,
        sensitive: bool = False,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Publish a new status (post) to Mastodon."""
        payload: Dict[str, Any] = {
            "status": status_text,
            "visibility": visibility or "public",
        }
        if media_ids:
            payload["media_ids"] = media_ids
        if spoiler_text and spoiler_text.strip():
            payload["spoiler_text"] = spoiler_text.strip()
            payload["sensitive"] = True
        elif sensitive:
            payload["sensitive"] = True
        if in_reply_to_id:
            payload["in_reply_to_id"] = in_reply_to_id
        if language:
            payload["language"] = language

        data, _ = self._request(
            "POST", "/api/v1/statuses", json_body=payload, require_auth=True
        )
        return data

    def get_status_source(self, status_id: str) -> Dict[str, Any]:
        """Fetch the raw unformatted text and spoiler_text of a status for editing."""
        data, _ = self._request(
            "GET", f"/api/v1/statuses/{status_id}/source", require_auth=True
        )
        return data if isinstance(data, dict) else {}

    def edit_status(
        self,
        status_id: str,
        status_text: str,
        media_ids: Optional[List[str]] = None,
        media_attributes: Optional[List[Dict[str, str]]] = None,
        spoiler_text: str = "",
        sensitive: bool = False,
        language: str = "en",
    ) -> Dict[str, Any]:
        """Edit an existing status on Mastodon (PUT /api/v1/statuses/:id)."""
        payload: Dict[str, Any] = {
            "status": status_text,
            "spoiler_text": spoiler_text.strip() if spoiler_text else "",
            "sensitive": bool(sensitive or (spoiler_text and spoiler_text.strip())),
        }
        if media_ids is not None:
            payload["media_ids"] = media_ids
        if media_attributes:
            payload["media_attributes"] = media_attributes
        if language:
            payload["language"] = language

        data, _ = self._request(
            "PUT",
            f"/api/v1/statuses/{status_id}",
            json_body=payload,
            require_auth=True,
        )
        return data

    def status_action(self, status_id: str, action: str) -> Dict[str, Any]:
        """Perform an action on a status: favourite, unfavourite, reblog, unreblog, bookmark, unbookmark, delete."""
        valid_actions = {
            "favourite",
            "unfavourite",
            "reblog",
            "unreblog",
            "bookmark",
            "unbookmark",
        }
        if action == "delete":
            data, _ = self._request(
                "DELETE", f"/api/v1/statuses/{status_id}", require_auth=True
            )
            return data
        if action not in valid_actions:
            raise MastodonAPIError(f"Unsupported status action: {action}", 400)

        data, _ = self._request(
            "POST", f"/api/v1/statuses/{status_id}/{action}", require_auth=True
        )
        return data


def fetch_remote_media_as_base64(media_url: str) -> Tuple[str, str]:
    """Download a remote media URL (e.g., timeline image) and return (mime_type, base64_data)."""
    parsed = urllib.parse.urlparse(media_url)
    if parsed.scheme not in ("http", "https"):
        raise MastodonAPIError("Invalid remote media URL scheme.", 400)

    req = urllib.request.Request(
        media_url,
        headers={"User-Agent": "FediCaption/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            raw = resp.read(20 * 1024 * 1024)  # Cap at 20MB
            b64 = base64.b64encode(raw).decode("ascii")
            return content_type, b64
    except Exception as exc:
        raise MastodonAPIError(f"Failed to fetch remote media: {exc}", 502) from exc
