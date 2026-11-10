"""Unit and integration test suite for FediCaption."""

import base64
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from server.ai_captioner import (
    MASTODON_ALT_TEXT_MAX_CHARS,
    build_accessibility_prompt,
    clean_caption_output,
    generate_alt_text,
    list_ollama_models,
)
from server.app import create_server
from server.config import load_config, save_config
from server.mastodon_client import (
    MastodonClient,
    _encode_multipart_formdata,
    extract_instance_limits,
    normalize_instance_url,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._bytes = json.dumps(payload).encode("utf-8")
        self._status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self, *_args):
        return self._bytes

    def getcode(self):
        return self._status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FediCaptionTestSuite(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        os.environ["FEDICAPTION_CONFIG_DIR"] = self.temp_dir.name
        os.environ["FEDICAPTION_QUIET"] = "1"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_config_load_and_save(self):
        cfg = load_config()
        self.assertEqual(cfg["ai"]["provider"], "ollama")
        self.assertTrue(cfg["ai"]["auto_generate"])

        updated = save_config(
            {
                "ai": {
                    "provider": "gemini",
                    "gemini": {"api_key": "test-gemini-key", "model": "gemini-2.5-flash"},
                },
                "ui": {"theme": "high-contrast"},
            }
        )
        self.assertEqual(updated["ai"]["provider"], "gemini")
        self.assertEqual(updated["ai"]["gemini"]["api_key"], "test-gemini-key")
        self.assertEqual(updated["ui"]["theme"], "high-contrast")
        # Ensure untouched defaults are preserved
        self.assertEqual(updated["ai"]["ollama"]["base_url"], "http://localhost:11434")

    def test_accessibility_prompt_image_and_video(self):
        img_prompt = build_accessibility_prompt(
            media_type="image",
            detail_level="descriptive",
            post_context="Sunset at Acadia National Park",
            language="English",
        )
        self.assertIn("describing an IMAGE for blind and low-vision Mastodon users", img_prompt)
        self.assertIn("Sunset at Acadia National Park", img_prompt)
        self.assertIn("Do NOT begin with 'Image of'", img_prompt)

        vid_prompt = build_accessibility_prompt(
            media_type="video",
            detail_level="detailed",
            frame_count=4,
        )
        self.assertIn("describing a VIDEO for blind and low-vision Mastodon users", vid_prompt)
        self.assertIn("extracted keyframes in chronological order", vid_prompt)

    def test_clean_caption_output(self):
        raw = '"Alt-text: A golden retriever leaping over a mossy log in a sunlit forest."'
        cleaned = clean_caption_output(raw)
        self.assertEqual(
            cleaned,
            "AI generated description: A golden retriever leaping over a mossy log in a sunlit forest.",
        )

        # Ensure prefix is not duplicated if already present
        already_prefixed = "AI generated description: A red bicycle parked against a brick wall."
        self.assertEqual(
            clean_caption_output(already_prefixed),
            "AI generated description: A red bicycle parked against a brick wall.",
        )

        long_text = "word " * 400
        truncated = clean_caption_output(long_text)
        self.assertTrue(truncated.startswith("AI generated description: "))
        self.assertLessEqual(len(truncated), MASTODON_ALT_TEXT_MAX_CHARS)

    @patch("urllib.request.urlopen")
    def test_ollama_caption_generation_and_model_detection(self, mock_urlopen):
        # 1. Model list
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "models": [
                    {"name": "llama3.2:latest", "details": {"families": ["llama"]}},
                    {"name": "llava:latest", "details": {"families": ["llama", "clip"]}},
                ]
            }
        )
        info = list_ollama_models("http://localhost:11434")
        self.assertTrue(info["reachable"])
        self.assertEqual(info["models"][0]["name"], "llava:latest")
        self.assertTrue(info["models"][0]["is_vision_likely"])

        # 2. Caption generation
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "message": {
                    "content": "A close-up of a steaming ceramic mug of coffee on a wooden desk beside a Braille display."
                }
            }
        )
        res = generate_alt_text(
            ai_config={
                "provider": "ollama",
                "ollama": {"base_url": "http://localhost:11434", "model": "llava:latest"},
            },
            media_type="image",
            images_b64=["data:image/jpeg;base64,dGVzdA=="],
        )
        self.assertEqual(res["provider"], "ollama")
        self.assertIn("Braille display", res["caption"])

    @patch("urllib.request.urlopen")
    def test_gemini_video_caption_generation(self, mock_urlopen):
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "A cyclist rides along a coastal bike path at dusk while waves crash against the rocks below."
                                }
                            ]
                        }
                    }
                ]
            }
        )
        res = generate_alt_text(
            ai_config={
                "provider": "gemini",
                "gemini": {"api_key": "fake-key", "model": "gemini-2.5-flash"},
            },
            media_type="video",
            images_b64=["data:image/jpeg;base64,ZnJhbWUx", "data:image/jpeg;base64,ZnJhbWUy"],
            raw_media_b64="data:video/mp4;base64,dmlkZW9ieXRlcw==",
            raw_media_mime="video/mp4",
        )
        self.assertEqual(res["provider"], "gemini")
        self.assertIn("cyclist rides", res["caption"])

    @patch("urllib.request.urlopen")
    def test_openai_and_anthropic_caption_generation(self, mock_urlopen):
        # OpenAI
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "An orange tabby cat curled up asleep on an open laptop keyboard."
                        }
                    }
                ]
            }
        )
        res_oai = generate_alt_text(
            ai_config={
                "provider": "openai",
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-test",
                    "model": "gpt-4o-mini",
                },
            },
            media_type="image",
            images_b64=["data:image/png;base64,dGVzdA=="],
        )
        self.assertIn("orange tabby cat", res_oai["caption"])

        # Anthropic
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "Terminal screenshot showing green test output reading 'Ran 7 tests in 0.05s OK'.",
                    }
                ]
            }
        )
        res_ant = generate_alt_text(
            ai_config={
                "provider": "anthropic",
                "anthropic": {
                    "api_key": "sk-ant-test",
                    "model": "claude-3-7-sonnet-latest",
                },
            },
            media_type="image",
            images_b64=["data:image/png;base64,dGVzdA=="],
        )
        self.assertIn("Terminal screenshot", res_ant["caption"])

    @patch("urllib.request.urlopen")
    def test_mastodon_client_media_upload_and_post(self, mock_urlopen):
        self.assertEqual(
            normalize_instance_url("fosstodon.org/"), "https://fosstodon.org"
        )

        body, content_type = _encode_multipart_formdata(
            fields={"description": "Accessible alt text"},
            files=[("file", "photo.jpg", "image/jpeg", b"fake-jpeg-bytes")],
        )
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b"Accessible alt text", body)
        self.assertIn(b"fake-jpeg-bytes", body)

        client = MastodonClient("https://mastodon.social", access_token="tok_123")
        mock_urlopen.return_value = _FakeHTTPResponse(
            {"client_id": "cid_1", "client_secret": "csec_1"},
            status=200,
        )
        client.register_oauth_app("http://127.0.0.1:8765/api/mastodon/oauth/callback")
        app_req = mock_urlopen.call_args[0][0]
        app_payload = json.loads(app_req.data.decode("utf-8"))
        self.assertEqual(
            app_payload["website"], "https://github.com/brokengoose/FediCaption"
        )

        mock_urlopen.return_value = _FakeHTTPResponse(
            {"id": "media_99", "type": "image", "description": "Accessible alt text"},
            status=200,
        )
        media = client.upload_media(
            file_bytes=b"fake-jpeg-bytes",
            filename="photo.jpg",
            mime_type="image/jpeg",
            description="Accessible alt text",
        )
        self.assertEqual(media["id"], "media_99")

        mock_urlopen.return_value = _FakeHTTPResponse(
            {"id": "status_42", "content": "<p>Hello Fediverse!</p>"},
            status=200,
        )
        posted = client.post_status(
            status_text="Hello Fediverse!",
            media_ids=["media_99"],
            visibility="public",
        )
        self.assertEqual(posted["id"], "status_42")

    def _invoke_handler(self, method: str, path: str, body: dict = None) -> tuple:
        """Invoke FediCaptionHandler in-memory without requiring TCP socket connect."""
        from server.app import FediCaptionHandler

        body_bytes = json.dumps(body).encode("utf-8") if body is not None else b""
        raw_req = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:8765\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"\r\n"
        ).encode("utf-8") + body_bytes

        rfile = io.BytesIO(raw_req)
        wfile = io.BytesIO()

        class _DummySocket:
            def makefile(self, mode, *args, **kwargs):
                return rfile if "r" in mode else wfile

            def sendall(self, data):
                wfile.write(data)

            def close(self):
                pass

        dummy_server = MagicMock()
        FediCaptionHandler(_DummySocket(), ("127.0.0.1", 55555), dummy_server)
        raw_resp = wfile.getvalue().decode("utf-8", errors="replace")
        header_part, _, payload_part = raw_resp.partition("\r\n\r\n")
        status_line = header_part.splitlines()[0]
        status_code = int(status_line.split(" ")[1])
        return status_code, payload_part

    @patch("urllib.request.urlopen")
    def test_http_server_endpoints(self, mock_urlopen):
        # 1. Health check
        code, body = self._invoke_handler("GET", "/api/health")
        self.assertEqual(code, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])

        # 2. Index HTML static serving
        code, html = self._invoke_handler("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("FediCaption", html)
        self.assertIn("Alt-Text", html)

        # 3. Config update via POST /api/config
        code, cfg_body = self._invoke_handler(
            "POST", "/api/config", {"ai": {"detail_level": "detailed"}}
        )
        self.assertEqual(code, 200)
        cfg_data = json.loads(cfg_body)
        self.assertEqual(cfg_data["config"]["ai"]["detail_level"], "detailed")

        # 4. AI caption endpoint via POST /api/ai/caption
        mock_urlopen.return_value = _FakeHTTPResponse(
            {"message": {"content": "A person walking with a white cane along a tree-lined sidewalk."}}
        )
        code, cap_body = self._invoke_handler(
            "POST",
            "/api/ai/caption",
            {
                "media_type": "image",
                "images_b64": ["data:image/jpeg;base64,dGVzdA=="],
            },
        )
        self.assertEqual(code, 200)
        cap_data = json.loads(cap_body)
        self.assertTrue(cap_data["ok"])
        self.assertIn("white cane", cap_data["caption"])


        # 5. Mastodon instance character limit endpoint via GET /api/mastodon/instance
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "domain": "tech.lgbt",
                "configuration": {
                    "statuses": {
                        "max_characters": 2500,
                        "max_media_attachments": 6,
                    },
                    "media_attachments": {
                        "description_limit": 2000,
                    },
                },
            }
        )
        code, inst_body = self._invoke_handler(
            "GET", "/api/mastodon/instance?instance_url=https://tech.lgbt"
        )
        self.assertEqual(code, 200)
        inst_data = json.loads(inst_body)
        self.assertEqual(inst_data["limits"]["max_post_chars"], 2500)
        self.assertEqual(inst_data["limits"]["max_media_attachments"], 6)
        self.assertEqual(inst_data["limits"]["alt_text_limit"], 2000)
        self.assertEqual(
            inst_data["config"]["mastodon"]["max_post_chars"], 2500
        )

    def test_extract_instance_limits_variants(self):
        # Standard Mastodon v2/v1 configuration
        std_limits = extract_instance_limits(
            {
                "configuration": {
                    "statuses": {"max_characters": 5000, "max_media_attachments": 8},
                    "media_attachments": {"description_limit": 1800},
                }
            }
        )
        self.assertEqual(std_limits["max_post_chars"], 5000)
        self.assertEqual(std_limits["max_media_attachments"], 8)
        self.assertEqual(std_limits["alt_text_limit"], 1800)

        # Glitch-soc / Hometown / Pleroma top-level max_toot_chars
        glitch_limits = extract_instance_limits({"max_toot_chars": 11000})
        self.assertEqual(glitch_limits["max_post_chars"], 11000)
        self.assertEqual(glitch_limits["max_media_attachments"], 4)
        self.assertEqual(glitch_limits["alt_text_limit"], 1500)

        # Fallback to defaults when missing
        default_limits = extract_instance_limits({})
        self.assertEqual(default_limits["max_post_chars"], 500)
        self.assertEqual(default_limits["max_media_attachments"], 4)
        self.assertEqual(default_limits["alt_text_limit"], 1500)

    @patch("urllib.request.urlopen")
    def test_user_personal_posts_and_subscription(self, mock_urlopen):
        save_config(
            {
                "mastodon": {
                    "instance_url": "https://mastodon.social",
                    "access_token": "tok_sub_test",
                }
            }
        )

        # 1. Fetch a clicked user's personal posts via GET /api/mastodon/timeline?type=account&account_id=777
        mock_urlopen.return_value = _FakeHTTPResponse(
            [{"id": "st_1", "content": "<p>User personal post</p>"}]
        )
        code, tl_body = self._invoke_handler(
            "GET", "/api/mastodon/timeline?type=account&account_id=777"
        )
        self.assertEqual(code, 200)
        tl_data = json.loads(tl_body)
        self.assertEqual(len(tl_data["statuses"]), 1)
        self.assertEqual(tl_data["statuses"][0]["id"], "st_1")

        # 2. Fetch user profile & relationship via GET /api/mastodon/account?id=777
        mock_urlopen.side_effect = [
            _FakeHTTPResponse(
                {"id": "777", "username": "alice", "acct": "alice", "display_name": "Alice"}
            ),
            _FakeHTTPResponse(
                [{"id": "777", "following": False, "notifying": False}]
            ),
        ]
        code, acct_body = self._invoke_handler("GET", "/api/mastodon/account?id=777")
        self.assertEqual(code, 200)
        acct_data = json.loads(acct_body)
        self.assertEqual(acct_data["account"]["acct"], "alice")
        self.assertFalse(acct_data["relationship"]["following"])

        # 3. Subscribe (follow) user via POST /api/mastodon/account/subscribe
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = _FakeHTTPResponse(
            {"id": "777", "following": True, "notifying": False}
        )
        code, sub_body = self._invoke_handler(
            "POST",
            "/api/mastodon/account/subscribe",
            {"account_id": "777", "action": "follow"},
        )
        self.assertEqual(code, 200)
        sub_data = json.loads(sub_body)
        self.assertTrue(sub_data["relationship"]["following"])

    @patch("urllib.request.urlopen")
    def test_edit_own_status(self, mock_urlopen):
        save_config(
            {
                "mastodon": {
                    "instance_url": "https://mastodon.social",
                    "access_token": "tok_edit_test",
                }
            }
        )

        # 1. Fetch raw status source via GET /api/mastodon/status/source?id=999
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "id": "999",
                "text": "Original post text",
                "spoiler_text": "",
            }
        )
        code, src_body = self._invoke_handler(
            "GET", "/api/mastodon/status/source?id=999"
        )
        self.assertEqual(code, 200)
        src_data = json.loads(src_body)
        self.assertEqual(src_data["source"]["text"], "Original post text")

        # 2. Edit status via PUT /api/mastodon/status
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "id": "999",
                "content": "<p>Updated post text</p>",
                "edited_at": "2026-11-09T22:00:00.000Z",
            }
        )
        code, edit_body = self._invoke_handler(
            "PUT",
            "/api/mastodon/status",
            {
                "status_id": "999",
                "status": "Updated post text",
                "media_ids": ["med_1"],
                "media_attributes": [
                    {
                        "id": "med_1",
                        "description": "AI generated description: A cat sleeping on a windowsill.",
                    }
                ],
                "spoiler_text": "",
                "sensitive": False,
                "language": "en",
            },
        )
        self.assertEqual(code, 200)
        edit_data = json.loads(edit_body)
        self.assertTrue(edit_data["ok"])
        self.assertEqual(edit_data["status"]["id"], "999")
        self.assertIn("Updated post text", edit_data["status"]["content"])

    @patch("urllib.request.urlopen")
    def test_provider_models_listing(self, mock_urlopen):
        # 1. Gemini models query via POST /api/ai/models
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "displayName": "Gemini 2.5 Flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "displayName": "Text Embedding 004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            }
        )
        code, gem_body = self._invoke_handler(
            "POST",
            "/api/ai/models",
            {"provider": "gemini", "api_key": "AIzaTestKey"},
        )
        self.assertEqual(code, 200)
        gem_data = json.loads(gem_body)
        self.assertTrue(gem_data["reachable"])
        self.assertEqual(len(gem_data["models"]), 1)
        self.assertEqual(gem_data["models"][0]["name"], "gemini-2.5-flash")

        # 2. OpenAI / compatible models query via POST /api/ai/models
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "data": [
                    {"id": "gpt-4o-mini"},
                    {"id": "gpt-4o"},
                    {"id": "whisper-1"},
                ]
            }
        )
        code, oai_body = self._invoke_handler(
            "POST",
            "/api/ai/models",
            {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-test",
            },
        )
        self.assertEqual(code, 200)
        oai_data = json.loads(oai_body)
        self.assertTrue(oai_data["reachable"])
        oai_names = [m["name"] for m in oai_data["models"]]
        self.assertIn("gpt-4o-mini", oai_names)
        self.assertIn("gpt-4o", oai_names)
        self.assertNotIn("whisper-1", oai_names)

        # 3. Anthropic models query via POST /api/ai/models
        mock_urlopen.return_value = _FakeHTTPResponse(
            {
                "data": [
                    {
                        "id": "claude-3-7-sonnet-20250219",
                        "display_name": "Claude 3.7 Sonnet",
                    }
                ]
            }
        )
        code, ant_body = self._invoke_handler(
            "POST",
            "/api/ai/models",
            {"provider": "anthropic", "api_key": "sk-ant-test"},
        )
        self.assertEqual(code, 200)
        ant_data = json.loads(ant_body)
        self.assertTrue(ant_data["reachable"])
        ant_names = [m["name"] for m in ant_data["models"]]
        self.assertIn("claude-3-7-sonnet-20250219", ant_names)


if __name__ == "__main__":
    unittest.main()
