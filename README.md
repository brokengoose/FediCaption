# FediCaption — Accessible Mastodon Client with AI Image & Video Alt-Text

<p align="left">
  <img src="static/logo.png" alt="FediCaption logo: a friendly grey elephant wearing dark sunglasses on a blue background" width="96" height="96" style="border-radius: 20px;" />
</p>

**FediCaption** is a cross-platform Mastodon / Fediverse client for **macOS** and **Linux** designed around accessibility for blind and low-vision users.

Whenever you attach an **image** or **video** to a post, FediCaption automatically scans the media using your choice of **local AI** (Ollama, LM Studio, llama.cpp) or **cloud AI** (Google Gemini, OpenAI, Anthropic Claude) and populates the Mastodon alt-text field with a clear, screen-reader-friendly description following W3C WAI and Perkins School for the Blind guidelines.

> *This project was generated with **Antigravity**.*

---

## Key Features

- **Automatic AI Image & Video Alt-Text Generation**:
  - **Images**: Scans JPEG, PNG, WebP, and GIF images on attach and fills in the alt-text field automatically.
  - **Videos**: Automatically extracts chronological keyframes across the video clip (and displays a visual keyframe strip) using HTML5 `<video>` + `<canvas>` (with optional server-side `ffmpeg` fallback), or sends native video bytes to multimodal models like Gemini, producing a cohesive description of the subjects, motion, actions over time, and on-screen captions.
  - **Standardized Prefix**: All AI-generated captions begin with `"AI generated description: "` and avoid redundant *"Image of..."* preambles while describing subject, setting, action, mood/lighting, and transcribing embedded text (OCR).
  - **Detail Levels & Context**: Choose between **Concise** (1–2 sentences), **Descriptive** (2–4 sentences), or **Detailed + OCR** (4–7 sentences), and optionally feed your draft post text as context so the AI uses accurate names and topics.
  - **Screen-Reader Preview ("Read Aloud")**: Built-in Web Speech synthesis button lets you listen to how a screen reader will read your alt-text before publishing.
  - **Pre-Post Accessibility Guard**: Warns or blocks publishing if an attached image or video is missing alt-text, with a one-click button to generate missing descriptions.
- **Pre-Post Racial & Gender Bias Review**:
  - Before publishing, checks your draft post text, content warning, media attachments (both visuals and alt-text), linked web pages, and referenced/replied-to messages for implicit or explicit racial and gender bias.
  - Publishes immediately if no bias is found; if bias is detected, pinpoints where it occurred and suggests less-biased alternate wording for the main post text.
  - Configurable under **Settings &rarr; 4. Bias Rules** (`Warn` by default, `Require correction`, or `Off`).
- **4 AI Provider Integrations with Dynamic Model Drop-Downs**:
  1. **Ollama (100% Local)**: Queries `/api/tags` at `http://localhost:11434` to populate a drop-down of installed vision models (`llava`, `llama3.2-vision`, `moondream`, `qwen2.5vl`, `gemma3`, `minicpm-v`).
  2. **Google Gemini API**: Queries `/v1beta/models` for available Gemini models (`gemini-2.5-flash`, `gemini-2.5-pro`, etc.) with both multi-frame and native inline video understanding.
  3. **OpenAI & OpenAI-Compatible APIs**: Queries `/models` from OpenAI (`gpt-4o-mini`, `gpt-4o`) or local servers like **LM Studio** (`http://localhost:1234/v1`), **vLLM**, **llama.cpp**, and **OpenRouter**.
  4. **Anthropic Claude API**: Queries `/v1/models` for available Claude vision models (`claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`).
- **Core Mastodon Client**:
  - **OAuth 2.0 & Access Token Login**: Connect to any Mastodon/Fediverse instance (`mastodon.social`, `fosstodon.org`, `hachyderm.io`, etc.) with client metadata linking directly to this GitHub repository.
  - **Dynamic Server Character & Media Limits**: Queries the connected Mastodon instance (`/api/v2/instance` / `/api/v1/instance`) for its actual maximum post length (`max_characters` / `max_toot_chars`), media attachment count, and alt-text limit.
  - **Timelines & User Profiles**: Browse Home, Local Server, Federated, Notifications, Bookmarks, Favorites, and My Posts; click any user (or your own upper-left profile card) to view their personal posts and **Subscribe / Unsubscribe** at the top of the timeline.
  - **Embedded Quote Posts (`RE: <URL>`)**: Posts beginning with `RE:` followed by a Mastodon post URL automatically resolve and embed the quoted original post in a visually distinct card above the reply.
  - **Edit Own Posts & Timestamp Permalinks**: Edit your own published posts (including updating media alt-text descriptions), and click any post's upper-right timestamp to open the original post in a new tab.
  - **Caption Uncaptioned Timeline Posts**: Includes a **"Describe with AI"** button on incoming timeline posts so blind and low-vision users can generate descriptions for posts where the original author forgot to add alt-text.
  - **WCAG AAA Accessible Interface & System Theme Detection**: Semantic ARIA landmarks, live screen-reader announcements (`aria-live`), keyboard shortcuts (`Alt+N`, `Alt+A`, `Cmd/Ctrl+Enter`), Dark, Light, and High-Contrast themes (automatically defaulting to your operating system's active mode), and automatic GitHub version update checks.

---

## Project Architecture

- [`server/app.py`](server/app.py): Local HTTP server, REST API endpoints, static file handler, and GitHub version update checker.
- [`server/ai_captioner.py`](server/ai_captioner.py): Multimodal AI alt-text generation engine, video keyframe extraction fallback, accessibility prompt builder, and live provider model discovery.
- [`server/bias_checker.py`](server/bias_checker.py): Pre-post racial and gender bias analysis across post text, attachments, fetched links, and referenced messages.
- [`server/mastodon_client.py`](server/mastodon_client.py): Mastodon OAuth 2.0 & REST API client (timelines, dynamic instance limits, media upload, post editing, user subscriptions, and quoted status resolution).
- [`server/config.py`](server/config.py): Cross-platform configuration persistence for macOS and Linux.
- [`static/index.html`](static/index.html), [`static/styles.css`](static/styles.css), [`static/app.js`](static/app.js): Accessible frontend interface and HTML5 `<video>` + `<canvas>` keyframe extractor.
- [`tests/test_fedicaption.py`](tests/test_fedicaption.py): Unit and integration test suite.

---

## Quick Start (macOS & Linux)

FediCaption is built on Python 3's standard library + modern HTML5/JS, requiring **zero external Python packages or native compilers** to run on macOS or Linux.

### 1. Run Directly
```bash
chmod +x fedicaption FediCaption.command
./fedicaption
```
Or with Python 3 directly:
```bash
python3 -m server.app
```
- On **macOS**, you can also double-click `FediCaption.command` in Finder.
- On **Linux**, you can copy `fedicaption.desktop` to `~/.local/share/applications/` for your desktop application launcher.

### 2. Optional CLI Flags
```bash
./fedicaption --host 127.0.0.1 --port 8765
./fedicaption --no-browser
```

---

## Setting Up Local AI (Ollama)

To generate image and video captions 100% locally without any API keys:
1. Start Ollama (`ollama serve` or open the Ollama app on macOS).
2. Pull a vision model:
   ```bash
   ollama pull llava
   # or
   ollama pull llama3.2-vision
   # or lightweight:
   ollama pull moondream
   ```
3. In FediCaption, open **Settings & API Keys** &rarr; **AI Vision & Alt-Text Providers** &rarr; click **Detect Models** under Ollama.

---

## Configuration Storage Paths

FediCaption stores your settings and credentials with `0600` user-only permissions in the standard OS configuration directory:
- **macOS**: `~/Library/Application Support/FediCaption/config.json`
- **Linux**: `${XDG_CONFIG_HOME:-~/.config}/fedicaption/config.json`
- **Custom Override**: Set `FEDICAPTION_CONFIG_DIR=/path/to/dir`

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```

---

## Acknowledgements

This project was generated with **Antigravity**.

