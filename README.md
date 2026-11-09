# FediCaption — Accessible Mastodon Client with AI Image & Video Alt-Text

**FediCaption** is a cross-platform Mastodon / Fediverse client for **macOS** and **Linux** designed around accessibility for blind and low-vision users.

Whenever you attach an **image** or **video** to a post, FediCaption automatically scans the media using your choice of **local AI** (Ollama, LM Studio, llama.cpp) or **cloud AI** (Google Gemini, OpenAI, Anthropic Claude) and populates the Mastodon alt-text field with a clear, screen-reader-friendly description following W3C WAI and Perkins School for the Blind guidelines.

> *This project was generated with **Antigravity**.*

---

## Key Features

- **Automatic AI Image & Video Alt-Text Generation**:
  - **Images**: Scans JPEG, PNG, WebP, and GIF images on attach and fills in the alt-text field automatically.
  - **Videos**: Automatically extracts chronological keyframes across the video clip (and displays a visual keyframe strip) using HTML5 `<video>` + `<canvas>` (with optional server-side `ffmpeg` fallback), or sends native video bytes to multimodal models like Gemini, producing a cohesive description of the subjects, motion, actions over time, and on-screen captions.
  - **Blind-First Prompt Engineering**: Avoids redundant *"Image of..."* prefixes, describes subject, setting, action, mood/lighting, and transcribes embedded text (OCR) inside quotation marks.
  - **Detail Levels & Context**: Choose between **Concise** (1–2 sentences), **Descriptive** (2–4 sentences), or **Detailed + OCR** (4–7 sentences), and optionally feed your draft post text as context so the AI uses accurate names and topics.
  - **Screen-Reader Preview ("Read Aloud")**: Built-in Web Speech synthesis button lets you listen to how a screen reader will read your alt-text before publishing.
  - **Pre-Post Accessibility Guard**: Warns or blocks publishing if an attached image or video is missing alt-text, with a one-click button to generate missing descriptions.
- **4 AI Provider Integrations (Local & Remote)**:
  1. **Ollama (100% Local)**: Auto-detects installed vision models (`llava`, `llama3.2-vision`, `moondream`, `qwen2.5vl`, `gemma3`, `minicpm-v`) at `http://localhost:11434`.
  2. **Google Gemini API**: Supports `gemini-2.5-flash` and `gemini-2.5-pro` with both multi-frame and native inline video understanding.
  3. **OpenAI & OpenAI-Compatible APIs**: Works with OpenAI (`gpt-4o-mini`, `gpt-4o`) as well as local servers like **LM Studio** (`http://localhost:1234/v1`), **vLLM**, **llama.cpp**, and **OpenRouter**.
  4. **Anthropic Claude API**: Supports `claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`, and `claude-3-5-haiku-latest`.
- **Core Mastodon Client**:
  - **OAuth 2.0 & Access Token Login**: Connect to any Mastodon/Fediverse instance (`mastodon.social`, `fosstodon.org`, `hachyderm.io`, etc.).
  - **Timelines**: Home, Local Server, Federated, Notifications, Bookmarks, Favorites, and My Posts.
  - **Caption Uncaptioned Timeline Posts**: Includes a **"Describe with AI"** button on incoming timeline posts so blind and low-vision users can generate descriptions for posts where the original author forgot to add alt-text.
  - **Full Interactions**: Reply, Boost, Favorite, Bookmark, Delete, Thread Conversation view, Content Warnings (CW), and Visibility controls.
  - **WCAG AAA Accessible Interface**: Semantic ARIA landmarks, live screen-reader announcements (`aria-live`), keyboard shortcuts (`Alt+N`, `Alt+A`, `Cmd/Ctrl+Enter`), and Dark, Light, and High-Contrast themes.

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

