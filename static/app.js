/**
 * FediCaption — Accessible Mastodon Client with AI Image & Video Alt-Text
 */

const state = {
  config: null,
  currentTimeline: "home",
  previousTimeline: "home",
  selectedUser: null,
  selectedUserRelationship: null,
  statuses: [],
  notifications: [],
  attachments: [], // Array of { id, file, mediaType, mimeType, previewUrl, fileB64, aiFramesB64, altText, status, providerUsed, errorMsg, detailLevel, existingMediaId, remoteUrl }
  replyToStatus: null,
  editingStatus: null,
  cwEnabled: false,
  isPublishing: false,
};

let attachmentIdCounter = 1;

/* ==========================================================================
   Accessibility & Screen Reader Utilities
   ========================================================================== */

function announce(message, assertive = false) {
  const el = document.getElementById(
    assertive ? "sr-alert-announcer" : "sr-announcer"
  );
  if (!el) return;
  el.textContent = "";
  setTimeout(() => {
    el.textContent = message;
  }, 40);
}

function speakTextAloud(text) {
  if (!("speechSynthesis" in window)) {
    alert("Speech synthesis is not supported in this browser.");
    return;
  }
  window.speechSynthesis.cancel();
  if (!text || !text.trim()) {
    announce("Alt-text field is currently empty.");
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text.trim());
  window.speechSynthesis.speak(utterance);
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function stripHtmlTags(html) {
  const tmp = document.createElement("div");
  tmp.innerHTML = html || "";
  return tmp.textContent || tmp.innerText || "";
}

function formatRelativeTime(isoDate) {
  if (!isoDate) return "";
  const date = new Date(isoDate);
  const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (diffSec < 60) return `${Math.max(1, diffSec)}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return date.toLocaleDateString();
}

function getMaxPostChars() {
  const val = Number(state.config?.mastodon?.max_post_chars);
  return Number.isFinite(val) && val > 0 ? val : 500;
}

function getMaxMediaAttachments() {
  const val = Number(state.config?.mastodon?.max_media_attachments);
  return Number.isFinite(val) && val > 0 ? val : 4;
}

function getAltTextLimit() {
  const val = Number(state.config?.mastodon?.alt_text_limit);
  return Number.isFinite(val) && val > 0 ? val : 1500;
}

/* ==========================================================================
   API Helper
   ========================================================================== */

async function apiRequest(path, options = {}) {
  const fetchOpts = {
    method: options.method || "GET",
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  };
  const resp = await fetch(path, fetchOpts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data.ok === false) {
    throw new Error(data.error || `Request failed with status ${resp.status}`);
  }
  return data;
}

/* ==========================================================================
   Image Downscaling & Video Keyframe Extraction (Pure HTML5 Canvas)
   ========================================================================== */

function readFileAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

async function prepareImageForAI(file, maxDimension = 1280) {
  const dataUrl = await readFileAsDataURL(file);
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      if (width <= maxDimension && height <= maxDimension) {
        resolve(dataUrl);
        return;
      }
      const scale = maxDimension / Math.max(width, height);
      width = Math.round(width * scale);
      height = Math.round(height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);
      resolve(canvas.toDataURL("image/jpeg", 0.86));
    };
    img.onerror = () => resolve(dataUrl);
    img.src = dataUrl;
  });
}

/**
 * Extracts chronological JPEG keyframes from a video file using an offscreen
 * HTML5 <video> and <canvas>, so both local vision models (Ollama LLaVA / Llama 3.2 Vision)
 * and cloud models can inspect the full sequence of events in a video!
 */
async function extractVideoKeyframes(file, frameCount = 4, maxDimension = 960) {
  return new Promise((resolve) => {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.playsInline = true;
    video.src = objectUrl;

    const cleanup = () => {
      try {
        URL.revokeObjectURL(objectUrl);
      } catch (_) {}
    };

    const timeoutId = setTimeout(() => {
      cleanup();
      resolve([]);
    }, 15000);

    video.onloadedmetadata = async () => {
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 4.0;
      const positions = [];
      for (let i = 0; i < frameCount; i++) {
        const ratio = (i + 0.5) / frameCount;
        positions.push(Math.min(Math.max(0.05, duration * ratio), Math.max(0.05, duration - 0.05)));
      }

      const frames = [];
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      for (const timestamp of positions) {
        // eslint-disable-next-line no-await-in-loop
        const frameDataUrl = await new Promise((seekResolve) => {
          const onSeeked = () => {
            video.removeEventListener("seeked", onSeeked);
            const vw = video.videoWidth || 640;
            const vh = video.videoHeight || 360;
            const scale = Math.min(1, maxDimension / Math.max(vw, vh));
            canvas.width = Math.max(1, Math.round(vw * scale));
            canvas.height = Math.max(1, Math.round(vh * scale));
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            seekResolve(canvas.toDataURL("image/jpeg", 0.84));
          };
          video.addEventListener("seeked", onSeeked, { once: true });
          try {
            video.currentTime = timestamp;
          } catch (_) {
            video.removeEventListener("seeked", onSeeked);
            seekResolve(null);
          }
        });
        if (frameDataUrl) {
          frames.push(frameDataUrl);
        }
      }

      clearTimeout(timeoutId);
      cleanup();
      resolve(frames);
    };

    video.onerror = () => {
      clearTimeout(timeoutId);
      cleanup();
      resolve([]);
    };
  });
}

/* ==========================================================================
   Media Attachments & Automatic AI Alt-Text Studio
   ========================================================================== */

async function handleFilesSelected(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  const maxMedia = getMaxMediaAttachments();
  for (const file of files) {
    if (state.attachments.length >= maxMedia) {
      showComposerFeedback(`This server allows up to ${maxMedia} media attachments per post.`, "warning");
      break;
    }

    const isVideo = file.type.startsWith("video/");
    const isImage = file.type.startsWith("image/");
    if (!isVideo && !isImage) {
      showComposerFeedback(`Unsupported file type: ${file.name}. Please attach an image or video.`, "error");
      continue;
    }

    const id = `att-${attachmentIdCounter++}`;
    const previewUrl = URL.createObjectURL(file);
    const defaultDetail = state.config?.ai?.detail_level || "descriptive";

    const attachment = {
      id,
      file,
      filename: file.name || (isVideo ? "video.mp4" : "image.jpg"),
      mediaType: isVideo ? "video" : "image",
      mimeType: file.type || (isVideo ? "video/mp4" : "image/jpeg"),
      previewUrl,
      fileB64: null,
      aiFramesB64: [],
      altText: "",
      status: "preparing", // "preparing" | "scanning" | "ready" | "missing" | "error"
      providerUsed: "",
      errorMsg: "",
      detailLevel: defaultDetail,
    };

    state.attachments.push(attachment);
    renderAttachments();

    // Prepare base64 & keyframes asynchronously
    prepareAttachmentMedia(attachment);
  }
}

async function prepareAttachmentMedia(attachment) {
  try {
    attachment.fileB64 = await readFileAsDataURL(attachment.file);

    if (attachment.mediaType === "video") {
      const kfCount = Number(state.config?.ui?.video_keyframes_count || 4);
      announce(`Extracting ${kfCount} keyframes from video ${attachment.filename} for AI captioning...`);
      attachment.aiFramesB64 = await extractVideoKeyframes(attachment.file, kfCount);
    } else {
      const resized = await prepareImageForAI(attachment.file);
      attachment.aiFramesB64 = [resized];
    }

    renderAttachments();

    const autoGenerate = document.getElementById("quick-auto-scan")?.checked ?? true;
    if (autoGenerate) {
      await generateAltTextForAttachment(attachment.id);
    } else {
      attachment.status = "missing";
      renderAttachments();
      announce(`${attachment.mediaType} attached. Press Scan with AI or enter alt-text.`);
    }
  } catch (err) {
    attachment.status = "error";
    attachment.errorMsg = err.message || "Failed to process media file.";
    renderAttachments();
  }
}

async function generateAltTextForAttachment(attachmentId) {
  const att = state.attachments.find((a) => a.id === attachmentId);
  if (!att) return;

  att.status = "scanning";
  att.errorMsg = "";
  renderAttachments();

  const provider = document.getElementById("quick-provider-select")?.value || state.config?.ai?.provider || "ollama";
  announce(`Scanning ${att.mediaType} ${att.filename} with ${provider} AI to generate alt-text...`);

  try {
    const postContext = document.getElementById("composer-textarea")?.value || "";
    const payload = {
      media_type: att.mediaType,
      images_b64: att.aiFramesB64,
      raw_media_b64: att.mediaType === "video" ? att.fileB64 : undefined,
      raw_media_mime: att.mimeType,
      remote_url:
        (!att.aiFramesB64 || att.aiFramesB64.length === 0) && att.remoteUrl
          ? att.remoteUrl
          : undefined,
      post_context: postContext,
      detail_level: att.detailLevel,
      ai_override: {
        provider,
      },
    };

    const result = await apiRequest("/api/ai/caption", {
      method: "POST",
      body: payload,
    });

    att.altText = result.caption || "";
    att.status = att.altText ? "ready" : "missing";
    att.providerUsed = `${result.provider} (${result.model})`;
    renderAttachments();

    announce(
      `AI alt-text generated for ${att.filename}: ${att.altText}`,
      false
    );
  } catch (err) {
    att.status = "error";
    att.errorMsg = err.message || "AI caption generation failed.";
    renderAttachments();
    announce(`AI alt-text error for ${att.filename}: ${att.errorMsg}`, true);
  }
}

async function scanAllAttachmentsWithAI() {
  for (const att of state.attachments) {
    // eslint-disable-next-line no-await-in-loop
    await generateAltTextForAttachment(att.id);
  }
}

function removeAttachment(attachmentId) {
  const idx = state.attachments.findIndex((a) => a.id === attachmentId);
  if (idx === -1) return;
  const [removed] = state.attachments.splice(idx, 1);
  if (removed?.previewUrl) {
    try {
      URL.revokeObjectURL(removed.previewUrl);
    } catch (_) {}
  }
  renderAttachments();
  announce(`Removed attachment ${removed.filename}.`);
}

function renderAttachments() {
  const container = document.getElementById("media-attachments-list");
  const scanAllBtn = document.getElementById("scan-all-media-btn");
  if (!container) return;

  if (scanAllBtn) {
    scanAllBtn.classList.toggle("hidden", state.attachments.length === 0);
  }

  if (state.attachments.length === 0) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = state.attachments
    .map((att, index) => {
      const isScanning = att.status === "scanning" || att.status === "preparing";
      const hasAlt = Boolean(att.altText && att.altText.trim().length > 0);
      let statusLabel = "Missing Alt-Text";
      let statusClass = "missing";

      if (att.status === "preparing") {
        statusLabel =
          att.mediaType === "video"
            ? "Extracting Video Keyframes..."
            : "Preparing Image...";
        statusClass = "scanning";
      } else if (att.status === "scanning") {
        statusLabel = "Scanning with AI...";
        statusClass = "scanning";
      } else if (att.status === "error") {
        statusLabel = "AI Scan Error";
        statusClass = "error";
      } else if (hasAlt) {
        statusLabel = att.providerUsed
          ? `AI Ready · ${att.providerUsed}`
          : "Alt-Text Ready";
        statusClass = "ready";
      }

      const cardStateClass = isScanning
        ? "scanning"
        : !hasAlt
        ? "missing-alt"
        : "";

      const mediaPreviewHtml =
        att.mediaType === "video"
          ? `<video src="${escapeHtml(att.previewUrl)}" controls muted playsinline aria-label="Attached video preview ${escapeHtml(att.filename)}"></video>`
          : `<img src="${escapeHtml(att.previewUrl)}" alt="${escapeHtml(att.altText || `Attached image ${att.filename}`)}" />`;

      const keyframesHtml =
        att.mediaType === "video" && att.aiFramesB64.length > 0
          ? `<div class="keyframes-strip-wrap">
               <span class="keyframes-strip-label">Extracted AI Keyframes (${att.aiFramesB64.length}):</span>
               <div class="keyframes-strip">
                 ${att.aiFramesB64
                   .map(
                     (frame, idx) =>
                       `<img src="${frame}" alt="Video keyframe ${idx + 1}" />`
                   )
                   .join("")}
               </div>
             </div>`
          : "";

      const errorHtml = att.errorMsg
        ? `<div class="composer-feedback error" style="margin-top:4px;padding:6px 10px;font-size:0.8rem;">
             ${escapeHtml(att.errorMsg)}
           </div>`
        : "";

      return `
        <div class="media-attachment-card ${cardStateClass}" data-att-id="${att.id}">
          <div class="media-preview-col">
            <div class="media-preview-frame">
              <span class="media-type-badge">${escapeHtml(att.mediaType)} #${index + 1}</span>
              ${mediaPreviewHtml}
            </div>
            ${keyframesHtml}
          </div>

          <div class="media-caption-col">
            <div class="media-caption-header">
              <label class="media-caption-title" for="alt-input-${att.id}">
                <span>Alt-Text for Blind Users (${escapeHtml(att.filename)})</span>
              </label>
              <span class="alt-status-pill ${statusClass}" role="status">
                ${isScanning ? '<span class="spinner" style="width:11px;height:11px;border-width:2px;margin-right:4px;"></span>' : ""}
                ${escapeHtml(statusLabel)}
              </span>
            </div>

            <textarea
              id="alt-input-${att.id}"
              class="alt-textarea"
              maxlength="${getAltTextLimit()}"
              placeholder="${
                isScanning
                  ? "AI is scanning this media and generating a blind-accessible description..."
                  : "Describe this image or video for screen-reader users (or click Regenerate with AI)..."
              }"
              aria-label="Alt-text description for ${escapeHtml(att.filename)}"
              data-att-input="${att.id}"
            >${escapeHtml(att.altText)}</textarea>

            ${errorHtml}

            <div class="media-caption-footer">
              <div class="media-caption-actions">
                <button
                  type="button"
                  class="btn btn-sm btn-accent"
                  data-action="rescan-att"
                  data-att-id="${att.id}"
                  ${isScanning ? "disabled" : ""}
                >
                  &#10024; ${hasAlt ? "Regenerate AI Alt-Text" : "Generate AI Alt-Text"}
                </button>

                <label for="detail-${att.id}" class="sr-only">Detail Level</label>
                <select
                  id="detail-${att.id}"
                  class="select-compact"
                  data-action="change-att-detail"
                  data-att-id="${att.id}"
                  title="Caption detail level"
                >
                  <option value="concise" ${att.detailLevel === "concise" ? "selected" : ""}>Concise</option>
                  <option value="descriptive" ${att.detailLevel === "descriptive" ? "selected" : ""}>Descriptive</option>
                  <option value="detailed" ${att.detailLevel === "detailed" ? "selected" : ""}>Detailed + OCR</option>
                </select>

                <button
                  type="button"
                  class="btn btn-sm btn-ghost"
                  data-action="speak-alt"
                  data-att-id="${att.id}"
                  title="Listen to how a screen reader reads this alt-text"
                >
                  &#128266; Read Aloud
                </button>

                <button
                  type="button"
                  class="btn btn-sm btn-danger-ghost"
                  data-action="remove-att"
                  data-att-id="${att.id}"
                  aria-label="Remove attachment ${escapeHtml(att.filename)}"
                >
                  &times; Remove
                </button>
              </div>

              <span class="alt-char-count" id="alt-count-${att.id}">
                ${(att.altText || "").length} / ${getAltTextLimit()} chars
              </span>
            </div>
          </div>
        </div>
      `;
    })
    .join("");
}

/* ==========================================================================
   Post Publishing & Accessibility Guard
   ========================================================================== */

function showComposerFeedback(message, type = "info", actionHtml = "") {
  const el = document.getElementById("composer-feedback");
  if (!el) return;
  if (!message) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  el.className = `composer-feedback ${type}`;
  el.innerHTML = `<div>${escapeHtml(message)}</div>${actionHtml ? `<div style="margin-top:8px;">${actionHtml}</div>` : ""}`;
  el.classList.remove("hidden");
}

async function publishPost(bypassWarn = false) {
  if (state.isPublishing) return;

  const statusText = document.getElementById("composer-textarea")?.value || "";
  const spoilerText = state.cwEnabled
    ? document.getElementById("cw-input")?.value || ""
    : "";
  const visibility = document.getElementById("post-visibility")?.value || "public";
  const language = document.getElementById("post-language")?.value || "en";

  if (!statusText.trim() && state.attachments.length === 0) {
    showComposerFeedback("Please write a message or attach an image/video before publishing.", "error");
    announce("Cannot publish an empty post.", true);
    return;
  }

  const maxChars = getMaxPostChars();
  const totalChars = statusText.length + spoilerText.length;
  if (totalChars > maxChars) {
    showComposerFeedback(
      `Your post is ${totalChars} characters, which exceeds this server's limit of ${maxChars} characters.`,
      "error"
    );
    announce(`Post exceeds the server limit of ${maxChars} characters.`, true);
    return;
  }

  // Check if any media is still being scanned by AI
  const scanningMedia = state.attachments.filter(
    (a) => a.status === "scanning" || a.status === "preparing"
  );
  if (scanningMedia.length > 0) {
    showComposerFeedback(
      "AI is currently scanning your attached media to fill in the alt-text. Please wait a moment.",
      "warning"
    );
    announce("Please wait for AI alt-text scanning to finish.", true);
    return;
  }

  // Pre-Post Accessibility Guard: Check for missing alt-text
  const missingAltMedia = state.attachments.filter(
    (a) => !a.altText || !a.altText.trim()
  );
  const policy = state.config?.ai?.require_alt_text || "warn";

  if (missingAltMedia.length > 0 && policy !== "off") {
    if (policy === "require") {
      showComposerFeedback(
        `Accessibility Policy: ${missingAltMedia.length} attached media file(s) do not have alt-text for blind users yet.`,
        "error",
        `<button type="button" class="btn btn-sm btn-accent" id="guard-autogen-btn">&#10024; Auto-Generate Missing Alt-Text with AI</button>`
      );
      document
        .getElementById("guard-autogen-btn")
        ?.addEventListener("click", () => scanAllAttachmentsWithAI());
      announce("Alt-text is required on all attached media before posting.", true);
      return;
    }

    if (policy === "warn" && !bypassWarn) {
      showComposerFeedback(
        `Heads up: ${missingAltMedia.length} attached media file(s) have empty alt-text, which makes them inaccessible to blind users.`,
        "warning",
        `<div style="display:flex;gap:8px;flex-wrap:wrap;">
           <button type="button" class="btn btn-sm btn-accent" id="guard-autogen-btn">&#10024; Generate Alt-Text with AI</button>
           <button type="button" class="btn btn-sm btn-ghost" id="guard-post-anyway-btn">Post Anyway</button>
         </div>`
      );
      document
        .getElementById("guard-autogen-btn")
        ?.addEventListener("click", () => {
          showComposerFeedback("");
          scanAllAttachmentsWithAI();
        });
      document
        .getElementById("guard-post-anyway-btn")
        ?.addEventListener("click", () => publishPost(true));
      announce("Warning: Attached media is missing alt-text.", true);
      return;
    }
  }

  // Ensure Mastodon account is connected
  if (!state.config?.mastodon?.access_token) {
    showComposerFeedback(
      "Connect your Mastodon account in Settings before publishing to the Fediverse.",
      "warning",
      `<button type="button" class="btn btn-sm btn-primary" id="feedback-open-auth-btn">Connect Mastodon Account</button>`
    );
    document
      .getElementById("feedback-open-auth-btn")
      ?.addEventListener("click", () => openSettingsModal("tab-mastodon"));
    announce("Please connect your Mastodon account to publish.", true);
    return;
  }

  const isEditing = Boolean(state.editingStatus?.id);
  state.isPublishing = true;
  const publishBtn = document.getElementById("publish-post-btn");
  if (publishBtn) {
    publishBtn.disabled = true;
    publishBtn.textContent = isEditing ? "Saving..." : "Publishing...";
  }

  try {
    const mediaIds = [];
    const mediaAttributes = [];
    for (let i = 0; i < state.attachments.length; i++) {
      const att = state.attachments[i];
      if (att.existingMediaId) {
        mediaIds.push(att.existingMediaId);
        mediaAttributes.push({
          id: att.existingMediaId,
          description: (att.altText || "").trim(),
        });
      } else {
        showComposerFeedback(
          `Uploading ${att.mediaType} ${i + 1} of ${state.attachments.length} with alt-text to Mastodon...`,
          "info"
        );
        // eslint-disable-next-line no-await-in-loop
        const uploadRes = await apiRequest("/api/mastodon/media", {
          method: "POST",
          body: {
            file_b64: att.fileB64,
            filename: att.filename,
            mime_type: att.mimeType,
            description: (att.altText || "").trim(),
          },
        });
        if (uploadRes.media?.id) {
          mediaIds.push(uploadRes.media.id);
        }
      }
    }

    if (isEditing) {
      showComposerFeedback("Saving edited post to Mastodon...", "info");
      const editRes = await apiRequest("/api/mastodon/status", {
        method: "PUT",
        body: {
          status_id: state.editingStatus.id,
          status: statusText,
          media_ids: mediaIds,
          media_attributes: mediaAttributes,
          spoiler_text: spoilerText,
          language,
        },
      });

      const updatedStatus = editRes.status;
      if (updatedStatus) {
        state.statuses = state.statuses.map((s) => {
          if (s.id === updatedStatus.id) return updatedStatus;
          if (s.reblog && s.reblog.id === updatedStatus.id) {
            return { ...s, reblog: updatedStatus };
          }
          return s;
        });
        renderTimeline();
      }

      document.getElementById("composer-textarea").value = "";
      document.getElementById("cw-input").value = "";
      updateCharCounter();
      clearEditContext();
      state.attachments.forEach((a) => {
        if (a.previewUrl && !a.existingMediaId) URL.revokeObjectURL(a.previewUrl);
      });
      state.attachments = [];
      renderAttachments();

      showComposerFeedback("Post updated successfully!", "success");
      announce("Your post has been updated on Mastodon!");
    } else {
      showComposerFeedback("Publishing status to Mastodon...", "info");
      const postRes = await apiRequest("/api/mastodon/status", {
        method: "POST",
        body: {
          status: statusText,
          media_ids: mediaIds,
          spoiler_text: spoilerText,
          visibility,
          language,
          in_reply_to_id: state.replyToStatus?.id || null,
        },
      });

      // Reset composer
      document.getElementById("composer-textarea").value = "";
      document.getElementById("cw-input").value = "";
      updateCharCounter();
      clearReplyContext();
      state.attachments.forEach((a) => {
        if (a.previewUrl && !a.existingMediaId) URL.revokeObjectURL(a.previewUrl);
      });
      state.attachments = [];
      renderAttachments();

      showComposerFeedback("Post published with accessible alt-text!", "success");
      announce("Post published to Mastodon successfully!");

      if (postRes.status && ["home", "local", "federated", "mine", "account"].includes(state.currentTimeline)) {
        state.statuses.unshift(postRes.status);
        renderTimeline();
      }
    }
  } catch (err) {
    showComposerFeedback(
      err.message || (isEditing ? "Failed to update post." : "Failed to publish post."),
      "error"
    );
    announce(`Failed: ${err.message}`, true);
  } finally {
    state.isPublishing = false;
    if (publishBtn) {
      publishBtn.disabled = false;
      publishBtn.textContent = state.editingStatus ? "Save Changes" : "Publish Post";
    }
  }
}

/* ==========================================================================
   Timelines, Notifications & Status Rendering
   ========================================================================== */

const TIMELINE_META = {
  home: {
    title: "Home Timeline",
    subtitle: "Posts from people and hashtags you follow",
  },
  local: {
    title: "Local Server Timeline",
    subtitle: "Public posts from your Mastodon server",
  },
  federated: {
    title: "Federated Timeline",
    subtitle: "Public posts from across the Fediverse",
  },
  notifications: {
    title: "Notifications",
    subtitle: "Mentions, boosts, favorites, and new followers",
  },
  bookmarks: {
    title: "Bookmarked Posts",
    subtitle: "Posts you saved for later",
  },
  favourites: {
    title: "Favorite Posts",
    subtitle: "Posts you starred",
  },
  mine: {
    title: "My Published Posts",
    subtitle: "Your own recent posts and media",
  },
};

async function loadTimeline(timelineType = state.currentTimeline, append = false) {
  if (timelineType === "account" && state.selectedUser?.id) {
    return loadUserTimeline(state.selectedUser.id, append);
  }

  state.currentTimeline = timelineType;
  state.selectedUser = null;
  state.selectedUserRelationship = null;

  // Hide user profile header controls when on standard timelines
  document.getElementById("user-profile-banner")?.classList.add("hidden");
  document.getElementById("back-from-user-btn")?.classList.add("hidden");
  document.getElementById("header-subscribe-btn")?.classList.add("hidden");

  // Update active nav item
  document.querySelectorAll(".nav-item").forEach((btn) => {
    const isMatch = btn.dataset.timeline === timelineType;
    btn.classList.toggle("active", isMatch);
    if (isMatch) {
      btn.setAttribute("aria-current", "page");
    } else {
      btn.removeAttribute("aria-current");
    }
  });

  const meta = TIMELINE_META[timelineType] || TIMELINE_META.home;
  document.getElementById("timeline-heading").textContent = meta.title;
  document.getElementById("timeline-subtitle").textContent = meta.subtitle;

  const loadingEl = document.getElementById("timeline-loading");
  const errorEl = document.getElementById("timeline-error");
  const feedEl = document.getElementById("timeline-feed");
  const loadMoreBtn = document.getElementById("load-more-btn");

  loadingEl.classList.remove("hidden");
  errorEl.classList.add("hidden");
  feedEl.setAttribute("aria-busy", "true");

  try {
    if (timelineType === "notifications") {
      const maxId =
        append && state.notifications.length > 0
          ? state.notifications[state.notifications.length - 1].id
          : "";
      const res = await apiRequest(
        `/api/mastodon/notifications${maxId ? `?max_id=${encodeURIComponent(maxId)}` : ""}`
      );
      state.notifications = append
        ? [...state.notifications, ...(res.notifications || [])]
        : res.notifications || [];
      renderNotifications();
      loadMoreBtn.classList.toggle("hidden", (res.notifications || []).length < 20);
    } else {
      const maxId =
        append && state.statuses.length > 0
          ? state.statuses[state.statuses.length - 1].id
          : "";
      const query = new URLSearchParams({ type: timelineType });
      if (maxId) query.set("max_id", maxId);
      const res = await apiRequest(`/api/mastodon/timeline?${query.toString()}`);
      state.statuses = append
        ? [...state.statuses, ...(res.statuses || [])]
        : res.statuses || [];
      renderTimeline();
      loadMoreBtn.classList.toggle("hidden", (res.statuses || []).length < 20);
    }
  } catch (err) {
    errorEl.innerHTML = `
      <div>
        <strong>Could not load ${escapeHtml(meta.title)}:</strong> ${escapeHtml(err.message)}
        <div style="margin-top:6px;">
          <button type="button" class="btn btn-sm btn-secondary" id="err-switch-local-btn">Browse Public Local Timeline</button>
          <button type="button" class="btn btn-sm btn-primary" id="err-open-settings-btn">Configure Mastodon Account</button>
        </div>
      </div>
    `;
    errorEl.classList.remove("hidden");
    document
      .getElementById("err-switch-local-btn")
      ?.addEventListener("click", () => loadTimeline("local"));
    document
      .getElementById("err-open-settings-btn")
      ?.addEventListener("click", () => openSettingsModal("tab-mastodon"));
  } finally {
    loadingEl.classList.add("hidden");
    feedEl.setAttribute("aria-busy", "false");
  }
}

async function viewUserPosts(accountId, fallbackMeta = {}) {
  if (!accountId) return;
  if (state.currentTimeline !== "account") {
    state.previousTimeline = state.currentTimeline || "home";
  }
  state.currentTimeline = "account";
  state.selectedUser = {
    id: accountId,
    display_name: fallbackMeta.displayName || fallbackMeta.username || "User",
    acct: (fallbackMeta.acct || "").replace(/^@/, ""),
    avatar: fallbackMeta.avatar || "",
  };
  state.selectedUserRelationship = null;

  // Close thread modal if open so user sees the timeline
  document.getElementById("thread-modal")?.classList.add("hidden");

  await loadUserTimeline(accountId, false);
}

async function loadUserTimeline(accountId, append = false) {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.remove("active");
    btn.removeAttribute("aria-current");
  });

  const displayName =
    state.selectedUser?.display_name ||
    state.selectedUser?.username ||
    "User";
  const acct = state.selectedUser?.acct ? `@${state.selectedUser.acct}` : "";

  document.getElementById("timeline-heading").textContent = `Posts by ${displayName}`;
  document.getElementById("timeline-subtitle").textContent = acct
    ? `${acct} · Personal Posts`
    : "Personal Posts";
  document.getElementById("back-from-user-btn")?.classList.remove("hidden");

  renderUserProfileBanner();

  const loadingEl = document.getElementById("timeline-loading");
  const errorEl = document.getElementById("timeline-error");
  const feedEl = document.getElementById("timeline-feed");
  const loadMoreBtn = document.getElementById("load-more-btn");

  loadingEl.classList.remove("hidden");
  errorEl.classList.add("hidden");
  feedEl.setAttribute("aria-busy", "true");

  try {
    const maxId =
      append && state.statuses.length > 0
        ? state.statuses[state.statuses.length - 1].id
        : "";
    const query = new URLSearchParams({
      type: "account",
      account_id: accountId,
    });
    if (maxId) query.set("max_id", maxId);

    const [timelineRes, accountRes] = await Promise.all([
      apiRequest(`/api/mastodon/timeline?${query.toString()}`),
      append
        ? Promise.resolve(null)
        : apiRequest(`/api/mastodon/account?id=${encodeURIComponent(accountId)}`).catch(
            () => null
          ),
    ]);

    if (accountRes?.account) {
      state.selectedUser = accountRes.account;
      state.selectedUserRelationship = accountRes.relationship || null;
      const newName =
        accountRes.account.display_name ||
        accountRes.account.username ||
        displayName;
      const newAcct = accountRes.account.acct
        ? `@${accountRes.account.acct}`
        : acct;
      document.getElementById(
        "timeline-heading"
      ).textContent = `Posts by ${newName}`;
      document.getElementById("timeline-subtitle").textContent = newAcct
        ? `${newAcct} · Personal Posts`
        : "Personal Posts";
    }

    renderUserProfileBanner();

    state.statuses = append
      ? [...state.statuses, ...(timelineRes.statuses || [])]
      : timelineRes.statuses || [];
    renderTimeline();
    loadMoreBtn.classList.toggle(
      "hidden",
      (timelineRes.statuses || []).length < 20
    );

    if (!append) {
      announce(
        `Showing personal posts by ${
          state.selectedUser?.display_name || displayName
        }.`
      );
      document
        .querySelector(".timeline-section")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (err) {
    errorEl.innerHTML = `<div><strong>Could not load user posts:</strong> ${escapeHtml(err.message)}</div>`;
    errorEl.classList.remove("hidden");
  } finally {
    loadingEl.classList.add("hidden");
    feedEl.setAttribute("aria-busy", "false");
  }
}

function renderUserProfileBanner() {
  const bannerEl = document.getElementById("user-profile-banner");
  const headerSubBtn = document.getElementById("header-subscribe-btn");
  if (!bannerEl) return;

  const user = state.selectedUser;
  if (!user || state.currentTimeline !== "account") {
    bannerEl.classList.add("hidden");
    headerSubBtn?.classList.add("hidden");
    return;
  }

  const myId = state.config?.mastodon?.account?.id;
  const isSelf = Boolean(myId && String(myId) === String(user.id));
  const rel = state.selectedUserRelationship;
  const isFollowing = Boolean(rel?.following);
  const isRequested = Boolean(rel?.requested);
  const isNotifying = Boolean(rel?.notifying);
  const displayName = user.display_name || user.username || "User";
  const acct = user.acct ? `@${user.acct}` : "";
  const avatar = user.avatar_static || user.avatar || "";

  // Update top-of-timeline header Subscribe button
  if (headerSubBtn) {
    if (isSelf) {
      headerSubBtn.classList.add("hidden");
    } else {
      headerSubBtn.classList.remove("hidden");
      if (isFollowing) {
        headerSubBtn.textContent = "✓ Subscribed (Unsubscribe)";
        headerSubBtn.className = "btn btn-sm btn-secondary";
        headerSubBtn.dataset.subAction = "unfollow";
      } else if (isRequested) {
        headerSubBtn.textContent = "Follow Requested (Cancel)";
        headerSubBtn.className = "btn btn-sm btn-secondary";
        headerSubBtn.dataset.subAction = "unfollow";
      } else {
        headerSubBtn.textContent = `+ Subscribe to ${displayName}`;
        headerSubBtn.className = "btn btn-sm btn-primary";
        headerSubBtn.dataset.subAction = "follow";
      }
    }
  }

  const subscribeControlsHtml = isSelf
    ? `<span class="composer-badge">Your Profile</span>`
    : `
      <div class="user-subscribe-controls">
        <button
          type="button"
          class="btn ${isFollowing || isRequested ? "btn-secondary" : "btn-primary"}"
          data-action="subscribe-user"
          data-sub-action="${isFollowing || isRequested ? "unfollow" : "follow"}"
          aria-pressed="${isFollowing ? "true" : "false"}"
        >
          ${
            isFollowing
              ? "&#10003; Subscribed to Posts (Unsubscribe)"
              : isRequested
              ? "Follow Requested (Cancel)"
              : "&#10133; Subscribe to Posts (Follow)"
          }
        </button>
        ${
          isFollowing
            ? `<button
                 type="button"
                 class="btn btn-sm ${isNotifying ? "btn-accent" : "btn-ghost"}"
                 data-action="subscribe-user"
                 data-sub-action="${isNotifying ? "notify_off" : "notify_on"}"
                 aria-pressed="${isNotifying ? "true" : "false"}"
                 title="Receive a notification every time ${escapeHtml(displayName)} posts"
               >
                 &#128276; ${isNotifying ? "Post Alerts: ON" : "Notify on New Posts"}
               </button>`
            : ""
        }
      </div>
    `;

  const statsHtml =
    user.statuses_count !== undefined
      ? `<div class="user-profile-stats">
           <span><strong>${Number(user.statuses_count || 0).toLocaleString()}</strong> Posts</span>
           <span><strong>${Number(user.following_count || 0).toLocaleString()}</strong> Following</span>
           <span><strong>${Number(user.followers_count || 0).toLocaleString()}</strong> Followers</span>
         </div>`
      : "";

  bannerEl.innerHTML = `
    <div class="user-profile-top">
      <div class="user-profile-identity">
        ${avatar ? `<img class="user-profile-avatar" src="${escapeHtml(avatar)}" alt="" />` : ""}
        <div>
          <div class="user-profile-name">${escapeHtml(displayName)}</div>
          <div class="user-profile-acct">${escapeHtml(acct)}</div>
        </div>
      </div>
      ${subscribeControlsHtml}
    </div>
    ${user.note ? `<div class="user-profile-bio">${user.note}</div>` : ""}
    ${statsHtml}
  `;
  bannerEl.classList.remove("hidden");
}

async function toggleUserSubscription(action = "follow") {
  const user = state.selectedUser;
  if (!user?.id) return;

  if (!state.config?.mastodon?.access_token) {
    openSettingsModal("tab-mastodon");
    announce("Please sign in to your Mastodon account to subscribe to users.", true);
    return;
  }

  try {
    const res = await apiRequest("/api/mastodon/account/subscribe", {
      method: "POST",
      body: {
        account_id: user.id,
        action,
      },
    });
    state.selectedUserRelationship = res.relationship || null;
    renderUserProfileBanner();

    const displayName = user.display_name || user.username || "user";
    if (action === "unfollow") {
      announce(`Unsubscribed from ${displayName}'s posts.`);
    } else if (action === "notify_on") {
      announce(`Enabled post notifications for ${displayName}.`);
    } else if (action === "notify_off") {
      announce(`Disabled post notifications for ${displayName}.`);
    } else {
      announce(`Subscribed to ${displayName}'s posts!`);
    }
  } catch (err) {
    announce(`Subscription action failed: ${err.message}`, true);
    alert(`Could not update subscription: ${err.message}`);
  }
}

function renderStatusCardHtml(rawStatus) {
  const isBoost = Boolean(rawStatus.reblog);
  const status = rawStatus.reblog || rawStatus;
  const account = status.account || {};
  const displayName = account.display_name || account.username || "Unknown";
  const acct = account.acct ? `@${account.acct}` : "";
  const createdAt = formatRelativeTime(status.created_at);
  const hasCW = Boolean(status.spoiler_text && status.spoiler_text.trim());
  const myId = state.config?.mastodon?.account?.id;
  const isOwnPost = Boolean(myId && account.id === myId);

  const mediaAttachments = Array.isArray(status.media_attachments)
    ? status.media_attachments
    : [];

  const mediaHtml =
    mediaAttachments.length > 0
      ? `<div class="status-media-grid">
           ${mediaAttachments
             .map((m) => {
               const alt = (m.description || "").trim();
               const mediaUrl = m.url || m.preview_url || "";
               const isVideo =
                 m.type === "video" || m.type === "gifv";
               const visual = isVideo
                 ? `<video src="${escapeHtml(mediaUrl)}" controls ${m.type === "gifv" ? "loop muted" : ""} preload="metadata" aria-label="${escapeHtml(alt || "Uncaptioned video attachment")}"></video>`
                 : `<img src="${escapeHtml(m.preview_url || mediaUrl)}" alt="${escapeHtml(alt || "Uncaptioned image attachment")}" loading="lazy" />`;

               return `
                 <div class="timeline-media-item" data-media-id="${escapeHtml(m.id)}">
                   <div class="timeline-media-visual">${visual}</div>
                   <div class="timeline-alt-box">
                     <div class="timeline-alt-header">
                       <span class="alt-tag ${alt ? "has-alt" : "no-alt"}">
                         ${alt ? "ALT TEXT PROVIDED" : "NO ALT TEXT"}
                       </span>
                       <div style="display:flex;gap:6px;">
                         ${
                           alt
                             ? `<button type="button" class="btn btn-sm btn-ghost" data-action="speak-timeline-alt" data-alt="${escapeHtml(alt)}">&#128266; Read Aloud</button>`
                             : ""
                         }
                         <button
                           type="button"
                           class="btn btn-sm btn-accent"
                           data-action="caption-timeline-media"
                           data-media-url="${escapeHtml(m.preview_url || mediaUrl)}"
                           data-media-type="${isVideo ? "video" : "image"}"
                           data-target-box="tl-alt-text-${escapeHtml(m.id)}"
                         >
                           &#10024; ${alt ? "Re-Describe with AI" : "Describe with AI"}
                         </button>
                       </div>
                     </div>
                     <div class="timeline-alt-text" id="tl-alt-text-${escapeHtml(m.id)}">
                       ${
                         alt
                           ? escapeHtml(alt)
                           : "<em>No author alt-text provided. Click &ldquo;Describe with AI&rdquo; to generate a screen-reader description.</em>"
                       }
                     </div>
                   </div>
                 </div>
               `;
             })
             .join("")}
         </div>`
      : "";

  const cwHtml = hasCW
    ? `<div class="status-cw-box">
         <span><strong>CW:</strong> ${escapeHtml(status.spoiler_text)}</span>
         <button type="button" class="btn btn-sm btn-secondary" data-action="toggle-status-cw" data-status-id="${escapeHtml(status.id)}">
           Show / Hide Content
         </button>
       </div>`
    : "";

  return `
    <article class="status-card" data-status-id="${escapeHtml(status.id)}" aria-label="Post by ${escapeHtml(displayName)}">
      ${
        isBoost
          ? `<div class="status-boost-banner">
               &#128257; Boosted by
               <button
                 type="button"
                 class="user-link-btn"
                 data-action="view-user"
                 data-account-id="${escapeHtml(rawStatus.account?.id || "")}"
                 data-account-name="${escapeHtml(rawStatus.account?.display_name || rawStatus.account?.username || "")}"
                 data-account-acct="${escapeHtml(rawStatus.account?.acct || "")}"
                 data-account-avatar="${escapeHtml(rawStatus.account?.avatar_static || rawStatus.account?.avatar || "")}"
               >${escapeHtml(rawStatus.account?.display_name || rawStatus.account?.username || "")}</button>
             </div>`
          : ""
      }
      <div class="status-header">
        <button
          type="button"
          class="status-author-btn"
          data-action="view-user"
          data-account-id="${escapeHtml(account.id || "")}"
          data-account-name="${escapeHtml(displayName)}"
          data-account-acct="${escapeHtml(account.acct || "")}"
          data-account-avatar="${escapeHtml(account.avatar_static || account.avatar || "")}"
          title="View personal posts by ${escapeHtml(displayName)}"
          aria-label="View personal posts by ${escapeHtml(displayName)} (${escapeHtml(acct)})"
        >
          <img class="status-avatar" src="${escapeHtml(account.avatar_static || account.avatar || "")}" alt="" loading="lazy" />
          <div class="status-author-names">
            <div class="status-display-name">${escapeHtml(displayName)}</div>
            <div class="status-acct">${escapeHtml(acct)}</div>
          </div>
        </button>
        <span class="status-time" title="${escapeHtml(status.created_at || "")}">${escapeHtml(createdAt)}</span>
      </div>

      ${cwHtml}

      <div id="status-body-${escapeHtml(status.id)}" class="${hasCW ? "hidden" : ""}">
        <div class="status-content">${status.content || ""}</div>
        ${mediaHtml}
      </div>

      <div class="status-actions" role="group" aria-label="Post actions">
        <button
          type="button"
          class="action-btn"
          data-action="reply"
          data-status-id="${escapeHtml(status.id)}"
          aria-label="Reply to ${escapeHtml(displayName)}"
        >
          &#8617; Reply (${status.replies_count || 0})
        </button>

        <button
          type="button"
          class="action-btn ${status.reblogged ? "active" : ""}"
          data-action="reblog"
          data-status-id="${escapeHtml(status.id)}"
          data-active="${status.reblogged ? "true" : "false"}"
          aria-pressed="${status.reblogged ? "true" : "false"}"
          aria-label="Boost post (${status.reblogs_count || 0})"
        >
          &#128257; Boost (${status.reblogs_count || 0})
        </button>

        <button
          type="button"
          class="action-btn ${status.favourited ? "active" : ""}"
          data-action="favourite"
          data-status-id="${escapeHtml(status.id)}"
          data-active="${status.favourited ? "true" : "false"}"
          aria-pressed="${status.favourited ? "true" : "false"}"
          aria-label="Favorite post (${status.favourites_count || 0})"
        >
          &#11088; Favorite (${status.favourites_count || 0})
        </button>

        <button
          type="button"
          class="action-btn ${status.bookmarked ? "active" : ""}"
          data-action="bookmark"
          data-status-id="${escapeHtml(status.id)}"
          data-active="${status.bookmarked ? "true" : "false"}"
          aria-pressed="${status.bookmarked ? "true" : "false"}"
          aria-label="Bookmark post"
        >
          &#128278; Bookmark
        </button>

        <button
          type="button"
          class="action-btn"
          data-action="thread"
          data-status-id="${escapeHtml(status.id)}"
          aria-label="View conversation thread"
        >
          &#128172; Thread
        </button>

        ${
          isOwnPost
            ? `<button
                 type="button"
                 class="action-btn"
                 data-action="edit"
                 data-status-id="${escapeHtml(status.id)}"
                 aria-label="Edit your post"
               >
                 &#9998; Edit
               </button>
               <button
                 type="button"
                 class="action-btn"
                 style="color:var(--danger-color);"
                 data-action="delete"
                 data-status-id="${escapeHtml(status.id)}"
                 aria-label="Delete your post"
               >
                 &#128465; Delete
               </button>`
            : ""
        }
      </div>
    </article>
  `;
}

function renderTimeline() {
  const feedEl = document.getElementById("timeline-feed");
  if (!feedEl) return;

  if (!state.statuses || state.statuses.length === 0) {
    feedEl.innerHTML = `
      <div class="status-card" style="text-align:center;color:var(--text-secondary);">
        No posts found in this timeline yet.
      </div>
    `;
    return;
  }

  feedEl.innerHTML = state.statuses.map(renderStatusCardHtml).join("");
}

function renderNotifications() {
  const feedEl = document.getElementById("timeline-feed");
  if (!feedEl) return;

  if (!state.notifications || state.notifications.length === 0) {
    feedEl.innerHTML = `
      <div class="status-card" style="text-align:center;color:var(--text-secondary);">
        No notifications yet.
      </div>
    `;
    return;
  }

  const typeLabels = {
    mention: "mentioned you",
    reblog: "boosted your post",
    favourite: "favorited your post",
    follow: "followed you",
    poll: "poll ended",
    update: "edited a post",
  };

  feedEl.innerHTML = state.notifications
    .map((n) => {
      const acct = n.account || {};
      const label = typeLabels[n.type] || n.type;
      const statusHtml = n.status ? renderStatusCardHtml(n.status) : "";
      return `
        <div class="status-card">
          <div class="status-boost-banner">
            &#128276;
            <button
              type="button"
              class="user-link-btn"
              data-action="view-user"
              data-account-id="${escapeHtml(acct.id || "")}"
              data-account-name="${escapeHtml(acct.display_name || acct.username || "Someone")}"
              data-account-acct="${escapeHtml(acct.acct || "")}"
              data-account-avatar="${escapeHtml(acct.avatar_static || acct.avatar || "")}"
            >${escapeHtml(acct.display_name || acct.username || "Someone")}</button>
            (@${escapeHtml(acct.acct || "")}) ${escapeHtml(label)} &bull; ${escapeHtml(formatRelativeTime(n.created_at))}
          </div>
          ${statusHtml}
        </div>
      `;
    })
    .join("");
}

/* ==========================================================================
   Timeline Post Actions & On-Demand AI Captioning for Timeline Posts
   ========================================================================== */

function findStatusById(statusId) {
  for (const s of state.statuses) {
    if (s.id === statusId) return s;
    if (s.reblog && s.reblog.id === statusId) return s.reblog;
  }
  return null;
}

function startReplyToStatus(statusId) {
  const target = findStatusById(statusId);
  if (!target) return;

  clearEditContext();
  state.replyToStatus = target;
  const banner = document.getElementById("reply-context-banner");
  const acctEl = document.getElementById("reply-target-acct");
  const excerptEl = document.getElementById("reply-target-excerpt");
  const textarea = document.getElementById("composer-textarea");

  const handle = `@${target.account?.acct || target.account?.username || ""}`;
  acctEl.textContent = handle;
  excerptEl.textContent = stripHtmlTags(target.content).slice(0, 110);
  banner.classList.remove("hidden");

  if (!textarea.value.includes(handle)) {
    textarea.value = `${handle} ${textarea.value}`;
  }
  if (target.spoiler_text) {
    state.cwEnabled = true;
    document.getElementById("cw-container").classList.remove("hidden");
    document.getElementById("cw-input").value = target.spoiler_text;
  }
  if (target.visibility) {
    document.getElementById("post-visibility").value = target.visibility;
  }

  updateCharCounter();
  textarea.focus();
  announce(`Replying to ${handle}. Composer focused.`);
}

function clearReplyContext() {
  state.replyToStatus = null;
  document.getElementById("reply-context-banner")?.classList.add("hidden");
}

async function startEditStatus(statusId) {
  const target = findStatusById(statusId);
  if (!target) return;

  clearReplyContext();
  state.editingStatus = target;

  const banner = document.getElementById("edit-context-banner");
  const excerptEl = document.getElementById("edit-target-excerpt");
  const textarea = document.getElementById("composer-textarea");
  const publishBtn = document.getElementById("publish-post-btn");

  const plainExcerpt = stripHtmlTags(target.content).slice(0, 110);
  if (excerptEl) excerptEl.textContent = plainExcerpt;
  banner?.classList.remove("hidden");
  if (publishBtn) publishBtn.textContent = "Save Changes";

  // Pre-fill with fallback plain text immediately, then fetch exact raw source from Mastodon
  textarea.value = stripHtmlTags(target.content);
  if (target.spoiler_text) {
    state.cwEnabled = true;
    document.getElementById("cw-container")?.classList.remove("hidden");
    document.getElementById("cw-input").value = target.spoiler_text;
  } else {
    state.cwEnabled = false;
    document.getElementById("cw-container")?.classList.add("hidden");
    document.getElementById("cw-input").value = "";
  }
  if (target.language) {
    const langSelect = document.getElementById("post-language");
    if (langSelect) langSelect.value = target.language;
  }

  // Load existing media attachments into composer so their alt-text can also be edited
  state.attachments.forEach((a) => {
    if (a.previewUrl && !a.existingMediaId) URL.revokeObjectURL(a.previewUrl);
  });
  const defaultDetail = state.config?.ai?.detail_level || "descriptive";
  state.attachments = Array.isArray(target.media_attachments)
    ? target.media_attachments.map((m, idx) => {
        const isVideo = m.type === "video" || m.type === "gifv";
        const alt = (m.description || "").trim();
        return {
          id: `att-${attachmentIdCounter++}`,
          existingMediaId: m.id,
          file: null,
          filename: `${isVideo ? "video" : "image"}-${idx + 1}`,
          mediaType: isVideo ? "video" : "image",
          mimeType: isVideo ? "video/mp4" : "image/jpeg",
          previewUrl: m.preview_url || m.url || "",
          remoteUrl: m.preview_url || m.url || "",
          fileB64: null,
          aiFramesB64: [],
          altText: alt,
          status: alt ? "ready" : "missing",
          providerUsed: "",
          errorMsg: "",
          detailLevel: defaultDetail,
        };
      })
    : [];
  renderAttachments();
  updateCharCounter();

  try {
    const res = await apiRequest(
      `/api/mastodon/status/source?id=${encodeURIComponent(statusId)}`
    );
    if (res.source && typeof res.source.text === "string") {
      textarea.value = res.source.text;
      if (typeof res.source.spoiler_text === "string") {
        if (res.source.spoiler_text) {
          state.cwEnabled = true;
          document.getElementById("cw-container")?.classList.remove("hidden");
          document.getElementById("cw-input").value = res.source.spoiler_text;
        }
      }
      updateCharCounter();
    }
  } catch (_) {
    // Keep fallback stripped HTML text if source endpoint fails
  }

  document
    .querySelector(".composer-card")
    ?.scrollIntoView({ behavior: "smooth", block: "start" });
  textarea.focus();
  announce("Editing your post in the composer.");
}

function clearEditContext() {
  state.editingStatus = null;
  document.getElementById("edit-context-banner")?.classList.add("hidden");
  const publishBtn = document.getElementById("publish-post-btn");
  if (publishBtn) publishBtn.textContent = "Publish Post";
}

async function performStatusAction(statusId, baseAction, isCurrentlyActive) {
  const actionMap = {
    reblog: isCurrentlyActive ? "unreblog" : "reblog",
    favourite: isCurrentlyActive ? "unfavourite" : "favourite",
    bookmark: isCurrentlyActive ? "unbookmark" : "bookmark",
    delete: "delete",
  };
  const action = actionMap[baseAction];
  if (!action) return;

  if (action === "delete" && !window.confirm("Delete this post from Mastodon?")) {
    return;
  }

  try {
    const res = await apiRequest("/api/mastodon/status/action", {
      method: "POST",
      body: { status_id: statusId, action },
    });

    if (action === "delete") {
      state.statuses = state.statuses.filter((s) => s.id !== statusId);
      renderTimeline();
      announce("Post deleted.");
      return;
    }

    const updated = res.status?.reblog || res.status;
    if (updated) {
      state.statuses = state.statuses.map((s) => {
        if (s.id === statusId) return updated;
        if (s.reblog && s.reblog.id === statusId) {
          return { ...s, reblog: updated };
        }
        return s;
      });
      renderTimeline();
      announce(`Action ${action} completed.`);
    }
  } catch (err) {
    announce(`Action failed: ${err.message}`, true);
    alert(`Mastodon error: ${err.message}`);
  }
}

async function openThreadModal(statusId) {
  const modal = document.getElementById("thread-modal");
  const content = document.getElementById("thread-content");
  if (!modal || !content) return;

  modal.classList.remove("hidden");
  content.innerHTML = `<div class="loading-banner"><span class="spinner"></span><span>Loading thread...</span></div>`;

  try {
    const res = await apiRequest(
      `/api/mastodon/thread?id=${encodeURIComponent(statusId)}`
    );
    const items = [
      ...(res.ancestors || []),
      res.status,
      ...(res.descendants || []),
    ].filter(Boolean);
    content.innerHTML = `<div class="timeline-feed">${items
      .map(renderStatusCardHtml)
      .join("")}</div>`;
    announce(`Loaded thread with ${items.length} posts.`);
  } catch (err) {
    content.innerHTML = `<div class="error-banner">${escapeHtml(err.message)}</div>`;
  }
}

async function captionTimelineMediaWithAI(mediaUrl, mediaType, targetBoxId, btnEl) {
  const boxEl = document.getElementById(targetBoxId);
  if (!boxEl) return;

  const originalBtnText = btnEl ? btnEl.innerHTML : "";
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.innerHTML = "Scanning...";
  }
  boxEl.innerHTML = `<span class="spinner" style="width:12px;height:12px;"></span> Generating AI description for screen reader...`;
  announce("Scanning timeline media with AI...");

  try {
    const provider =
      document.getElementById("quick-provider-select")?.value ||
      state.config?.ai?.provider ||
      "ollama";
    const res = await apiRequest("/api/ai/caption", {
      method: "POST",
      body: {
        remote_url: mediaUrl,
        media_type: mediaType,
        ai_override: { provider },
      },
    });
    boxEl.textContent = res.caption || "No description generated.";
    announce(`AI description: ${res.caption}`);
  } catch (err) {
    boxEl.innerHTML = `<span style="color:var(--danger-color);">AI error: ${escapeHtml(err.message)}</span>`;
    announce(`Failed to describe timeline media: ${err.message}`, true);
  } finally {
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.innerHTML = originalBtnText;
    }
  }
}

/* ==========================================================================
   Settings, AI Provider Config & Mastodon OAuth
   ========================================================================== */

function applyConfigToUI(cfg) {
  state.config = cfg;
  if (!cfg) return;

  // Theme
  const theme = cfg.ui?.theme || "dark";
  document.documentElement.setAttribute("data-theme", theme);
  const themeSelect = document.getElementById("theme-select");
  if (themeSelect) themeSelect.value = theme;

  // Default visibility
  const visSelect = document.getElementById("post-visibility");
  if (visSelect && cfg.ui?.default_visibility) {
    visSelect.value = cfg.ui.default_visibility;
  }

  // AI Status Pill & Quick Controls
  const ai = cfg.ai || {};
  const provider = ai.provider || "ollama";
  const modelMap = {
    ollama: ai.ollama?.model || "llava:latest",
    gemini: ai.gemini?.model || "gemini-2.5-flash",
    openai: ai.openai?.model || "gpt-4o-mini",
    anthropic: ai.anthropic?.model || "claude-3-7-sonnet-latest",
  };
  const labelEl = document.getElementById("ai-status-label");
  if (labelEl) {
    labelEl.textContent = `AI: ${provider.toUpperCase()} · ${modelMap[provider] || ""}`;
  }

  const quickProvider = document.getElementById("quick-provider-select");
  if (quickProvider) quickProvider.value = provider;

  const quickDetail = document.getElementById("quick-detail-select");
  if (quickDetail) quickDetail.value = ai.detail_level || "descriptive";

  const quickAuto = document.getElementById("quick-auto-scan");
  if (quickAuto) quickAuto.checked = ai.auto_generate !== false;

  // Guard badge
  const guardBadge = document.getElementById("composer-guard-badge");
  if (guardBadge) {
    const policy = ai.require_alt_text || "warn";
    guardBadge.textContent =
      policy === "require"
        ? "Alt-Text Guard: Required"
        : policy === "warn"
        ? "Alt-Text Guard: Warn"
        : "Alt-Text Guard: Off";
  }

  // Account card
  const m = cfg.mastodon || {};
  const connectedEl = document.getElementById("account-connected");
  const disconnectedEl = document.getElementById("account-disconnected");
  if (m.access_token && m.account) {
    connectedEl.classList.remove("hidden");
    disconnectedEl.classList.add("hidden");
    const displayName =
      m.account.display_name || m.account.username || "Connected";
    document.getElementById("account-avatar").src =
      m.account.avatar_static || m.account.avatar || "";
    document.getElementById("account-display-name").textContent = displayName;
    const host = (m.instance_url || "").replace(/^https?:\/\//, "");
    document.getElementById("account-handle").textContent = `@${m.account.acct}@${host}`;
    const profileBtn = document.getElementById("account-profile-btn");
    if (profileBtn) {
      profileBtn.setAttribute(
        "aria-label",
        `View your own posts (${displayName})`
      );
      profileBtn.title = `Click to view your own posts (${displayName})`;
    }
  } else {
    connectedEl.classList.add("hidden");
    disconnectedEl.classList.remove("hidden");
  }

  // Populate Settings Modal Inputs
  document
    .querySelectorAll('input[name="settings-ai-provider"]')
    .forEach((radio) => {
      radio.checked = radio.value === provider;
    });
  updateVisibleProviderConfigBox(provider);

  document.getElementById("ollama-base-url").value =
    ai.ollama?.base_url || "http://localhost:11434";
  document.getElementById("ollama-model").value =
    ai.ollama?.model || "llava:latest";

  document.getElementById("gemini-api-key").value = ai.gemini?.api_key || "";
  document.getElementById("gemini-model").value =
    ai.gemini?.model || "gemini-2.5-flash";

  document.getElementById("openai-base-url").value =
    ai.openai?.base_url || "https://api.openai.com/v1";
  document.getElementById("openai-api-key").value = ai.openai?.api_key || "";
  document.getElementById("openai-model").value =
    ai.openai?.model || "gpt-4o-mini";

  document.getElementById("anthropic-api-key").value =
    ai.anthropic?.api_key || "";
  document.getElementById("anthropic-model").value =
    ai.anthropic?.model || "claude-3-7-sonnet-latest";

  document.getElementById("mastodon-instance-url").value =
    m.instance_url || "https://mastodon.social";
  document.getElementById("mastodon-access-token").value =
    m.access_token || "";

  document.getElementById("settings-require-alt").value =
    ai.require_alt_text || "warn";
  document.getElementById("settings-caption-language").value =
    ai.language || "English";
  document.getElementById("settings-keyframes-count").value = String(
    cfg.ui?.video_keyframes_count || 4
  );
  document.getElementById("settings-include-context").checked =
    ai.include_post_context !== false;
  document.getElementById("settings-custom-instructions").value =
    ai.custom_instructions || "";

  const pathEl = document.getElementById("config-path-display");
  if (pathEl && cfg.config_path) {
    pathEl.textContent = `Config: ${cfg.config_path}`;
  }

  updateCharCounter();
}

async function refreshInstanceLimits() {
  try {
    const res = await apiRequest("/api/mastodon/instance");
    if (res.config) {
      applyConfigToUI(res.config);
    }
  } catch (_) {
    // Keep stored/default limit if instance is temporarily unreachable
  }
}

function updateVisibleProviderConfigBox(provider) {
  ["ollama", "gemini", "openai", "anthropic"].forEach((p) => {
    const box = document.getElementById(`provider-config-${p}`);
    if (box) {
      box.classList.toggle("hidden", p !== provider);
    }
  });
}

function openSettingsModal(tabId = "tab-ai") {
  const modal = document.getElementById("settings-modal");
  if (!modal) return;
  modal.classList.remove("hidden");
  switchSettingsTab(tabId);
}

function closeSettingsModal() {
  document.getElementById("settings-modal")?.classList.add("hidden");
}

function switchSettingsTab(tabId) {
  document.querySelectorAll(".modal-tab").forEach((btn) => {
    const isTarget = btn.dataset.tab === tabId;
    btn.classList.toggle("active", isTarget);
    btn.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.id !== tabId);
  });
}

async function detectLocalOllamaModels() {
  const baseUrl =
    document.getElementById("ollama-base-url")?.value ||
    "http://localhost:11434";
  const feedbackEl = document.getElementById("ollama-detect-feedback");
  const datalist = document.getElementById("ollama-models-datalist");
  const modelInput = document.getElementById("ollama-model");

  if (feedbackEl) {
    feedbackEl.textContent = `Checking Ollama at ${baseUrl}...`;
  }

  try {
    const res = await apiRequest(
      `/api/ai/ollama-models?base_url=${encodeURIComponent(baseUrl)}`
    );
    if (!res.reachable) {
      feedbackEl.innerHTML = `<span style="color:var(--warning-color);">Could not reach Ollama at ${escapeHtml(baseUrl)}. Ensure Ollama is running (<code>ollama serve</code>).</span>`;
      return;
    }
    const models = res.models || [];
    if (models.length === 0) {
      feedbackEl.innerHTML = `Connected to Ollama, but no models are installed yet. Run <code>ollama pull llava</code> or <code>ollama pull llama3.2-vision</code> in your terminal.`;
      return;
    }

    datalist.innerHTML = models
      .map((m) => `<option value="${escapeHtml(m.name)}"></option>`)
      .join("");

    const visionModels = models.filter((m) => m.is_vision_likely);
    if (visionModels.length > 0 && modelInput) {
      modelInput.value = visionModels[0].name;
    } else if (models.length > 0 && modelInput && !modelInput.value) {
      modelInput.value = models[0].name;
    }

    feedbackEl.innerHTML = `<span style="color:var(--success-color);">Found ${models.length} local Ollama model(s): ${models
      .map((m) => `<code>${escapeHtml(m.name)}</code>`)
      .join(", ")}</span>`;
    announce(`Detected ${models.length} local Ollama models.`);
  } catch (err) {
    if (feedbackEl) {
      feedbackEl.textContent = `Error checking Ollama: ${err.message}`;
    }
  }
}

async function saveSettingsFromModal() {
  const selectedProvider =
    document.querySelector('input[name="settings-ai-provider"]:checked')
      ?.value || "ollama";

  const updates = {
    mastodon: {
      instance_url:
        document.getElementById("mastodon-instance-url")?.value.trim() ||
        "https://mastodon.social",
    },
    ai: {
      provider: selectedProvider,
      auto_generate:
        document.getElementById("quick-auto-scan")?.checked ?? true,
      detail_level:
        document.getElementById("quick-detail-select")?.value || "descriptive",
      require_alt_text:
        document.getElementById("settings-require-alt")?.value || "warn",
      language:
        document.getElementById("settings-caption-language")?.value.trim() ||
        "English",
      include_post_context:
        document.getElementById("settings-include-context")?.checked ?? true,
      custom_instructions:
        document.getElementById("settings-custom-instructions")?.value || "",
      ollama: {
        base_url:
          document.getElementById("ollama-base-url")?.value.trim() ||
          "http://localhost:11434",
        model:
          document.getElementById("ollama-model")?.value.trim() ||
          "llava:latest",
      },
      gemini: {
        api_key: document.getElementById("gemini-api-key")?.value.trim() || "",
        model:
          document.getElementById("gemini-model")?.value.trim() ||
          "gemini-2.5-flash",
      },
      openai: {
        base_url:
          document.getElementById("openai-base-url")?.value.trim() ||
          "https://api.openai.com/v1",
        api_key: document.getElementById("openai-api-key")?.value.trim() || "",
        model:
          document.getElementById("openai-model")?.value.trim() ||
          "gpt-4o-mini",
      },
      anthropic: {
        api_key:
          document.getElementById("anthropic-api-key")?.value.trim() || "",
        model:
          document.getElementById("anthropic-model")?.value.trim() ||
          "claude-3-7-sonnet-latest",
      },
    },
    ui: {
      theme: document.getElementById("theme-select")?.value || "dark",
      video_keyframes_count: Number(
        document.getElementById("settings-keyframes-count")?.value || 4
      ),
    },
  };

  try {
    const res = await apiRequest("/api/config", {
      method: "POST",
      body: updates,
    });
    applyConfigToUI(res.config);
    closeSettingsModal();
    announce("Settings saved successfully.");
  } catch (err) {
    alert(`Failed to save settings: ${err.message}`);
  }
}

async function startMastodonOAuth() {
  const instanceUrl =
    document.getElementById("mastodon-instance-url")?.value.trim() ||
    "https://mastodon.social";
  const feedbackEl = document.getElementById("mastodon-auth-feedback");
  feedbackEl.className = "composer-feedback info";
  feedbackEl.textContent = `Registering OAuth application with ${instanceUrl}...`;
  feedbackEl.classList.remove("hidden");

  try {
    const res = await apiRequest("/api/mastodon/oauth/start", {
      method: "POST",
      body: { instance_url: instanceUrl },
    });
    if (res.authorize_url) {
      window.location.href = res.authorize_url;
    }
  } catch (err) {
    feedbackEl.className = "composer-feedback error";
    feedbackEl.textContent = err.message;
  }
}

async function verifyAndSaveDirectToken() {
  const instanceUrl =
    document.getElementById("mastodon-instance-url")?.value.trim() ||
    "https://mastodon.social";
  const accessToken =
    document.getElementById("mastodon-access-token")?.value.trim() || "";
  const feedbackEl = document.getElementById("mastodon-auth-feedback");

  if (!accessToken) {
    feedbackEl.className = "composer-feedback error";
    feedbackEl.textContent = "Please paste a valid Mastodon access token.";
    feedbackEl.classList.remove("hidden");
    return;
  }

  feedbackEl.className = "composer-feedback info";
  feedbackEl.textContent = `Verifying token with ${instanceUrl}...`;
  feedbackEl.classList.remove("hidden");

  try {
    const res = await apiRequest("/api/mastodon/oauth/token", {
      method: "POST",
      body: { instance_url: instanceUrl, access_token: accessToken },
    });
    applyConfigToUI(res.config);
    feedbackEl.className = "composer-feedback success";
    feedbackEl.textContent = `Connected as @${res.account?.acct}!`;
    announce(`Connected to Mastodon as ${res.account?.acct}.`);
    loadTimeline("home");
  } catch (err) {
    feedbackEl.className = "composer-feedback error";
    feedbackEl.textContent = err.message;
  }
}

async function logoutMastodon() {
  if (!window.confirm("Disconnect your Mastodon account from FediCaption?")) {
    return;
  }
  const res = await apiRequest("/api/mastodon/logout", { method: "POST" });
  applyConfigToUI(res.config);
  announce("Disconnected from Mastodon.");
  loadTimeline("local");
}

function updateCharCounter() {
  const text = document.getElementById("composer-textarea")?.value || "";
  const cw = state.cwEnabled
    ? document.getElementById("cw-input")?.value || ""
    : "";
  const maxChars = getMaxPostChars();
  const remaining = maxChars - (text.length + cw.length);
  const counterEl = document.getElementById("char-counter");
  if (counterEl) {
    counterEl.textContent = String(remaining);
    counterEl.title = `Server maximum post length: ${maxChars} characters (${remaining} remaining)`;
    counterEl.setAttribute(
      "aria-label",
      `${remaining} characters remaining out of ${maxChars}`
    );
    counterEl.classList.toggle("over-limit", remaining < 0);
  }
}

/* ==========================================================================
   Event Listeners & Initialization
   ========================================================================== */

function bindEvents() {
  // Header & Settings Modal
  document
    .getElementById("settings-button")
    ?.addEventListener("click", () => openSettingsModal("tab-ai"));
  document
    .getElementById("ai-status-pill")
    ?.addEventListener("click", () => openSettingsModal("tab-ai"));
  document
    .getElementById("connect-account-btn")
    ?.addEventListener("click", () => openSettingsModal("tab-mastodon"));
  document
    .getElementById("close-settings-btn")
    ?.addEventListener("click", closeSettingsModal);
  document
    .getElementById("save-settings-btn")
    ?.addEventListener("click", saveSettingsFromModal);

  document.querySelectorAll(".modal-tab").forEach((btn) => {
    btn.addEventListener("click", () => switchSettingsTab(btn.dataset.tab));
  });

  document
    .querySelectorAll('input[name="settings-ai-provider"]')
    .forEach((radio) => {
      radio.addEventListener("change", () =>
        updateVisibleProviderConfigBox(radio.value)
      );
    });

  document
    .getElementById("detect-ollama-btn")
    ?.addEventListener("click", detectLocalOllamaModels);
  document
    .getElementById("start-oauth-btn")
    ?.addEventListener("click", startMastodonOAuth);
  document
    .getElementById("save-token-btn")
    ?.addEventListener("click", verifyAndSaveDirectToken);
  document
    .getElementById("account-profile-btn")
    ?.addEventListener("click", () => {
      loadTimeline("mine", false);
      announce("Showing your own published posts.");
    });
  document
    .getElementById("logout-btn")
    ?.addEventListener("click", logoutMastodon);
  document
    .getElementById("refresh-account-btn")
    ?.addEventListener("click", async () => {
      try {
        await apiRequest("/api/mastodon/verify");
        const cfgRes = await apiRequest("/api/config");
        applyConfigToUI(cfgRes.config);
        announce("Account verified.");
      } catch (err) {
        alert(err.message);
      }
    });

  // Theme switcher
  document
    .getElementById("theme-select")
    ?.addEventListener("change", async (e) => {
      const theme = e.target.value;
      document.documentElement.setAttribute("data-theme", theme);
      const res = await apiRequest("/api/config", {
        method: "POST",
        body: { ui: { theme } },
      });
      applyConfigToUI(res.config);
    });

  // Quick sidebar AI controls
  document
    .getElementById("quick-provider-select")
    ?.addEventListener("change", async (e) => {
      const provider = e.target.value;
      const res = await apiRequest("/api/config", {
        method: "POST",
        body: { ai: { provider } },
      });
      applyConfigToUI(res.config);
      announce(`Active AI provider switched to ${provider}.`);
    });

  document
    .getElementById("quick-detail-select")
    ?.addEventListener("change", async (e) => {
      const detail_level = e.target.value;
      const res = await apiRequest("/api/config", {
        method: "POST",
        body: { ai: { detail_level } },
      });
      applyConfigToUI(res.config);
    });

  document
    .getElementById("quick-auto-scan")
    ?.addEventListener("change", async (e) => {
      const auto_generate = e.target.checked;
      const res = await apiRequest("/api/config", {
        method: "POST",
        body: { ai: { auto_generate } },
      });
      applyConfigToUI(res.config);
    });

  // Timeline navigation
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      loadTimeline(btn.dataset.timeline, false);
    });
  });

  document
    .getElementById("refresh-timeline-btn")
    ?.addEventListener("click", () => loadTimeline(state.currentTimeline, false));
  document
    .getElementById("load-more-btn")
    ?.addEventListener("click", () => loadTimeline(state.currentTimeline, true));
  document
    .getElementById("back-from-user-btn")
    ?.addEventListener("click", () =>
      loadTimeline(state.previousTimeline || "home", false)
    );
  document
    .getElementById("header-subscribe-btn")
    ?.addEventListener("click", (e) => {
      const subAction = e.currentTarget.dataset.subAction || "follow";
      toggleUserSubscription(subAction);
    });
  document
    .getElementById("user-profile-banner")
    ?.addEventListener("click", (e) => {
      const btn = e.target.closest('button[data-action="subscribe-user"]');
      if (!btn) return;
      toggleUserSubscription(btn.dataset.subAction || "follow");
    });

  // Composer controls
  const textarea = document.getElementById("composer-textarea");
  textarea?.addEventListener("input", updateCharCounter);
  document
    .getElementById("cw-input")
    ?.addEventListener("input", updateCharCounter);

  document.getElementById("toggle-cw-btn")?.addEventListener("click", (e) => {
    state.cwEnabled = !state.cwEnabled;
    e.currentTarget.setAttribute("aria-pressed", String(state.cwEnabled));
    document
      .getElementById("cw-container")
      ?.classList.toggle("hidden", !state.cwEnabled);
    updateCharCounter();
  });

  document
    .getElementById("cancel-reply-btn")
    ?.addEventListener("click", clearReplyContext);
  document
    .getElementById("cancel-edit-btn")
    ?.addEventListener("click", () => {
      clearEditContext();
      document.getElementById("composer-textarea").value = "";
      document.getElementById("cw-input").value = "";
      state.attachments.forEach((a) => {
        if (a.previewUrl && !a.existingMediaId) URL.revokeObjectURL(a.previewUrl);
      });
      state.attachments = [];
      renderAttachments();
      updateCharCounter();
      announce("Canceled editing post.");
    });
  document
    .getElementById("scan-all-media-btn")
    ?.addEventListener("click", scanAllAttachmentsWithAI);
  document
    .getElementById("publish-post-btn")
    ?.addEventListener("click", () => publishPost(false));

  // Media file picker, drag-and-drop, and clipboard paste
  const dropzone = document.getElementById("media-dropzone");
  const fileInput = document.getElementById("media-file-input");

  dropzone?.addEventListener("click", (e) => {
    if (e.target !== fileInput) {
      fileInput?.click();
    }
  });
  dropzone?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput?.click();
    }
  });
  fileInput?.addEventListener("change", (e) => {
    handleFilesSelected(e.target.files);
    fileInput.value = "";
  });

  dropzone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone?.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });
  dropzone?.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer?.files?.length) {
      handleFilesSelected(e.dataTransfer.files);
    }
  });

  window.addEventListener("paste", (e) => {
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      handleFilesSelected(files);
    }
  });

  // Delegated events inside media attachments list
  const attList = document.getElementById("media-attachments-list");
  attList?.addEventListener("input", (e) => {
    const attId = e.target.dataset?.attInput;
    if (!attId) return;
    const att = state.attachments.find((a) => a.id === attId);
    if (!att) return;
    att.altText = e.target.value;
    att.status = att.altText.trim() ? "ready" : "missing";
    const countEl = document.getElementById(`alt-count-${attId}`);
    if (countEl) {
      countEl.textContent = `${att.altText.length} / ${getAltTextLimit()} chars`;
    }
  });

  attList?.addEventListener("change", (e) => {
    if (e.target.dataset?.action === "change-att-detail") {
      const attId = e.target.dataset.attId;
      const att = state.attachments.find((a) => a.id === attId);
      if (att) {
        att.detailLevel = e.target.value;
      }
    }
  });

  attList?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const attId = btn.dataset.attId;
    if (action === "rescan-att") {
      generateAltTextForAttachment(attId);
    } else if (action === "remove-att") {
      removeAttachment(attId);
    } else if (action === "speak-alt") {
      const att = state.attachments.find((a) => a.id === attId);
      if (att) speakTextAloud(att.altText);
    }
  });

  // Delegated events inside timeline feed & thread modal
  const handleStatusFeedClick = (e) => {
    const btn = e.target.closest("button[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    const statusId = btn.dataset.statusId;

    if (action === "view-user") {
      const accountId = btn.dataset.accountId;
      if (accountId) {
        viewUserPosts(accountId, {
          displayName: btn.dataset.accountName,
          acct: btn.dataset.accountAcct,
          avatar: btn.dataset.accountAvatar,
        });
      }
    } else if (action === "reply") {
      startReplyToStatus(statusId);
    } else if (action === "edit") {
      startEditStatus(statusId);
    } else if (["reblog", "favourite", "bookmark", "delete"].includes(action)) {
      const isActive = btn.dataset.active === "true";
      performStatusAction(statusId, action, isActive);
    } else if (action === "thread") {
      openThreadModal(statusId);
    } else if (action === "toggle-status-cw") {
      document
        .getElementById(`status-body-${statusId}`)
        ?.classList.toggle("hidden");
    } else if (action === "speak-timeline-alt") {
      speakTextAloud(btn.dataset.alt || "");
    } else if (action === "caption-timeline-media") {
      captionTimelineMediaWithAI(
        btn.dataset.mediaUrl,
        btn.dataset.mediaType,
        btn.dataset.targetBox,
        btn
      );
    }
  };

  document
    .getElementById("timeline-feed")
    ?.addEventListener("click", handleStatusFeedClick);
  document
    .getElementById("thread-content")
    ?.addEventListener("click", handleStatusFeedClick);

  document.getElementById("close-thread-btn")?.addEventListener("click", () => {
    document.getElementById("thread-modal")?.classList.add("hidden");
  });

  // Global keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeSettingsModal();
      document.getElementById("thread-modal")?.classList.add("hidden");
    }
    if (e.altKey && e.key.toLowerCase() === "n") {
      e.preventDefault();
      document.getElementById("composer-textarea")?.focus();
      announce("Focused post composer.");
    }
    if (e.altKey && e.key.toLowerCase() === "a") {
      e.preventDefault();
      if (state.attachments.length > 0) {
        scanAllAttachmentsWithAI();
      }
    }
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      const active = document.activeElement;
      if (
        active &&
        (active.id === "composer-textarea" ||
          active.classList.contains("alt-textarea"))
      ) {
        e.preventDefault();
        publishPost(false);
      }
    }
  });
}

async function initApp() {
  bindEvents();

  try {
    const res = await apiRequest("/api/config");
    applyConfigToUI(res.config);

    // Query the Mastodon server for its maximum post character limit & media limits
    refreshInstanceLimits();

    // Check if returning from OAuth redirect
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("oauth") === "success") {
      window.history.replaceState({}, document.title, "/");
      showComposerFeedback("Successfully signed in to Mastodon via OAuth!", "success");
      announce("Signed in to Mastodon successfully.");
    }

    // Load home timeline if authenticated, else public local timeline
    const initialTimeline = res.config?.mastodon?.access_token
      ? "home"
      : "local";
    await loadTimeline(initialTimeline, false);
  } catch (err) {
    console.error("Failed to initialize FediCaption:", err);
  }
}

document.addEventListener("DOMContentLoaded", initApp);
