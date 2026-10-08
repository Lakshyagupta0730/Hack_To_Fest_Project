// Paper Explorer Client Controller

// Dynamic API Base URL Discovery
let API_BASE = "http://127.0.0.1:8000";
let isBackendConnected = false;

function getCandidateEndpoints() {
    const list = [];
    const origin = window.location.origin;
    const port = window.location.port;

    // 1. If served directly by FastAPI (port 8000 or production deployment)
    if (origin && origin.startsWith("http")) {
        if (port === "8000" || (!port || port === "80" || port === "443")) {
            list.push(origin);
        }
    }

    // 2. Standard local backend endpoints
    list.push("http://127.0.0.1:8000");
    list.push("http://localhost:8000");

    // 3. Fallback to origin if not already added
    if (origin && origin.startsWith("http") && !list.includes(origin)) {
        list.push(origin);
    }

    return list;
}

let currentPaperId = null;
let currentPaperData = null;
let isProcessing = false;

// Sample Paper Descriptions
const SAMPLE_INFO = {
    transformer: "Introduced the Transformer architecture, replacing RNNs with Multi-Head Self-Attention for sequence modeling.",
    resnet: "Introduced Deep Residual Learning with skip connections, solving the degradation problem and enabling training of networks over 100 layers deep.",
    ddpm: "Showed that diffusion probabilistic models can synthesize high quality images, establishing deep connections to denoising score matching."
};

// DOM Elements
const backendStatusBadge = document.getElementById("backend-status");
const activePaperBadge = document.getElementById("active-paper-badge");
const btnApiKey = document.getElementById("btn-api-key");
const sampleSelect = document.getElementById("sample-select");
const sampleDesc = document.getElementById("sample-desc");
const btnLoadSample = document.getElementById("btn-load-sample");
const loadSampleText = document.getElementById("load-sample-text");

const dropZone = document.getElementById("drop-zone");
const pdfFileInput = document.getElementById("pdf-file-input");
const selectedFileName = document.getElementById("selected-file-name");
const btnUploadFile = document.getElementById("btn-upload-file");
const uploadFileText = document.getElementById("upload-file-text");

const ingestionStatusBox = document.getElementById("ingestion-status-box");
const ingestionStepTitle = document.getElementById("ingestion-step-title");
const ingestionStepDesc = document.getElementById("ingestion-step-desc");

const sectionOverview = document.getElementById("section-overview");
const currentPaperTitle = document.getElementById("current-paper-title");
const statPages = document.getElementById("stat-pages");
const statChunks = document.getElementById("stat-chunks");
const statSections = document.getElementById("stat-sections");
const sectionsPillContainer = document.getElementById("sections-pill-container");
const suggestedQuestionsContainer = document.getElementById("suggested-questions-container");

const chatStream = document.getElementById("chat-stream");
const qaForm = document.getElementById("qa-form");
const userQuestionInput = document.getElementById("user-question");
const btnSubmitQuestion = document.getElementById("btn-submit-question");
const submitBtnText = document.getElementById("submit-btn-text");
const btnClearChat = document.getElementById("btn-clear-chat");
const toast = document.getElementById("toast");

// Toast Notification
function showToast(message, isError = false) {
    if (!toast) return;
    toast.textContent = message;
    toast.style.background = isError ? "#ef4444" : "#1e293b";
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

// Check Backend Health with multi-endpoint probing
async function checkHealth() {
    const candidates = getCandidateEndpoints();
    const testList = [API_BASE, ...candidates.filter(c => c !== API_BASE)];
    let connected = false;
    let healthyData = null;

    for (const endpoint of testList) {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2500);
            const res = await fetch(`${endpoint}/health`, { signal: controller.signal });
            clearTimeout(timeoutId);

            if (res.ok) {
                healthyData = await res.json();
                API_BASE = endpoint;
                connected = true;
                break;
            }
        } catch (e) {
            // Next candidate
        }
    }

    if (connected && healthyData && backendStatusBadge) {
        isBackendConnected = true;
        backendStatusBadge.className = "badge badge-green";
        const engine = healthyData.dense_embeddings_active ? "ChromaDB" : "FastBM25";
        backendStatusBadge.innerHTML = `<span class="pulse-dot"></span> Backend Connected (${engine})`;
        backendStatusBadge.title = `Connected to ${API_BASE} (${healthyData.total_chunks_indexed} chunks indexed)`;
    } else if (backendStatusBadge) {
        isBackendConnected = false;
        backendStatusBadge.className = "badge badge-amber";
        backendStatusBadge.innerHTML = `⚠️ Backend Connecting... <button id="retry-health-btn" style="background:none; border:none; color:#b45309; text-decoration:underline; cursor:pointer; font-size:0.75rem; padding:0 0 0 4px;">Retry</button>`;
        const retryBtn = document.getElementById("retry-health-btn");
        if (retryBtn) {
            retryBtn.onclick = (e) => {
                e.stopPropagation();
                checkHealth();
            };
        }
    }
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    setInterval(checkHealth, 15000);

    // API Key Configuration
    if (btnApiKey) {
        const updateApiKeyBadge = () => {
            const current = localStorage.getItem("paper_explorer_api_key");
            if (current) {
                btnApiKey.textContent = "🔑 API Key Configured";
                btnApiKey.style.background = "#f3e8ff";
                btnApiKey.style.color = "#7e22ce";
                btnApiKey.style.borderColor = "rgba(147, 51, 234, 0.3)";
            } else {
                btnApiKey.textContent = "🔑 API Key (Optional)";
                btnApiKey.style.background = "#f1f5f9";
                btnApiKey.style.color = "#475569";
                btnApiKey.style.borderColor = "rgba(148, 163, 184, 0.3)";
            }
        };
        updateApiKeyBadge();

        btnApiKey.addEventListener("click", () => {
            const current = localStorage.getItem("paper_explorer_api_key") || "";
            const key = prompt(
                "Enter OpenRouter API Key (e.g. sk-or-v1-...):\n\n(Leave blank to use the built-in Grounded Local Vector Engine without external APIs):",
                current
            );
            if (key !== null) {
                if (key.trim()) {
                    localStorage.setItem("paper_explorer_api_key", key.trim());
                    showToast("API Key saved successfully.");
                } else {
                    localStorage.removeItem("paper_explorer_api_key");
                    showToast("Switched to Local Vector Engine mode.");
                }
                updateApiKeyBadge();
            }
        });
    }

    // Sample Selector change
    if (sampleSelect) {
        sampleSelect.addEventListener("change", (e) => {
            const val = e.target.value;
            if (SAMPLE_INFO[val]) {
                sampleDesc.textContent = SAMPLE_INFO[val];
            }
        });
    }

    // Load Benchmark Paper Click
    if (btnLoadSample) {
        btnLoadSample.addEventListener("click", handleLoadSample);
    }

    // Drop Zone Events
    if (dropZone && pdfFileInput) {
        dropZone.addEventListener("click", () => pdfFileInput.click());

        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        });

        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.toLowerCase().endsWith(".pdf")) {
                    pdfFileInput.files = e.dataTransfer.files;
                    handleFileSelected(file);
                } else {
                    showToast("Please drop a valid PDF file.", true);
                }
            }
        });

        pdfFileInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                handleFileSelected(e.target.files[0]);
            }
        });
    }

    // Upload & Index Button
    if (btnUploadFile) {
        btnUploadFile.addEventListener("click", handleUploadPdf);
    }

    // QA Form Submit
    if (qaForm) {
        qaForm.addEventListener("submit", (e) => {
            e.preventDefault();
            handleAskQuestion();
        });
    }

    // Keyboard shortcut in textarea: Enter submits, Shift+Enter new line
    if (userQuestionInput) {
        userQuestionInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleAskQuestion();
            }
        });
    }

    // Clear Chat
    if (btnClearChat) {
        btnClearChat.addEventListener("click", () => {
            chatStream.innerHTML = `
                <div class="chat-message-assistant">
                    <div class="msg-header">
                        <span>🤖 Paper Explorer Assistant</span>
                        <span style="font-size: 0.75rem; color: #94a3b8;">Reset</span>
                    </div>
                    <div class="markdown-body">
                        <p>Chat cleared. Ask any new question about the loaded paper!</p>
                    </div>
                </div>
            `;
            showToast("Conversation cleared.");
        });
    }
});

function handleFileSelected(file) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    selectedFileName.textContent = `Selected: ${file.name} (${sizeMb} MB)`;
    selectedFileName.style.color = "#1e40af";
    selectedFileName.style.fontWeight = "600";
    btnUploadFile.disabled = false;
}

// Ingestion UI helper
function setIngestionState(isActive, title = "", desc = "") {
    if (isActive) {
        ingestionStatusBox.style.display = "block";
        ingestionStepTitle.textContent = title;
        ingestionStepDesc.textContent = desc;
    } else {
        ingestionStatusBox.style.display = "none";
    }
}

// Load Benchmark Paper
async function handleLoadSample() {
    if (isProcessing) return;
    const sampleId = sampleSelect.value;
    
    isProcessing = true;
    btnLoadSample.disabled = true;
    loadSampleText.innerHTML = '<span class="spinner"></span> Loading Benchmark...';
    setIngestionState(true, "Loading Benchmark Paper...", "Reading PDF, generating ChromaDB embeddings...");

    try {
        const res = await fetch(`${API_BASE}/load-sample/${sampleId}`, { method: "POST" });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Failed to load sample paper.");
        }

        const data = await res.json();
        currentPaperId = data.paper_id;
        currentPaperData = data;

        updateActivePaperUI(data);
        showToast(`Loaded: ${data.title}`);
    } catch (err) {
        console.error(err);
        showToast(err.message || "Failed to load paper.", true);
    } finally {
        isProcessing = false;
        btnLoadSample.disabled = false;
        loadSampleText.textContent = "🚀 Load Benchmark Paper";
        setIngestionState(false);
    }
}

// Upload Custom PDF
async function handleUploadPdf() {
    if (isProcessing) return;
    const file = pdfFileInput.files[0];
    if (!file) {
        showToast("Please choose a PDF file first.", true);
        return;
    }

    isProcessing = true;
    btnUploadFile.disabled = true;
    uploadFileText.innerHTML = '<span class="spinner"></span> Uploading & Indexing...';
    setIngestionState(true, "Indexing Uploaded PDF...", "Extracting sections, computing sentence embeddings with all-MiniLM-L6-v2...");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_BASE}/upload-paper`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "PDF processing failed.");
        }

        const data = await res.json();
        currentPaperId = data.paper_id;
        currentPaperData = data;

        updateActivePaperUI(data);
        showToast(`Indexed: ${data.title}`);
    } catch (err) {
        console.error(err);
        showToast(err.message || "Failed to process PDF.", true);
    } finally {
        isProcessing = false;
        btnUploadFile.disabled = false;
        uploadFileText.textContent = "⚡ Upload & Index Paper";
        setIngestionState(false);
    }
}

// Update UI with Active Paper metadata
function updateActivePaperUI(data) {
    activePaperBadge.className = "badge badge-green";
    activePaperBadge.innerHTML = `📄 Active: ${data.title.substring(0, 32)}${data.title.length > 32 ? '...' : ''}`;

    sectionOverview.style.display = "block";
    currentPaperTitle.textContent = data.title;
    statPages.textContent = `📄 ${data.total_pages} Pages`;
    statChunks.textContent = `🧩 ${data.total_chunks} Chunks`;
    statSections.textContent = `📑 ${data.sections ? data.sections.length : 0} Sections`;

    // Render Section Pills
    sectionsPillContainer.innerHTML = "";
    if (data.sections && data.sections.length > 0) {
        data.sections.forEach(sec => {
            const pill = document.createElement("span");
            pill.className = "section-pill";
            pill.textContent = sec;
            pill.title = "Click to ask about this section";
            pill.style.cursor = "pointer";
            pill.addEventListener("click", () => {
                userQuestionInput.value = `Explain the methodology and key findings in section: ${sec}`;
                userQuestionInput.focus();
            });
            sectionsPillContainer.appendChild(pill);
        });
    } else {
        sectionsPillContainer.innerHTML = "<p style='font-size:0.85rem; color:#64748b;'>Document structure indexed directly.</p>";
    }

    // Render Suggested Questions
    suggestedQuestionsContainer.innerHTML = "";
    if (data.sample_questions && data.sample_questions.length > 0) {
        data.sample_questions.forEach(qText => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "prompt-chip";
            chip.textContent = qText;
            chip.addEventListener("click", () => {
                userQuestionInput.value = qText;
                handleAskQuestion();
            });
            suggestedQuestionsContainer.appendChild(chip);
        });
    }

    // Add assistant announcement in chat
    appendAssistantMessage(`
**Loaded Paper:** *${data.title}*  
- **Pages:** ${data.total_pages} | **Vector Chunks:** ${data.total_chunks} | **Embedding:** all-MiniLM-L6-v2  
- **ChromaDB Index:** Ready for grounded semantic retrieval.  

Feel free to click any of the suggested questions above or ask your own question below!
    `);
}

// Render markdown safely
function renderMarkdown(content) {
    if (window.marked && typeof window.marked.parse === "function") {
        return window.marked.parse(content);
    }
    // Fallback basic formatter
    return content
        .replace(/\n\n/g, "<br><br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.*?)\*/g, "<em>$1</em>");
}

// Append User Message to Chat
function appendUserMessage(text) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message-user";
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    msgDiv.innerHTML = `
        <div class="msg-header">
            <span>👤 Researcher</span>
            <span style="font-size: 0.75rem; color: #475569;">${timeStr}</span>
        </div>
        <div class="markdown-body">
            <p>${escapeHtml(text)}</p>
        </div>
    `;
    chatStream.appendChild(msgDiv);
    chatStream.scrollTop = chatStream.scrollHeight;
}

// Append Assistant Message to Chat
function appendAssistantMessage(text, citations = [], modelUsed = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = "chat-message-assistant";
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const modelTag = modelUsed ? `<span style="font-size:0.75rem; color:#64748b; margin-left:8px;">(${modelUsed})</span>` : "";

    let citationsHtml = "";
    if (citations && citations.length > 0) {
        let citationPills = citations.map((c, idx) => {
            const pageInfo = c.page ? `Page ${c.page}` : "Ref";
            return `
                <button type="button" class="citation-pill" data-idx="${idx}">
                    📌 ${pageInfo} · ${escapeHtml(c.section || "General")}
                </button>
            `;
        }).join("");

        citationsHtml = `
            <div class="citations-wrapper">
                <div class="citations-title">
                    <span>📑 Verified Grounded Citations (Click to inspect source excerpt):</span>
                </div>
                <div class="citations-list">
                    ${citationPills}
                </div>
                <div class="citation-content-box" id="citation-box-${Date.now()}"></div>
            </div>
        `;
    }

    msgDiv.innerHTML = `
        <div class="msg-header">
            <span>🤖 Paper Explorer Assistant ${modelTag}</span>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span style="font-size: 0.75rem; color: #94a3b8;">${timeStr}</span>
                <button type="button" class="btn-sm btn-secondary copy-btn" title="Copy answer">📋 Copy</button>
            </div>
        </div>
        <div class="markdown-body">
            ${renderMarkdown(text)}
        </div>
        ${citationsHtml}
    `;

    // Wire up citation pill click to show source text
    if (citations && citations.length > 0) {
        const previewBox = msgDiv.querySelector(".citation-content-box");
        const pills = msgDiv.querySelectorAll(".citation-pill");
        pills.forEach(pill => {
            pill.addEventListener("click", () => {
                const idx = parseInt(pill.getAttribute("data-idx"));
                const citation = citations[idx];
                if (previewBox) {
                    if (previewBox.style.display === "block" && previewBox.getAttribute("data-active-idx") == idx) {
                        previewBox.style.display = "none";
                    } else {
                        previewBox.setAttribute("data-active-idx", idx);
                        previewBox.style.display = "block";
                        previewBox.innerHTML = `
                            <strong>Source Excerpt [Page ${citation.page || 'N/A'}, Section: ${escapeHtml(citation.section || 'General')}]:</strong><br>
                            "${escapeHtml(citation.content)}"
                        `;
                    }
                }
            });
        });
    }

    // Wire up copy button
    const copyBtn = msgDiv.querySelector(".copy-btn");
    if (copyBtn) {
        copyBtn.addEventListener("click", () => {
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.textContent = "✅ Copied!";
                setTimeout(() => { copyBtn.textContent = "📋 Copy"; }, 2000);
            });
        });
    }

    chatStream.appendChild(msgDiv);
    chatStream.scrollTop = chatStream.scrollHeight;
}

// Ask Question Handler
async function handleAskQuestion() {
    const question = userQuestionInput.value.trim();
    if (!question) {
        showToast("Please enter a question.", true);
        return;
    }

    if (!currentPaperId) {
        showToast("Please load a benchmark paper or upload a PDF first!", true);
        return;
    }

    if (isProcessing) return;

    // Append user query
    appendUserMessage(question);
    userQuestionInput.value = "";

    // Show temporary thinking message
    const thinkingId = "thinking-" + Date.now();
    const thinkingDiv = document.createElement("div");
    thinkingDiv.id = thinkingId;
    thinkingDiv.className = "chat-message-assistant";
    thinkingDiv.innerHTML = `
        <div class="msg-header">
            <span>🤖 Paper Explorer Assistant</span>
            <span style="font-size: 0.75rem; color: #94a3b8;">Searching embeddings...</span>
        </div>
        <div class="markdown-body" style="display: flex; align-items: center; gap: 8px;">
            <span class="spinner spinner-dark"></span>
            <span style="color: #64748b;">Retrieving top ChromaDB chunks and synthesizing answer with OpenRouter...</span>
        </div>
    `;
    chatStream.appendChild(thinkingDiv);
    chatStream.scrollTop = chatStream.scrollHeight;

    isProcessing = true;
    btnSubmitQuestion.disabled = true;
    submitBtnText.innerHTML = '<span class="spinner"></span> Thinking...';

    try {
        const storedApiKey = localStorage.getItem("paper_explorer_api_key") || "";
        const payload = {
            paper_id: currentPaperId,
            question: question
        };
        if (storedApiKey) {
            payload.api_key = storedApiKey;
        }

        const res = await fetch(`${API_BASE}/ask-question`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        // Remove thinking message
        const thinkingElem = document.getElementById(thinkingId);
        if (thinkingElem) thinkingElem.remove();

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || "Error generating answer.");
        }

        const data = await res.json();
        appendAssistantMessage(data.answer, data.citations, data.model_used);
    } catch (err) {
        console.error(err);
        const thinkingElem = document.getElementById(thinkingId);
        if (thinkingElem) thinkingElem.remove();

        appendAssistantMessage(`❌ **Error:** ${err.message || "Failed to retrieve an answer. Please verify the backend connection and try again."}`);
        showToast(err.message || "Request failed.", true);
    } finally {
        isProcessing = false;
        btnSubmitQuestion.disabled = false;
        submitBtnText.textContent = "💬 Ask Assistant";
        userQuestionInput.focus();
    }
}

// Utility HTML escape
function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}