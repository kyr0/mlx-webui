// Tab Switching Logic
function openTab(evt, tabName) {
    var i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
        tabcontent[i].classList.remove("active");
    }
    tablinks = document.getElementsByClassName("tab-btn");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    document.getElementById(tabName).classList.add("active");
    if (evt) {
        evt.currentTarget.className += " active";
    }
}

// Initialize default tab
document.addEventListener('DOMContentLoaded', () => {
    // Show chat tab by default
    document.getElementById("chat-tab").style.display = "block";

    // Also attach event listeners here to ensure elements exist
    attachEventListeners();
});

function attachEventListeners() {
    // Chat Elements
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');

    if (promptInput && sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
        promptInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Auto-resize textarea
        promptInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }

    // Embedding Elements
    const embedBtn = document.getElementById('embed-btn');
    const copyEmbeddingBtn = document.getElementById('copy-embedding-btn');

    if (embedBtn) {
        embedBtn.addEventListener('click', generateEmbedding);
    }

    if (copyEmbeddingBtn) {
        copyEmbeddingBtn.addEventListener('click', copyEmbedding);
    }
}

// Chat Logic
const chatHistory = document.getElementById('chat-history');
let messages = [{
    role: "system",
    content: "Du bist ein nett. Antworte in ein bis zwei Sätzen."
}];

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    msgDiv.textContent = text;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return msgDiv;
}

async function sendMessage() {
    const promptInput = document.getElementById('prompt-input');
    const sendBtn = document.getElementById('send-btn');
    const text = promptInput.value.trim();

    if (!text) return;

    appendMessage('user', text);
    promptInput.value = '';
    promptInput.style.height = 'auto';

    // Update messages history
    messages.push({ role: "user", content: text });

    // Disable input
    promptInput.disabled = true;
    sendBtn.disabled = true;

    try {
        // Prepare context: System message + last 6 messages
        let contextMessages = messages;
        if (messages.length > 7) {
            contextMessages = [messages[0], ...messages.slice(-6)];
        }

        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: "kyr0/aidana-slm-mlx",
                messages: contextMessages,
                stream: true
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessageDiv = appendMessage('assistant', '');
        let assistantText = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                    try {
                        const data = JSON.parse(line.slice(6));
                        const content = data.choices[0].message.content;
                        if (content) {
                            assistantText += content;
                            assistantMessageDiv.textContent = assistantText;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Error parsing chunk:', e);
                    }
                }
            }
        }

        // Update history
        messages.push({ role: "assistant", content: assistantText });

    } catch (error) {
        console.error('Error:', error);
        appendMessage('system', 'Error: Failed to send message');
    } finally {
        promptInput.disabled = false;
        sendBtn.disabled = false;
        promptInput.focus();
    }
}

// Embedding Logic
async function generateEmbedding() {
    const embeddingInput = document.getElementById('embedding-input');
    const embedBtn = document.getElementById('embed-btn');
    const embeddingResultContainer = document.getElementById('embedding-result-container');
    const embeddingResult = document.getElementById('embedding-result');

    const text = embeddingInput.value.trim();
    if (!text) return;

    embedBtn.disabled = true;
    embedBtn.textContent = "Embedding...";

    try {
        const response = await fetch('/v1/embeddings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                input: text,
                model: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
            })
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const data = await response.json();
        embeddingResult.textContent = JSON.stringify(data, null, 2);
        embeddingResultContainer.style.display = "block";
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to generate embedding');
    } finally {
        embedBtn.disabled = false;
        embedBtn.textContent = "Embed";
    }
}

function copyEmbedding() {
    const embeddingResult = document.getElementById('embedding-result');
    const copyEmbeddingBtn = document.getElementById('copy-embedding-btn');

    if (!embeddingResult || !embeddingResult.textContent) {
        console.error('No embedding result to copy');
        return;
    }

    const textToCopy = embeddingResult.textContent;

    // Check if clipboard API is available
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(textToCopy).then(() => {
            if (copyEmbeddingBtn) {
                const originalText = copyEmbeddingBtn.textContent;
                copyEmbeddingBtn.textContent = "Copied!";
                setTimeout(() => {
                    copyEmbeddingBtn.textContent = originalText;
                }, 2000);
            }
        }).catch(err => {
            console.error('Failed to copy:', err);
            alert('Failed to copy to clipboard');
        });
    } else {
        // Fallback for older browsers or non-secure contexts
        const textArea = document.createElement('textarea');
        textArea.value = textToCopy;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            if (copyEmbeddingBtn) {
                const originalText = copyEmbeddingBtn.textContent;
                copyEmbeddingBtn.textContent = "Copied!";
                setTimeout(() => {
                    copyEmbeddingBtn.textContent = originalText;
                }, 2000);
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
            alert('Failed to copy to clipboard');
        }
        document.body.removeChild(textArea);
    }
}
