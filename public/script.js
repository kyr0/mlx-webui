const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

let messages = [{
    role: "system",
    content: "You are a terse assistant. Answer in a single sentence."
}];

function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.textContent = content;
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    messages.push({ role, content });
}

async function sendMessage() {
    const content = userInput.value.trim();
    if (!content) return;

    // Add user message
    addMessage('user', content);
    userInput.value = '';
    userInput.style.height = 'auto'; // Reset height

    // Disable input while loading
    userInput.disabled = true;
    sendBtn.disabled = true;

    try {
        // Prepare context: System message + last 6 messages (3 turns)
        let contextMessages = messages;
        if (messages.length > 7) {
            contextMessages = [messages[0], ...messages.slice(-6)];
        }

        const response = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model: "kyr0/aidana-slm-mlx",
                messages: contextMessages,
                temperature: 0.7,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Create a placeholder for the assistant's message
        addMessage('assistant', '');
        const assistantMsgDiv = chatHistory.lastElementChild;
        let fullContent = '';

        const startTime = Date.now();
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices[0].message.content;
                        if (content) {
                            fullContent += content;
                            assistantMsgDiv.textContent = fullContent;
                            chatHistory.scrollTop = chatHistory.scrollHeight;
                        }
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }
                }
            }
        }

        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);

        const timeDiv = document.createElement('div');
        timeDiv.style.fontSize = '0.8em';
        timeDiv.style.color = '#888';
        timeDiv.style.marginTop = '5px';
        timeDiv.textContent = `Generated in ${duration}s`;
        assistantMsgDiv.appendChild(timeDiv);

        // Update history with full message
        messages.push({ role: 'assistant', content: fullContent });

    } catch (error) {
        console.error('Error:', error);
        addMessage('assistant', `Error: ${error.message}`);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});
