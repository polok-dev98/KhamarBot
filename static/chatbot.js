let userId = localStorage.getItem("user_id");
let sessionId = localStorage.getItem("session_id");
let chatInitialized = false;

async function startSession() {
  const res = await fetch('/start', {
    method: "POST",
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId || null })
  });
  const data = await res.json();
  userId = data.user_id;
  sessionId = data.session_id;
  localStorage.setItem("user_id", userId);
  localStorage.setItem("session_id", sessionId);
}

async function sendMessage(message = null) {
  const input = document.getElementById("user-input");
  const userMessage = message || input.value.trim();
  if (!userMessage) return;

  appendMessage("You", userMessage, true);
  if (!message) input.value = "";

  showTyping();

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessage,
        session_id: sessionId,
        user_id: userId
      })
    });

    const data = await response.json();
    hideTyping();
    appendMessage("Bot", data.response, false);
  } catch (error) {
    hideTyping();
    appendMessage("Bot", "❌ Error contacting the bot.", false);
  }
}

function appendMessage(sender, message, isUser = false) {
  const chatBox = document.getElementById("chat-box");
  const msg = document.createElement("div");
  msg.className = `message ${isUser ? "user" : "bot"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${isUser ? "user" : "bot"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (!isUser) {
    let html = marked.parse(message);
    html = html.replace(/<pre><code[^>]*>/g, '').replace(/<\/code><\/pre>/g, '');
    bubble.innerHTML = html;
  } else {
    bubble.textContent = message;
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping() {
  const chatBox = document.getElementById("chat-box");
  const typingMsg = document.createElement("div");
  typingMsg.className = "message bot typing-indicator";
  typingMsg.innerHTML = `
    <div class="avatar bot"></div>
    <div class="bubble">
      <span></span><span></span><span></span>
    </div>
  `;
  chatBox.appendChild(typingMsg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function hideTyping() {
  const typingIndicator = document.querySelector(".typing-indicator");
  if (typingIndicator) typingIndicator.remove();
}

function updateClock() {
  const clockEl = document.getElementById("clock");
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  clockEl.textContent = timeString;
}
setInterval(updateClock, 1000);
updateClock();

// Show initial "Hi" button message
function showStartButton() {
  const chatBox = document.getElementById("chat-box");

  const container = document.createElement("div");
  container.className = "message bot";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  bubble.innerHTML = `
    👋 Hello! Click the button below to start chatting.<br><br>
    <button onclick="sendMessage('hi')" class="start-button">Hi</button>
  `;

  container.appendChild(avatar);
  container.appendChild(bubble);
  chatBox.appendChild(container);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Toggle Chatbox
document.getElementById("chat-button").addEventListener("click", async () => {
  const chatWindow = document.getElementById("chat-window");
  chatWindow.classList.toggle("hidden");

  if (!chatInitialized) {
    await startSession();
    showStartButton();
    chatInitialized = true;
  }
});

// Trigger sendMessage() when Enter key is pressed
document.getElementById("user-input").addEventListener("keypress", function (e) {
  if (e.key === "Enter") {
    sendMessage();
  }
});