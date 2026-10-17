
const API_URL = "http://127.0.0.1:8000/ask";

const form = document.getElementById("question-form");
const input = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");

function addMessage(role, text, sources = []) {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "message-label";
    label.textContent = role === "user" ? "You" : "✦ AI Assistant";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    message.append(label, bubble);

    if (role === "assistant" && sources.length > 0) {
        const sourceContainer = document.createElement("div");
        sourceContainer.className = "sources";

        const heading = document.createElement("div");
        heading.textContent = "Sources";
        sourceContainer.appendChild(heading);

        sources.forEach(source => {
            const tag = document.createElement("span");
            tag.className = "source";
            tag.textContent = "📄 " + source;
            sourceContainer.appendChild(tag);
        });

        message.appendChild(sourceContainer);
    }

    chat.appendChild(message);
    chat.scrollTop = chat.scrollHeight;

    return message;
}

async function askQuestion(question) {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || sendButton.disabled) return;

    welcome.hidden = true;

    addMessage("user", trimmedQuestion);
    input.value = "";
    input.style.height = "auto";
    sendButton.disabled = true;
    input.disabled = true;

    const loadingMessage = addMessage(
        "assistant",
        "Searching your documents and generating an answer..."
    );

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: trimmedQuestion
            })
        });

        if (!response.ok) {
            throw new Error(`API returned HTTP ${response.status}`);
        }

        const data = await response.json();

        loadingMessage.remove();

        addMessage(
            "assistant",
            data.answer || "No answer was returned.",
            data.sources || []
        );
    } catch (error) {
        loadingMessage.remove();

        addMessage(
            "assistant",
            "Could not connect to the RAG backend. Make sure FastAPI and Ollama are running. Details: " +
                error.message
        );
    } finally {
        sendButton.disabled = false;
        input.disabled = false;
        input.focus();
    }
}

form.addEventListener("submit", event => {
    event.preventDefault();
    askQuestion(input.value);
});

input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 130) + "px";
});

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
    }
});

document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", () => {
        askQuestion(button.textContent);
    });
});
