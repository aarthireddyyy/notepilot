const API_URL = "http://localhost:8000/api/ask";

const chat = document.getElementById("chat");
const input = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const themeToggle = document.getElementById("themeToggle");

function addMessage(text, type, sources = []) {
  const div = document.createElement("div");
  div.className = `message ${type}`;
  div.innerHTML = `<p>${text}</p>`;

  if (sources.length > 0) {
    const src = document.createElement("div");
    src.className = "sources";
    src.innerHTML = "<strong>Sources:</strong><br>" + sources.join("<br>");
    div.appendChild(src);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

sendBtn.onclick = async () => {
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";

  addMessage("Thinking...", "bot");

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    chat.lastChild.remove();

    addMessage(data.answer, "bot", data.sources || []);
  } catch (err) {
    chat.lastChild.remove();
    addMessage("Error connecting to backend.", "bot");
  }
};

themeToggle.onclick = () => {
  document.body.classList.toggle("dark");
};
