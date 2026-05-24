const form = document.getElementById('chat-form');
const input = document.getElementById('question-input');
const chatWindow = document.getElementById('chat-window');
const submitBtn = document.getElementById('submit-btn');

function appendMessage(role, text) {
  const row = document.createElement('div');
  row.className = 'message-row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

if (form) {
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const q = (input.value || '').trim();
    if (!q) return;

    appendMessage('user', q);
    input.value = '';
    input.focus();
    submitBtn.disabled = true;

    try {
      const resp = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });
      const data = await resp.json();
      appendMessage('assistant', data.answer || 'I could not generate an answer.');
    } catch (err) {
      appendMessage('assistant', 'There was an error contacting the server.');
    } finally {
      submitBtn.disabled = false;
    }
  });
}
