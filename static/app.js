const messagesEl = document.getElementById('messages');
const formEl = document.getElementById('ask-form');
const inputEl = document.getElementById('question');

function addMsg(role, text){
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

formEl.addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = inputEl.value.trim();
  if(!q) return;
  addMsg('user', q);
  inputEl.value = '';

  const thinking = document.createElement('div');
  thinking.className = 'msg system';
  thinking.textContent = 'Thinking…';
  messagesEl.appendChild(thinking);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try{
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    messagesEl.removeChild(thinking);
    if(!data.ok){
      addMsg('bot', data.error || 'Error');
    } else {
      addMsg('bot', data.answer);
    }
  } catch(err){
    messagesEl.removeChild(thinking);
    addMsg('bot', 'Network error');
  }
});

// Optional: focus the input on load
inputEl.focus();
