// ── READ: load the document list ──────────────────────────────────────────────
async function loadDocuments() {
  const res = await fetch("/documents");
  const docs = await res.json();
  const tbody = document.getElementById("doc-list");

  if (docs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-msg">No documents yet. Upload a PDF to get started.</td></tr>';
    return;
  }

  tbody.innerHTML = docs.map(d => `
    <tr>
      <td>${d.id}</td>
      <td>${d.filename}</td>
      <td>${d.upload_date}</td>
      <td>${(d.file_size_bytes / 1024).toFixed(1)} KB</td>
      <td>${d.page_count}</td>
      <td style="display:flex;gap:4px;">
        <button class="secondary" onclick="renameDoc(${d.id}, '${d.filename}', this)">Rename</button>
        <button class="danger"    onclick="deleteDoc(${d.id}, '${d.filename}')">Delete</button>
      </td>
    </tr>
  `).join("");
}


// ── CREATE: upload PDFs ────────────────────────────────────────────────────────
async function uploadFiles() {
  const fileInput = document.getElementById("file-input");
  const status    = document.getElementById("upload-status");
  const files     = fileInput.files;

  if (!files.length) {
    status.textContent = "Please select at least one PDF file.";
    return;
  }

  status.textContent = "Uploading…";

  const form = new FormData();
  for (const f of files) form.append("files", f);

  const res  = await fetch("/documents", { method: "POST", body: form });
  const data = await res.json();

  status.textContent = `Uploaded: ${data.uploaded.map(d => d.filename).join(", ")}`;
  fileInput.value = "";
  loadDocuments();
}


// ── UPDATE: rename a document ──────────────────────────────────────────────────
async function renameDoc(id, currentName, btn) {
  const newName = prompt("Enter a new filename:", currentName);
  if (!newName || newName.trim() === currentName) return;

  btn.textContent = "Renaming…";
  btn.disabled = true;

  await fetch(`/documents/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: newName.trim() }),
  });

  loadDocuments();   // refresh table
}


// ── DELETE: remove a document ──────────────────────────────────────────────────
async function deleteDoc(id, filename) {
  if (!confirm(`Delete "${filename}"? This will also remove it from the knowledge base.`)) return;

  await fetch(`/documents/${id}`, { method: "DELETE" });  // DELETE /documents/{id}  →  DELETE

  loadDocuments();   // refresh table
}


// ── CHAT ───────────────────────────────────────────────────────────────────────
async function sendMessage() {
  const input   = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;

  addBubble("user", message);
  const thinking = addBubble("thinking", "Thinking…");

  const res  = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await res.json();

  thinking.remove();
  addBubble("assistant", data.reply);

  input.disabled = false;
  sendBtn.disabled = false;
  input.focus();
}

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  const history = document.getElementById("chat-history");
  history.appendChild(div);
  div.scrollIntoView({ behavior: "smooth" });
  return div;
}


// Load documents when the page first opens
loadDocuments();
