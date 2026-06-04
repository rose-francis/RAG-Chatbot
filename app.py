import os
import shutil
import gradio as gr
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.knowledge import Knowledge
from agno.knowledge.reader.pdf_reader import PDFReader
from agno.vectordb.lancedb.lance_db import LanceDb
from agno.knowledge.embedder.ollama import OllamaEmbedder

class NomicEmbedder(OllamaEmbedder):
    def get_embedding(self, text: str) -> list:
        return super().get_embedding(f"search_document: {text}")

    def get_query_embedding(self, text: str) -> list:
        return super().get_embedding(f"search_query: {text}")

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "tmp", "lancedb")

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

vector_db = LanceDb(
    table_name="my_pdf_docs",
    uri=DB_DIR,
    embedder=NomicEmbedder(id="nomic-embed-text", dimensions=768),
)

knowledge_base = Knowledge(
    vector_db=vector_db,
    readers={".pdf": PDFReader()},
)

agent = Agent(
    model=Groq(id="llama-3.1-8b-instant"),
    markdown=True,
    instructions="Answer ONLY using the context provided to you. If the answer is not in the context, say 'I don't know'. Do not use any outside knowledge.",
)


def respond(message, history):
    if not message.strip():
        yield "", history or []
        return

    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    results = knowledge_base.search(query=message, max_results=5)

    if results:
        context = "\n\n".join(r.content for r in results)
    else:
        context = "No relevant content found."

    prompt = f"""Using ONLY the context below, answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {message}
Answer:"""

    buffer = ""
    for chunk in agent.run(prompt, stream=True):
        if chunk.content:
            buffer += chunk.content
            history[-1]["content"] += chunk.content
            if len(buffer) >= 15:
                yield "", history
                buffer = ""
    if buffer:
        yield "", history


def upload_pdfs(files):
    if not files:
        return "No files selected."
    messages = []
    for file in files:
        filename = os.path.basename(file.name)
        dest = os.path.join(DATA_DIR, filename)
        shutil.copy(file.name, dest)
        knowledge_base.insert(path=dest, skip_if_exists=True)
        messages.append(f"Indexed: {filename}")
    return "\n".join(messages)


with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("# RAG Chatbot\nAsk questions about your PDF documents.")

    with gr.Row():
        with gr.Column(scale=1, min_width=240):
            gr.Markdown("### Upload PDFs")
            file_input = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Select PDF files",
            )
            upload_status = gr.Textbox(label="Status", interactive=False, lines=4)
            file_input.upload(fn=upload_pdfs, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=460, show_label=False, buttons=["copy", "copy_all"])

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    show_label=False,
                    scale=5,
                    submit_btn=False,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

    msg_input.submit(respond, [msg_input, chatbot], [msg_input, chatbot])
    send_btn.click(respond, [msg_input, chatbot], [msg_input, chatbot])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), footer_links=[])
