# 📚 NCERT-AI: Intelligent Study Assistant for NCERT PDFs

NCERT-AI is a command-line based AI Agent that lets you interact with NCERT textbooks like never before. Ask questions, generate MCQ quizzes, or get summaries — all strictly based on the contents of the uploaded NCERT PDF.

---

## 🚀 Features

- ✅ **PDF Ingestion & Embedding**: Upload and embed NCERT PDFs into a vector database.
- ❓ **Ask Factual/Conceptual Questions**: Get accurate answers using Retrieval-Augmented Generation.
- 📝 **Topic Summarization**: Get simple, clear summaries of complex textbook topics.
- 🎯 **MCQ Quiz Generator**: Auto-generate interactive quizzes from any chapter or topic.
- ❌ **No Hallucinations**: If a topic isn’t found in the NCERT content, the agent says so.

---

## 🛠 Tech Stack

- **LangChain** (Tools, Agents, Chains)
- **OpenAI GPT-3.5 Turbo**
- **Chroma Vector DB**
- **PyPDFLoader**
- **Python + CLI Interface**

---

## 📂 Project Structure

```bash
.
├── agent.py                 # Main agent loop and CLI interface
├── requirements.txt         # Dependency list
├── Tools/
│   ├── ingestion.py         # Load + split + embed NCERT PDFs
│   ├── retrieval.py         # Q&A over embedded content
│   ├── summariser.py        # Topic summarizer using LLM + context
│   └── quiz.py              # Quiz generator with MCQs from content
└── db/                      # Auto-created vector store after ingestion
