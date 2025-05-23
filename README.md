﻿# Langchain_Basic
# 📚 PDF Q&A Assistant using LangChain, OpenAI, and Streamlit

This project is an AI-powered Q&A app that lets you ask questions about the contents of any PDF document via a public link. It downloads the PDF, reads its content, stores it in a vector database (FAISS), and uses an LLM (OpenAI/Groq) to answer questions about it.

---

## 🚀 Features

- 🔗 Input PDF via URL  
- 📄 Load and split PDF into chunks  
- 🧠 Embed using OpenAI Embeddings  
- 📦 Store in FAISS vector store  
- 🤖 Answer questions using GPT (via LangChain)  
- 🧹 Auto-deletes temporary files  
- 🌐 Streamlit interface ready (optional)

---

## 🧠 Tech Stack

- `LangChain`  
- `OpenAI` & `langchain-openai`  
- `FAISS` (local vector store)  
- `Streamlit` (UI)  
- `FastAPI` & `LangServe` (optional API)  
- `dotenv`, `requests`, `PyPDF` for PDF handling

---

## 📦 Installation

1. **Clone the repo:**

```bash
git clone https://github.com/MetalCloth/Langchain_Basic.git
cd Langchain_Basic
