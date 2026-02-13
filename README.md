# multimodal-rag
A Multimodal RAG system that processes text and images to generate context-aware answers. It uses document chunking, embeddings, and FAISS for semantic search, and integrates an LLM for accurate response generation.
# 📚 Multimodal RAG System

A production-ready Multimodal Retrieval-Augmented Generation (RAG) system built using Streamlit, FAISS, Sentence Transformers, and Groq LLM.

This application allows users to upload documents (PDF) and ask context-aware questions using vector search and LLM generation.

---

## 🚀 Features

- 📄 PDF Document Upload
- 🔎 Text Chunking with Overlap
- 🧠 Sentence-Transformer Embeddings
- 📦 FAISS Vector Database
- 🎯 Cross-Encoder Reranking
- 🤖 Groq LLM for Context-Aware Answers
- 🌐 Streamlit Web Interface
- ☁️ Deployable on Streamlit Cloud

---

## 🏗️ Project Structure

multimodal_rag/

│
├── app.py
├── config.py
├── requirements.txt
│
└── rag/
├── init.py
├── embeddings.py
├── retriever.py
├── reranker.py
├── chunking.py
├── llm.py
├── vision.py
