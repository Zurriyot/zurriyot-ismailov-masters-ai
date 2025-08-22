---
title: AI Customer Support Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🤖 AI Customer Support Chatbot

An intelligent customer support solution powered by OpenAI that can answer questions from PDF documents and create GitHub support tickets.

[![Open in Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/customer-support-ai)

## ✨ Features

- **📚 Document Q&A**: Ask questions about your PDF documents and get AI-powered answers
- **🎫 Support Tickets**: Create GitHub issues for questions that can't be answered
- **🔍 Smart Search**: Vector-based document search with source citations
- **💬 Chat History**: Maintains conversation context throughout the session
- **🚀 Auto-processing**: Automatically loads and processes documents on startup
- **💾 Persistent Storage**: Vector database persists between sessions

## 🌐 Live Demo

**Try it now on Hugging Face Spaces!** Click the badge above to access the live application.

## 🔧 Configuration

This app requires the following environment variables to be set in your Hugging Face Space:

```bash
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here  # Optional: for support ticket creation
GITHUB_REPO=username/repository_name  # Optional: format username/repo
```

## 📚 Documents

The app comes with three sample PDF documents:
- `Learning_Python.pdf`
- `PDF Reference 1.0.pdf` 
- `Toyota Hilux Manual.pdf`

You can replace these with your own documents by uploading them to the `data/` folder.

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **AI**: OpenAI GPT models for text generation
- **Vector Database**: ChromaDB for document embeddings
- **Document Processing**: PyPDF2 with PyCryptodome support
- **GitHub Integration**: PyGithub for support ticket creation

## 🚀 Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys
4. Run: `streamlit run app.py`

## 📁 Project Structure

```
AI_Chat_bot/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration and settings
├── document_processor.py # PDF processing and vector storage
├── chat_manager.py       # Chat logic and AI responses
├── github_integration.py # GitHub API integration
├── requirements.txt      # Python dependencies
├── data/                # PDF documents folder
└── README.md            # This file
```

## 🔒 Security Notes

- Never commit your API keys to the repository
- Use Hugging Face Spaces' environment variables for sensitive data
- GitHub tokens should have minimal required permissions
- OpenAI API keys should be kept secure

## 📝 Usage

1. **Ask Questions**: Type questions in the chat about your documents
2. **Get Answers**: AI searches documents and provides answers with citations
3. **Create Tickets**: If no answer is found, create a support ticket
4. **Track Issues**: View recent tickets and their status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using Streamlit, OpenAI, and LangChain**