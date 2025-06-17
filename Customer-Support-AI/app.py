# Customer Support AI Solution
# Complete system with document search, chat interface, and GitHub ticket creation

import streamlit as st
import openai
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List, Dict, Tuple
import base64
from datetime import datetime
import pickle
import tempfile
import io
import glob
from pathlib import Path
import time

# Configuration
class Config:
    COMPANY_NAME = "TechCorp Solutions"
    COMPANY_EMAIL = "support@techcorp.com"
    COMPANY_PHONE = "+1-555-TECH-HELP"
    OPENAI_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    MAX_CONTEXT_LENGTH = 4000
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

# Set cache directories for HuggingFace models
os.environ['HF_HOME'] = os.path.join(os.getcwd(), '.cache', 'huggingface')
os.environ['TORCH_HOME'] = os.path.join(os.getcwd(), '.cache', 'torch')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.getcwd(), '.cache', 'huggingface', 'transformers')

# Create cache directories if they don't exist
for cache_dir in [os.environ['HF_HOME'], os.environ['TORCH_HOME'], os.environ['TRANSFORMERS_CACHE']]:
    os.makedirs(cache_dir, exist_ok=True)

class DocumentProcessor:
    def __init__(self):
        try:
            # Set cache directory for sentence transformers
            cache_folder = os.path.join(os.getcwd(), '.cache', 'sentence_transformers')
            os.makedirs(cache_folder, exist_ok=True)
            
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, cache_folder=cache_folder)
        except Exception as e:
            st.error(f"Error initializing embedding model: {str(e)}")
            # Fallback: try without cache_folder parameter
            try:
                self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
            except Exception as e2:
                st.error(f"Failed to initialize embedding model: {str(e2)}")
                self.embedding_model = None
        
    def extract_text_from_pdf(self, pdf_file_path_or_stream, filename: str = None) -> List[Dict]:
        """Extract text from PDF with page numbers - supports both file paths and streams"""
        chunks = []
        
        try:
            if isinstance(pdf_file_path_or_stream, str):
                # File path
                with open(pdf_file_path_or_stream, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    filename = filename or os.path.basename(pdf_file_path_or_stream)
                    chunks = self._process_pdf_pages(pdf_reader, filename)
            else:
                # Stream object (uploaded file)
                pdf_reader = PyPDF2.PdfReader(pdf_file_path_or_stream)
                filename = filename or "uploaded_file.pdf"
                chunks = self._process_pdf_pages(pdf_reader, filename)
                
        except Exception as e:
            error_msg = str(e)
            if "PyCryptodome" in error_msg or "AES algorithm" in error_msg:
                st.error(f"❌ {filename} is encrypted and requires decryption. Installing pycryptodome...")
                st.info("💡 The PDF appears to be password-protected. Common solutions:")
                st.info("1. Use an unencrypted version of the PDF")
                st.info("2. The system will attempt to decrypt with common passwords")
                st.info("3. Contact support if the issue persists")
            else:
                st.error(f"❌ Error processing PDF {filename}: {error_msg}")
            return []
        
        if chunks:
            st.success(f"✅ Successfully processed {filename}: {len(chunks)} text chunks extracted")
        else:
            st.warning(f"⚠️ No text chunks extracted from {filename}. PDF may be image-based or encrypted.")
        
        return chunks
    
    def _process_pdf_pages(self, pdf_reader, filename: str) -> List[Dict]:
        """Process PDF pages and extract text chunks"""
        chunks = []
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            st.warning(f"PDF {filename} is encrypted. Attempting to decrypt...")
            # Try common passwords or empty password
            passwords_to_try = ['', 'password', '123456', 'admin', 'user']
            decrypted = False
            
            for password in passwords_to_try:
                try:
                    if pdf_reader.decrypt(password):
                        st.success(f"Successfully decrypted {filename}")
                        decrypted = True
                        break
                except Exception as e:
                    continue
            
            if not decrypted:
                st.error(f"Could not decrypt {filename}. PDF may require a specific password.")
                return []
        
        total_pages = len(pdf_reader.pages)
        st.info(f"Processing {filename}: {total_pages} pages")
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                text = page.extract_text()
                if text and text.strip():
                    # Split page into chunks
                    page_chunks = self.split_text_into_chunks(text, page_num, filename)
                    chunks.extend(page_chunks)
                else:
                    st.warning(f"No text found on page {page_num} of {filename}")
            except Exception as e:
                st.warning(f"Error processing page {page_num} of {filename}: {str(e)}")
                continue
        
        return chunks
    
    def split_text_into_chunks(self, text: str, page_num: int, filename: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk_words = words[i:i + Config.CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'page': page_num,
                'chunk_id': len(chunks),
                'filename': filename,
                'document': filename  # Ensure both keys are present
            })
        
        return chunks
    
    def load_documents_from_folder(self, folder_path: str = "data") -> List[Dict]:
        """Load all PDF documents from the specified folder"""
        all_chunks = []
        
        if not os.path.exists(folder_path):
            st.error(f"Data folder '{folder_path}' not found. Please create it and add your PDF files.")
            return []
        
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        if not pdf_files:
            st.warning(f"No PDF files found in '{folder_path}' folder.")
            return []
        
        st.info(f"Found {len(pdf_files)} PDF files in data folder:")
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            st.write(f"📄 {filename}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            filename = os.path.basename(pdf_file)
            status_text.text(f"Processing {filename}...")
            
            try:
                chunks = self.extract_text_from_pdf(pdf_file, filename)
                all_chunks.extend(chunks)
                st.success(f"✅ Processed {filename}: {len(chunks)} chunks extracted")
            except Exception as e:
                st.error(f"❌ Failed to process {filename}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text("Document processing completed!")
        return all_chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Create embeddings for text chunks"""
        if self.embedding_model is None:
            st.error("Embedding model not initialized. Cannot create embeddings.")
            return np.array([]), chunks
            
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            return embeddings, chunks
        except Exception as e:
            st.error(f"Error creating embeddings: {str(e)}")
            return np.array([]), chunks

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.document_metadata = {}
        
    def build_index(self, embeddings: np.ndarray, chunks: List[Dict], filename: str = None):
        """Build FAISS index"""
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        
        # Add document metadata
        start_idx = len(self.chunks)
        for i, chunk in enumerate(chunks):
            if 'filename' not in chunk and filename:
                chunk['document'] = filename
                chunk['filename'] = filename
            elif 'filename' in chunk:
                chunk['document'] = chunk['filename']
            chunk['global_id'] = start_idx + i
        
        self.chunks.extend(chunks)
        self.index.add(embeddings.astype('float32'))
        
        # Store document info
        doc_name = filename or chunks[0].get('filename', 'unknown')
        if doc_name not in self.document_metadata:
            self.document_metadata[doc_name] = {
                'total_chunks': 0,
                'pages': set()
            }
        
        self.document_metadata[doc_name]['total_chunks'] += len(chunks)
        for chunk in chunks:
            self.document_metadata[doc_name]['pages'].add(chunk.get('page', 0))
        
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict]:
        """Search for similar chunks"""
        if self.index is None:
            return []
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results

class GitHubTicketManager:
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def create_ticket(self, title: str, description: str, user_name: str, user_email: str) -> Dict:
        """Create a GitHub issue as support ticket"""
        url = f"https://api.github.com/repos/{self.repo}/issues"
        
        body = f"""**Support Ticket**

**User Information:**
- Name: {user_name}
- Email: {user_email}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Description:**
{description}

---
*This ticket was automatically created by the Customer Support AI System*
"""
        
        data = {
            'title': title,
            'body': body,
            'labels': ['support', 'customer-inquiry']
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=data)
            if response.status_code == 201:
                return {
                    'success': True,
                    'ticket_number': response.json()['number'],
                    'url': response.json()['html_url']
                }
            else:
                return {
                    'success': False,
                    'error': f"GitHub API error: {response.status_code}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class CustomerSupportAI:
    def __init__(self):
        self.vector_store = VectorStore()
        self.doc_processor = DocumentProcessor()
        self.github_manager = None
        self.conversation_history = []
        
    def initialize_openai(self, api_key: str):
        """Initialize OpenAI client"""
        openai.api_key = api_key
        
    def initialize_github(self, token: str, repo: str):
        """Initialize GitHub ticket manager"""
        self.github_manager = GitHubTicketManager(token, repo)
    
    def process_documents(self, uploaded_files: List = None) -> str:
        """Process uploaded documents and build vector store"""
        processed_docs = []
        
        # First, try to load documents from data folder
        if not uploaded_files:
            st.info("🔍 Loading documents from data folder...")
            folder_chunks = self.doc_processor.load_documents_from_folder("data")
            
            if folder_chunks:
                # Group chunks by document
                docs_by_file = {}
                for chunk in folder_chunks:
                    filename = chunk.get('filename', 'unknown')
                    if filename not in docs_by_file:
                        docs_by_file[filename] = []
                    docs_by_file[filename].append(chunk)
                
                # Process each document
                for filename, chunks in docs_by_file.items():
                    try:
                        embeddings, _ = self.doc_processor.create_embeddings(chunks)
                        self.vector_store.build_index(embeddings, chunks, filename)
                        processed_docs.append(f"✅ {filename} ({len(chunks)} chunks)")
                    except Exception as e:
                        processed_docs.append(f"❌ {filename} (error: {str(e)})")
                
                return "\n".join(processed_docs)
        
        # Process uploaded files (fallback or additional documents)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.type == "application/pdf":
                        chunks = self.doc_processor.extract_text_from_pdf(uploaded_file, uploaded_file.name)
                        embeddings, chunks = self.doc_processor.create_embeddings(chunks)
                        self.vector_store.build_index(embeddings, chunks, uploaded_file.name)
                        processed_docs.append(f"✅ {uploaded_file.name} ({len(chunks)} chunks)")
                    else:
                        processed_docs.append(f"❌ {uploaded_file.name} (unsupported format)")
                except Exception as e:
                    processed_docs.append(f"❌ {uploaded_file.name} (error: {str(e)})")
        
        if not processed_docs:
            return "❌ No documents processed. Please ensure PDF files are in the 'data' folder or upload them manually."
        
        return "\n".join(processed_docs)
    
    def search_documents(self, query: str) -> List[Dict]:
        """Search documents for relevant information"""
        if self.doc_processor.embedding_model is None:
            return []
        query_embedding = self.doc_processor.embedding_model.encode([query])
        return self.vector_store.search(query_embedding, k=3)
    
    def generate_response(self, user_query: str) -> str:
        """Generate AI response using OpenAI and document search"""
        # Search documents
        relevant_docs = self.search_documents(user_query)
        
        # Build context
        context = self.build_context(relevant_docs)
        
        # Create system message with explicit citation instructions
        system_message = f"""You are a helpful customer support agent for {Config.COMPANY_NAME}.
Company contact information:
- Email: {Config.COMPANY_EMAIL}
- Phone: {Config.COMPANY_PHONE}

Available Knowledge Base:
- Learning Python.pdf: Comprehensive Python programming guide
- PDF Reference 1.0.pdf: Technical PDF format specification
- Toyota Hilux Manual.pdf: Vehicle operation and maintenance manual

CITATION REQUIREMENTS - VERY IMPORTANT:
When answering questions using document information, you MUST:
1. Always cite the source document name AND page number
2. Use this exact format: "According to [Document Name], page [Page Number], ..."
3. Example: "According to Learning Python.pdf, page 45, you can create a for loop using..."
4. If you find relevant information in the context, always mention the source
5. If no relevant documents are found, say so and suggest creating a support ticket

When answering questions:
1. Search through Learning Python.pdf for Python programming questions
2. Use PDF Reference 1.0.pdf for PDF format and technical questions
3. Reference Toyota Hilux Manual.pdf for vehicle-related inquiries
4. ALWAYS cite sources with document name and page number
5. Be helpful and professional
6. If you cannot find the answer in the documents, suggest creating a support ticket
7. Keep conversation history in mind for continuity

Available actions:
- Answer questions using document knowledge with proper citations
- Suggest creating support tickets for unresolved issues
"""

        # Build messages
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history (last 5 exchanges)
        for msg in self.conversation_history[-10:]:
            messages.append(msg)
        
        # Add current query with context
        if context and context != "No relevant documents found.":
            user_message = f"User Query: {user_query}\n\nRelevant Context from Documents:\n{context}\n\nPlease answer the user's question and cite the specific document and page number where you found the information."
        else:
            user_message = f"User Query: {user_query}\n\nNo relevant documents found. Please provide a general answer and suggest creating a support ticket if needed."
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_query})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            return f"I apologize, but I'm experiencing technical difficulties. Please contact support at {Config.COMPANY_EMAIL} or {Config.COMPANY_PHONE}. Error: {str(e)}"
    
    def build_context(self, relevant_docs: List[Dict]) -> str:
        """Build context string from relevant documents with clear source information"""
        if not relevant_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            # Get document name and page
            doc_name = doc.get('document', doc.get('filename', 'Unknown Document'))
            page_num = doc.get('page', 'Unknown')
            
            # Create source citation
            source = f"[SOURCE {i}] Document: {doc_name}, Page: {page_num}"
            
            # Truncate text if too long
            text = doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text']
            
            context_parts.append(f"{source}\nContent: {text}\n")
        
        return "\n".join(context_parts)
    
    def create_support_ticket(self, title: str, description: str, user_name: str, user_email: str) -> Dict:
        """Create a support ticket via GitHub"""
        if not self.github_manager:
            return {'success': False, 'error': 'GitHub integration not configured'}
        
        return self.github_manager.create_ticket(title, description, user_name, user_email)

# Streamlit UI
def main():
    st.set_page_config(
        page_title=f"{Config.COMPANY_NAME} - Customer Support",
        page_icon="🎧",
        layout="wide"
    )
    
    st.title(f"🎧 {Config.COMPANY_NAME} Customer Support")
    st.write(f"Get instant help with AI-powered support | Contact: {Config.COMPANY_EMAIL} | {Config.COMPANY_PHONE}")
    
    # Initialize session state
    if 'support_ai' not in st.session_state:
        st.session_state.support_ai = CustomerSupportAI()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_query' not in st.session_state:
        st.session_state.processing_query = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        openai_key = st.text_input("OpenAI API Key", type="password", help="Required for AI responses")
        github_token = st.text_input("GitHub Token", type="password", help="Required for ticket creation")
        github_repo = st.text_input("GitHub Repository", placeholder="username/repo-name", help="Format: owner/repository")
        
        if openai_key:
            st.session_state.support_ai.initialize_openai(openai_key)
        
        if github_token and github_repo:
            st.session_state.support_ai.initialize_github(github_token, github_repo)
        
        st.divider()
        
        # Document Upload and Auto-load
        st.header("📄 Knowledge Base")
        
        # Auto-load documents from data folder
        if st.button("🔄 Load Documents from Data Folder", key="load_docs_btn"):
            with st.spinner("Loading documents from data folder..."):
                result = st.session_state.support_ai.process_documents()
                if "✅" in result:
                    st.success("Documents loaded successfully!")
                    st.text(result)
                    
                    # Show loaded documents summary
                    if st.session_state.support_ai.vector_store.document_metadata:
                        st.subheader("📊 Document Summary")
                        for doc_name, metadata in st.session_state.support_ai.vector_store.document_metadata.items():
                            st.write(f"**{doc_name}**")
                            st.write(f"  - Chunks: {metadata['total_chunks']}")
                            st.write(f"  - Pages: {len(metadata['pages'])}")
                else:
                    st.error("Failed to load documents!")
                    st.text(result)
        
        st.divider()
        
        # Manual upload (additional documents)
        st.subheader("📎 Upload Additional Documents")
        uploaded_files = st.file_uploader(
            "Upload additional PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload extra PDF documents beyond those in the data folder"
        )
        
        if uploaded_files and st.button("Process Additional Documents", key="process_additional_btn"):
            with st.spinner("Processing additional documents..."):
                result = st.session_state.support_ai.process_documents(uploaded_files)
                st.success("Additional documents processed!")
                st.text(result)
        
        st.divider()
        
        # Clear Chat
        if st.button("🗑️ Clear Chat History", key="sidebar_clear_chat"):
            st.session_state.chat_history = []
            st.session_state.support_ai.conversation_history = []
            st.session_state.processing_query = False
            st.rerun()
    
    # Main chat interface (INPUT CLEARING VERSION)
    st.header("💬 Chat Support")

    # Initialize input clearing mechanism
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    
    if 'last_message' not in st.session_state:
        st.session_state.last_message = ""

    # Display chat history
    if st.session_state.chat_history:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.write(f"**🧑 You:** {message}")
            else:
                st.write(f"**🤖 Assistant:** {message}")
            st.write("---")
    else:
        st.write("Start a conversation by asking a question below!")

    # Chat input with dynamic key (this clears the input automatically)
    user_message = st.text_input(
        "Ask your question:", 
        placeholder="e.g., How do I create a Python function?", 
        key=f"chat_input_{st.session_state.input_key}"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("Send", key="send_button")
    with col2:
        if st.button("🗑️ Clear Chat", key="main_clear_chat"):
            st.session_state.chat_history = []
            st.session_state.support_ai.conversation_history = []
            st.session_state.input_key += 1  # Clear input
            st.session_state.last_message = ""
            st.rerun()
    
    # Process message when button is clicked OR Enter is pressed
    if user_message and user_message.strip() and user_message != st.session_state.last_message:
        if not openai_key:
            st.error("⚠️ Please provide your OpenAI API key in the sidebar first!")
        else:
            # Store the message to prevent reprocessing
            st.session_state.last_message = user_message
            
            # Add user message
            st.session_state.chat_history.append(("user", user_message))
            
            # Generate AI response
            try:
                with st.spinner("🤔 Thinking..."):
                    ai_response = st.session_state.support_ai.generate_response(user_message)
                
                # Add AI response  
                st.session_state.chat_history.append(("assistant", ai_response))
                
                # Clear the input by incrementing the key
                st.session_state.input_key += 1
                st.session_state.last_message = ""
                
                # Rerun to show updated conversation with cleared input
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                st.session_state.chat_history.append(("assistant", f"Sorry, I encountered an error: {str(e)}"))
                # Still clear input even on error
                st.session_state.input_key += 1
                st.session_state.last_message = ""
    
    # Support ticket form in columns
    col1, col2 = st.columns([1, 1])
    
    with col2:
        st.header("🎫 Create Support Ticket")
        
        with st.form("ticket_form"):
            st.write("Can't find what you're looking for? Create a support ticket!")
            
            user_name = st.text_input("Your Name*")
            user_email = st.text_input("Your Email*")
            ticket_title = st.text_input("Issue Summary*")
            ticket_description = st.text_area("Detailed Description*", height=100)
            
            submitted = st.form_submit_button("Create Ticket")
            
            if submitted:
                if not all([user_name, user_email, ticket_title, ticket_description]):
                    st.error("Please fill in all required fields")
                elif not github_token or not github_repo:
                    st.error("GitHub integration not configured. Please contact support directly at " + Config.COMPANY_EMAIL)
                else:
                    with st.spinner("Creating ticket..."):
                        result = st.session_state.support_ai.create_support_ticket(
                            ticket_title, ticket_description, user_name, user_email
                        )
                    
                    if result['success']:
                        st.success(f"✅ Ticket #{result['ticket_number']} created successfully!")
                        st.write(f"[View Ticket]({result['url']})")
                    else:
                        st.error(f"❌ Failed to create ticket: {result['error']}")
        
        st.divider()
        
        # Quick actions
        st.header("🚀 Quick Actions")
        
        if st.button("📞 Request Callback", key="callback_button"):
            st.info(f"Please call us at {Config.COMPANY_PHONE} or email {Config.COMPANY_EMAIL}")
        
        if st.button("📧 Email Support", key="email_button"):
            st.info(f"Send us an email at {Config.COMPANY_EMAIL}")
        
        if st.button("❓ FAQ", key="faq_button"):
            st.info("Ask me questions like:\n- How do I reset my password?\n- What are your business hours?\n- How do I return a product?")

if __name__ == "__main__":
    main()