import os
import PyPDF2
from typing import List, Dict, Tuple
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import streamlit as st
from config import VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY

class DocumentProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        self.vector_store = None
        self.documents_loaded = False
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    st.warning(f"PDF '{pdf_path}' is encrypted. Installing 'pycryptodome' may help decrypt it.")
                    st.info("To fix this, run: pip install pycryptodome")
                    return ""
                
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as page_error:
                        st.warning(f"Could not extract text from page {page_num + 1}: {str(page_error)}")
                        continue
                
                return text
        except Exception as e:
            error_msg = str(e)
            if "PyCryptodome is required" in error_msg:
                st.error(f"PDF '{pdf_path}' is encrypted and requires PyCryptodome library.")
                st.info("To fix this, install: pip install pycryptodome")
            elif "password" in error_msg.lower():
                st.error(f"PDF '{pdf_path}' is password-protected.")
                st.info("Please provide the password or use an unencrypted version.")
            else:
                st.error(f"Error reading PDF {pdf_path}: {error_msg}")
            return ""
    
    def load_existing_vector_store(self) -> bool:
        """Load existing vector store if available."""
        try:
            if VECTOR_DB_PATH and os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
                # Try to load existing persistent ChromaDB
                self.vector_store = Chroma(
                    persist_directory=VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
                
                # Test if the vector store is working
                test_results = self.vector_store.similarity_search("test", k=1)
                if test_results:
                    self.documents_loaded = True
                    return True
                else:
                    return False
            else:
                # No persistent storage available
                st.info("ℹ️ No persistent vector store found - will process documents on startup")
                return False
        except Exception as e:
            st.warning(f"Error loading existing vector store: {str(e)}")
            return False

    def process_documents(self, data_folder: str = "data") -> bool:
        """Process all PDF documents in the data folder and create vector embeddings."""
        if not os.path.exists(data_folder):
            st.error(f"Data folder '{data_folder}' not found!")
            return False
            
        pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.error("No PDF files found in data folder!")
            return False
            
        st.info(f"Found {len(pdf_files)} PDF files. Processing...")
        
        all_texts = []
        all_metadatas = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_folder, pdf_file)
            st.info(f"Processing {pdf_file}...")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                continue
                
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_metadatas.append({
                    "source": pdf_file,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
        
        if not all_texts:
            st.error("No text could be extracted from the PDFs!")
            return False
            
        # Create vector store
        try:
            if VECTOR_DB_PATH:
                # Persistent storage
                os.makedirs(VECTOR_DB_PATH, exist_ok=True)
                self.vector_store = Chroma.from_texts(
                    texts=all_texts,
                    metadatas=all_metadatas,
                    embedding=self.embeddings,
                    persist_directory=VECTOR_DB_PATH
                )
            else:
                # In-memory storage (better for Docker)
                self.vector_store = Chroma.from_texts(
                    texts=all_texts,
                    metadatas=all_metadatas,
                    embedding=self.embeddings
                )
            
            self.documents_loaded = True
            st.success(f"Successfully processed {len(all_texts)} text chunks from {len(pdf_files)} documents!")
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            if VECTOR_DB_PATH:
                st.info(f"Vector DB path: {VECTOR_DB_PATH}")
                st.info(f"Directory exists: {os.path.exists(VECTOR_DB_PATH)}")
                st.info(f"Directory writable: {os.access(VECTOR_DB_PATH, os.W_OK)}")
            return False
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents based on the query."""
        if not self.documents_loaded or not self.vector_store:
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            formatted_results = []
            
            for doc, score in results:
                # Extract page number from the chunk text
                page_num = "Unknown"
                if "--- Page" in doc.page_content:
                    lines = doc.page_content.split('\n')
                    for line in lines:
                        if line.startswith("--- Page"):
                            page_num = line.replace("--- Page ", "").replace(" ---", "")
                            break
                
                formatted_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": page_num,
                    "score": score
                })
            
            return formatted_results
        except Exception as e:
            st.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_document_summary(self) -> Dict[str, int]:
        """Get a summary of loaded documents."""
        if not self.documents_loaded or not self.vector_store:
            return {}
            
        try:
            if VECTOR_DB_PATH:
                # Persistent storage - get from vector store
                all_docs = self.vector_store.get()
                source_counts = {}
                
                for metadata in all_docs['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    source_counts[source] = source_counts.get(source, 0) + 1
                    
                return source_counts
            else:
                # In-memory storage - return basic info
                return {"Documents": "Loaded in memory"}
        except Exception as e:
            st.error(f"Error getting document summary: {str(e)}")
            return {}

