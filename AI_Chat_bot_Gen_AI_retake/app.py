import streamlit as st
import os
from datetime import datetime
from config import COMPANY_INFO, OPENAI_API_KEY, VECTOR_DB_PATH
from document_processor import DocumentProcessor
from chat_manager import ChatManager
from github_integration import GitHubIntegration

# Ensure VECTOR_DB_PATH is defined
if VECTOR_DB_PATH is None:
    st.info("ℹ️ Using in-memory vector storage for Docker deployment")
else:
    st.info(f"ℹ️ Using persistent vector storage at: {VECTOR_DB_PATH}")

# Load environment variables for Hugging Face
from dotenv import load_dotenv
load_dotenv()

# Page configuration
st.set_page_config(
    page_title=f"{COMPANY_INFO['name']} - AI Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disable Streamlit metrics and telemetry for Docker deployment
if os.environ.get('STREAMLIT_BROWSER_GATHER_USAGE_STATS') != 'true':
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .stButton > button {
        width: 100%;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .fixed-right-pane {
        position: sticky;
        top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = None
    if 'timestamp' not in st.session_state:
        st.session_state.timestamp = 0

def auto_process_documents():
    """Automatically process documents on startup."""
    if not st.session_state.documents_processed:
        # Check if vector storage already exists (only for persistent storage)
        if VECTOR_DB_PATH and os.path.exists(VECTOR_DB_PATH) and os.listdir(VECTOR_DB_PATH):
            st.info("🔄 Loading existing vector database...")
            try:
                doc_processor = DocumentProcessor()
                # Try to load existing vector store
                if doc_processor.load_existing_vector_store():
                    st.session_state.documents_processed = True
                    st.session_state.chat_manager = ChatManager(doc_processor)
                    st.success("✅ Existing documents loaded successfully!")
                    return True
                else:
                    st.warning("⚠️ Existing vector database corrupted, reprocessing documents...")
            except Exception as e:
                st.warning(f"⚠️ Error loading existing database: {str(e)}")
                st.info("🔄 Reprocessing documents...")
        else:
            # No persistent storage or in-memory mode
            st.info("🔄 No existing database found - processing documents...")
        
        # Process documents if no existing storage or loading failed
        with st.spinner("🔄 Processing documents..."):
            doc_processor = DocumentProcessor()
            if doc_processor.process_documents():
                st.session_state.documents_processed = True
                st.session_state.chat_manager = ChatManager(doc_processor)
                st.success("✅ Documents processed successfully!")
                return True
            else:
                st.error("❌ Failed to process documents")
                return False
    return True

def main():
    initialize_session_state()
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🤖 {COMPANY_INFO['name']} AI Support</h1>
        <p>Your intelligent customer support assistant powered by OpenAI</p>
        <p>📞 {COMPANY_INFO['phone']} | 📧 {COMPANY_INFO['email']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check OpenAI API key first
    if not OPENAI_API_KEY:
        st.error("⚠️ **OPENAI_API_KEY is required!**")
        st.info("""
        **To use this app, you need to set your OpenAI API key:**
        
        🔧 **For Hugging Face Spaces:**
        1. Go to your Space settings
        2. Click on **Variables and Secrets**
        3. Add a new secret: `OPENAI_API_KEY` with your OpenAI API key
        4. Restart the space
        
        🔧 **For Local Development:**
        1. Create a `.env` file in the project root
        2. Add: `OPENAI_API_KEY=your_api_key_here`
        
        **Get your API key from:** https://platform.openai.com/api-keys
        """)
        return
    
    # Auto-process documents on startup
    if not auto_process_documents():
        st.error("❌ Cannot proceed without processing documents")
        return
    
    # Main content area with fixed layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 AI Support Chat")
        
        # Chat interface
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask me anything about the documents..."):
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate AI response
                if st.session_state.chat_manager:
                    with st.chat_message("assistant"):
                        with st.spinner("🤔 Thinking..."):
                            response = st.session_state.chat_manager.generate_response(prompt)
                            st.markdown(response)
                        
                        # Add AI response to chat
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Add to chat manager history
                        st.session_state.chat_manager.add_message("user", prompt)
                        st.session_state.chat_manager.add_message("assistant", response)
    
    # Fixed right pane for support ticket
    with col2:
        st.header("🎫 Create Support Ticket")
        
        # GitHub integration status
        github_integration = GitHubIntegration()
        
        if not github_integration.is_configured():
            st.warning("⚠️ GitHub integration not configured")
            st.info("Set GITHUB_TOKEN and GITHUB_REPO in .env file")
            
            # Show help information even without GitHub
            st.markdown("""
            <div class="info-box">
                <h4>💡 How to use:</h4>
                <ol>
                    <li>Ask questions in the chat</li>
                    <li>AI will search documents for answers</li>
                    <li>If no answer found, create a ticket</li>
                    <li>Support team will review your ticket</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Recent tickets
        recent_issues = github_integration.get_recent_issues(3)
        if recent_issues:
            st.subheader("📋 Recent Tickets")
            for issue in recent_issues:
                st.write(f"• #{issue['number']}: {issue['title'][:50]}...")
                st.write(f"  [View Issue]({issue['url']})")
        
        # Support ticket form
        with st.form("support_ticket_form"):
            user_name = st.text_input("Your Name *", placeholder="Enter your full name")
            user_email = st.text_input("Your Email *", placeholder="Enter your email address")
            ticket_title = st.text_input("Issue Summary *", placeholder="Brief description of the issue")
            ticket_description = st.text_area(
                "Detailed Description *", 
                placeholder="Please provide detailed information about your issue...",
                height=150
            )
            
            submitted = st.form_submit_button("🚀 Create Support Ticket", type="primary")
            
            if submitted:
                if not all([user_name, user_email, ticket_title, ticket_description]):
                    st.error("Please fill in all required fields")
                else:
                    with st.spinner("Creating support ticket..."):
                        ticket_data = {
                            "user_name": user_name,
                            "user_email": user_email,
                            "title": ticket_title,
                            "description": ticket_description
                        }
                        
                        issue_url = github_integration.create_support_ticket(ticket_data)
                        
                        if issue_url:
                            st.success("✅ Support ticket created successfully!")
                            st.info(f"View your ticket: [GitHub Issue]({issue_url})")
                            
                            # Clear form
                            st.rerun()
                        else:
                            st.error("❌ Failed to create support ticket")
        
        # Help information
        st.markdown("""
        <div class="info-box">
            <h4>💡 How to use:</h4>
            <ol>
                <li>Ask questions in the chat</li>
                <li>AI will search documents for answers</li>
                <li>If no answer found, create a ticket</li>
                <li>Support team will review your ticket</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar with additional controls
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Show document status
        if st.session_state.documents_processed:
            st.success("✅ Documents loaded and ready")
            
            # Show document summary
            if st.session_state.chat_manager:
                doc_summary = st.session_state.chat_manager.document_processor.get_document_summary()
                if doc_summary:
                    st.subheader("📋 Document Summary")
                    for doc, chunks in doc_summary.items():
                        st.write(f"• {doc}: {chunks} chunks")
        
        # Clear conversation
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.chat_manager:
                st.session_state.chat_manager.clear_conversation()
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please check the logs and ensure all dependencies are properly configured.")
        st.exception(e)

