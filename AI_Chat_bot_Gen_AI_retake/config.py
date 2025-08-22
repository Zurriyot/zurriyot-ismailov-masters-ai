import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Company Information
COMPANY_INFO = {
    "name": "TechCorp Solutions",
    "phone": "+1 (555) 123-4567",
    "email": "support@techcorp.com",
    "website": "https://www.techcorp.com",
    "address": "123 Innovation Drive, Tech City, TC 12345"
}

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4-1106-preview"  # Using latest GPT-4 model
MAX_TOKENS = 4000

# GitHub Configuration (Optional)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")

# Vector Database Configuration
# Use in-memory for Docker deployment to avoid permission issues
VECTOR_DB_PATH = None  # None means in-memory storage
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Chat Configuration
MAX_CONVERSATION_HISTORY = 10
TEMPERATURE = 0.7

# System Prompt
SYSTEM_PROMPT = f"""You are a helpful customer support AI assistant for {COMPANY_INFO['name']}. 

Company Information:
- Company: {COMPANY_INFO['name']}
- Phone: {COMPANY_INFO['phone']}
- Email: {COMPANY_INFO['email']}
- Website: {COMPANY_INFO['website']}

Your role is to:
1. Answer customer questions based on the provided documents
2. Always cite the source document and page number when answering
3. If you cannot find an answer in the documents, suggest creating a support ticket
4. Help users create support tickets with proper information
5. Be professional, helpful, and accurate

When citing sources, use the format: "Source: [Document Name], Page [Page Number]"
"""

