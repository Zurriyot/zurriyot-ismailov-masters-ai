import openai
from typing import List, Dict, Optional
import streamlit as st
from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT, MAX_CONVERSATION_HISTORY
from document_processor import DocumentProcessor

class ChatManager:
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.conversation_history = []
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": st.session_state.get("timestamp", 0)
        })
        
        # Keep only the last N messages to manage context window
        if len(self.conversation_history) > MAX_CONVERSATION_HISTORY * 2:
            self.conversation_history = self.conversation_history[-MAX_CONVERSATION_HISTORY:]
    
    def get_context_from_documents(self, query: str) -> str:
        """Get relevant context from documents for the query."""
        if not self.document_processor.documents_loaded:
            return "No documents are currently loaded."
            
        search_results = self.document_processor.search_documents(query, k=3)
        if not search_results:
            return "No relevant information found in the documents."
            
        context = "Relevant information from documents:\n\n"
        for result in search_results:
            context += f"Source: {result['source']}, Page: {result['page']}\n"
            context += f"Content: {result['content'][:500]}...\n\n"
            
        return context
    
    def generate_response(self, user_message: str) -> str:
        """Generate AI response based on user message and document context."""
        try:
            # Get relevant context from documents
            document_context = self.get_context_from_documents(user_message)
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add conversation history
            for msg in self.conversation_history[-MAX_CONVERSATION_HISTORY:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add current user message with document context
            enhanced_message = f"{user_message}\n\nDocument Context:\n{document_context}"
            messages.append({"role": "user", "content": enhanced_message})
            
            # Generate response from OpenAI
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            ai_response = response.choices[0].message.content
            
            # Check if the response indicates no answer was found
            if self._should_suggest_ticket(ai_response, document_context):
                ai_response += "\n\n**No specific answer found in the documents.** Would you like me to help you create a support ticket?"
            
            return ai_response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again or contact support."
    
    def _should_suggest_ticket(self, response: str, context: str) -> bool:
        """Determine if we should suggest creating a support ticket."""
        # Check if the context indicates no relevant information was found
        if "No relevant information found" in context or "No documents are currently loaded" in context:
            return True
            
        # Check if the response indicates uncertainty
        uncertainty_phrases = [
            "I don't have enough information",
            "I cannot find specific information",
            "The documents don't contain",
            "I'm unable to find",
            "No specific answer found"
        ]
        
        return any(phrase.lower() in response.lower() for phrase in uncertainty_phrases)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def format_conversation_for_display(self) -> List[Dict]:
        """Format conversation history for Streamlit display."""
        formatted_messages = []
        
        for msg in self.conversation_history:
            if msg["role"] == "user":
                formatted_messages.append({
                    "role": "user",
                    "content": msg["content"],
                    "avatar": "👤"
                })
            elif msg["role"] == "assistant":
                formatted_messages.append({
                    "role": "assistant", 
                    "content": msg["content"],
                    "avatar": "🤖"
                })
                
        return formatted_messages

