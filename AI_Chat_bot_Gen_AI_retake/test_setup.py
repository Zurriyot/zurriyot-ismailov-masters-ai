#!/usr/bin/env python3
"""
Test script to verify the Customer Support AI system setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing package imports...")
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import openai
        print(f"✅ OpenAI {openai.__version__}")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__}")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        import chromadb
        print(f"✅ ChromaDB {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print(f"✅ PyPDF2 {PyPDF2.__version__}")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import COMPANY_INFO, OPENAI_API_KEY
        print(f"✅ Company: {COMPANY_INFO['name']}")
        print(f"✅ Phone: {COMPANY_INFO['phone']}")
        print(f"✅ Email: {COMPANY_INFO['email']}")
        
        if OPENAI_API_KEY:
            print("✅ OpenAI API key found")
        else:
            print("⚠️ OpenAI API key not set")
            
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_modules():
    """Test if all custom modules can be imported."""
    print("\n🔍 Testing custom modules...")
    
    try:
        from document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported")
    except Exception as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    try:
        from chat_manager import ChatManager
        print("✅ ChatManager imported")
    except Exception as e:
        print(f"❌ ChatManager import failed: {e}")
        return False
    
    try:
        from github_integration import GitHubIntegration
        print("✅ GitHubIntegration imported")
    except Exception as e:
        print(f"❌ GitHubIntegration import failed: {e}")
        return False
    
    return True

def test_data_folder():
    """Test data folder structure."""
    print("\n🔍 Testing data folder...")
    
    if os.path.exists("data"):
        print("✅ Data folder exists")
        
        pdf_files = [f for f in os.listdir("data") if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF files:")
            for pdf in pdf_files:
                print(f"   • {pdf}")
        else:
            print("⚠️ No PDF files found in data folder")
            print("   Please add your PDF files to the data/ folder")
    else:
        print("❌ Data folder not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Customer Support AI System - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_modules,
        test_data_folder
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to run.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set environment variables in .env file")
        print("3. Add PDF files to data/ folder")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

