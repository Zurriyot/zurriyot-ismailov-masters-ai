# Data Folder

Place your PDF documents in this folder for the AI system to process and use as knowledge sources.

## Required PDF Files

The system expects the following PDF files:
- Learning_Python.pdf
- PDF Reference 1.0.pdf  
- Toyota Hilux Manual.pdf

## How it works

1. The system will automatically detect PDF files in this folder
2. Each PDF will be processed and converted to searchable text chunks
3. Text chunks will be embedded using OpenAI's embedding model
4. The AI can then search through these embeddings to answer questions
5. Answers will include source document name and page number citations

## File Format

- Only PDF files (.pdf extension) are supported
- Files should contain readable text (not scanned images)
- Larger files will be split into smaller chunks for better processing
- Each chunk will maintain page number information for accurate citations

