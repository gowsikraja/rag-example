# RAG Example Project

This repository contains an example implementation of a Retrieval-Augmented Generation (RAG) pipeline. The project demonstrates how to use vector stores, embeddings, and search techniques to build a system capable of retrieving and summarizing relevant information from a document corpus.

## Features
- **Document Loading**: Load and preprocess documents from the `data/` directory.
- **Vector Store**: Use FAISS or ChromaDB to store and query document embeddings.
- **Search and Summarization**: Perform RAG-based search and generate summaries for user queries.
- **Notebooks**: Jupyter notebooks for experimentation and pipeline demonstration.

## Project Structure
```
├── app.py                # Example script for running the RAG pipeline
├── main.py               # Entry point for the project
├── src/                  # Source code for the pipeline
│   ├── data_loader.py    # Document loading and preprocessing
│   ├── embedding.py      # Embedding generation
│   ├── search.py         # RAG search logic
│   ├── vector_store.py   # Vector store implementation
├── data/                 # Directory for storing documents
├── faiss_store/          # Directory for FAISS index files
├── notebook/             # Jupyter notebooks for experimentation
├── requirement.txt       # Python dependencies
├── pyproject.toml        # Project metadata and dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gowsikraja/rag-example.git
   cd rag-example
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirement.txt
   ```

## Usage

### Running the Example Script
To run the example RAG pipeline:
```bash
python app.py
```

### Jupyter Notebooks
Explore the pipeline using the provided notebooks:
```bash
jupyter notebook notebook/rag_pipeline.ipynb
```

## Dependencies
This project uses the following key libraries:
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)
- [PyPDF](https://github.com/py-pdf/pypdf)


## Acknowledgments
- Inspired by the LangChain community and open-source RAG implementations.