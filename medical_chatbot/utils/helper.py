# helper.py

# Import PDF loaders and text splitters from LangChain and community integrations
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader  # PDF loader to read and extract text from PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Splits text into manageable chunks for LLM processing
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embedding integration with LangChain


# 📄 Function to extract text from all PDF files in a specified directory
def load_pdf_file(data):
    """
    Loads PDF documents from the given directory.

    Args:
        data (str): Path to the directory containing PDF files.

    Returns:
        list: List of LangChain Document objects extracted from the PDFs.
    """
    # Load PDF files using PyPDFLoader through DirectoryLoader
    loader = DirectoryLoader(
        data,                       # Directory path
        glob="*.pdf",               # Pattern to match only PDF files
        loader_cls=PyPDFLoader      # Use PyPDFLoader to parse PDF content
    )

    # Read and extract text from all matching PDFs
    documents = loader.load()
    
    return documents


# ✂️ Function to split large documents into smaller text chunks
def text_split(data, chunk_size=500, chunk_overlap=20):
    """
    Splits input documents into smaller chunks to support better LLM input handling.

    Args:
        data (list): List of Document objects.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        list: List of split text chunks (Document objects).
    """
    # Initialize the text splitter with configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split and return the document chunks
    return text_splitter.split_documents(data)


# 🔎 Function to initialize a sentence embedding model
def download_huggingface_embeddings():
    """
    Loads a pre-trained sentence embedding model from Hugging Face.

    Returns:
        HuggingFaceEmbeddings: Embedding model instance.
    """
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"  # Compact, high-performance sentence embedding model
    )
    
    return embed_model
