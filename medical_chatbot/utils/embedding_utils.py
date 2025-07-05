from typing import List
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document


def store_documents_in_pinecone(
    documents: List[Document],
    index_name: str,
    embed_model
) -> PineconeVectorStore:
    """
    Creates or updates a Pinecone index by embedding and storing new documents.

    Args:
        documents (List[Document]): List of LangChain Document chunks.
        index_name (str): Name of the Pinecone index.
        embed_model: Embedding model compatible with LangChain.

    Returns:
        PineconeVectorStore: The initialized vector store.
    """
    if not documents:
        raise ValueError("No documents provided for indexing.")
    
    return PineconeVectorStore.from_documents(
        documents=documents,
        index_name=index_name,
        embedding=embed_model,
    )


def load_existing_pinecone_index(
    index_name: str,
    embed_model
) -> PineconeVectorStore:
    """
    Loads an existing Pinecone index for retrieval or inference.

    Args:
        index_name (str): Name of the existing Pinecone index.
        embed_model: Embedding model compatible with LangChain.

    Returns:
        PineconeVectorStore: The loaded vector store.
    """
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embed_model,
    )
