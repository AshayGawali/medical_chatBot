from medical_chatbot.utils.helper import load_pdf_file, text_split, download_huggingface_embeddings
from medical_chatbot.config import DATA_PATH, PINECONE_API_KEY
from medical_chatbot.utils.pinecone_utils import initialize_pinecone, create_index_if_not_exists
from medical_chatbot.utils.embedding_utils import store_documents_in_pinecone
from indexer.constants import INDEX_NAME


def run_index_pipeline():
    print("📄 Extracting and splitting PDF documents...")
    extracted_data = load_pdf_file(DATA_PATH)
    text_chunks = text_split(extracted_data)

    print("📥 Initializing embedding model...")
    embed_model = download_huggingface_embeddings()

    print("🔗 Connecting to Pinecone...")
    pc = initialize_pinecone(PINECONE_API_KEY)

    print(f"📌 Creating/validating index: {INDEX_NAME}")
    create_index_if_not_exists(pc, INDEX_NAME)

    print("📦 Uploading documents to Pinecone...")
    store_documents_in_pinecone(text_chunks, INDEX_NAME, embed_model)

    print("✅ Indexing pipeline completed successfully.")
