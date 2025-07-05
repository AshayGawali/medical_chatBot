from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from medical_chatbot.indexer.constants import INDEX_NAME, EMBEDDING_DIMENSION, SIMILARITY_METRIC, PINECONE_REGION, PINECONE_CLOUD


def initialize_pinecone(api_key: str) -> Pinecone:
    if not api_key:
        raise EnvironmentError("Missing Pinecone API Key.")
    return Pinecone(api_key=api_key)


def create_index_if_not_exists(pinecone_client: Pinecone, index_name: str = INDEX_NAME):
    if not pinecone_client.has_index(index_name):
        pinecone_client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=SIMILARITY_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
