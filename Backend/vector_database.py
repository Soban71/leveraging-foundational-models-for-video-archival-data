import chromadb

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "video_frame_embeddings"
TEXT_COLLECTION_NAME = "video_text_embeddings"


def load_vector_data():
    # Loading all frame embeddings and metadata from ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    data = collection.get(
        include=["embeddings", "metadatas"]
    )

    embeddings = data["embeddings"]
    metadatas = data["metadatas"]

    print(f"Loaded {len(embeddings)} frame embeddings from ChromaDB.")

    return embeddings, metadatas


def load_text_vector_data():
    """Load one transcript embedding and its metadata for every video clip."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=TEXT_COLLECTION_NAME)

    data = collection.get(include=["embeddings", "metadatas"])
    embeddings = data["embeddings"]
    metadatas = data["metadatas"]

    print(f"Loaded {len(embeddings)} text embeddings from ChromaDB.")
    return embeddings, metadatas


if __name__ == "__main__":
    embeddings, metadatas = load_vector_data()
    print("Vector database is ready.")