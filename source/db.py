import chromadb
import hashlib
import json
from sentence_transformers import SentenceTransformer


class ChromaDB:
    def __init__(self, path, collection_name, embedding_model_id):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(embedding_model_id)

    def insert(self, data):
        id = self._get_id(data["conversation"])
        embedding = self.embedding_model.encode(str(data))
        metadata = {"data": json.dumps(data)}
        self.collection.add(ids=[id], embeddings=[embedding], metadatas=[metadata])

    def query(self, query):
        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=1)

        if not any(results["ids"][0]): return None
        return json.loads(results["metadatas"][0][0]["data"])

    def _get_id(self, text):
        return hashlib.sha256(text.encode()).hexdigest()
