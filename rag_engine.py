import chromadb
from sentence_transformers import SentenceTransformer  # ✅ Changé
import re


class RAGEngine:
    def __init__(self, chroma_path, collection_name="cv_segments"):

        self.chroma_path = chroma_path
        self.collection_name = collection_name

        # Charger ChromaDB
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(f"Erreur ouverture ChromaDB : {e}")

        # ✅ CORRECTION : Utiliser le même modèle que pour l'indexation
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def extraire_personal_detail(segment, question):
        question_clean = question.lower()
        lines = segment.splitlines()
        for line in lines:
            if question_clean in line.lower():
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    def query(self, user_query, top_k=10):

        if not user_query or not user_query.strip():
            return "⚠️ La question est vide."

        # ✅ Embedding avec le bon modèle
        embedding = self.model.encode(user_query).tolist()

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return "❌ Aucun résultat trouvé."

        # Extraction précise
        for doc, doc_id, dist in zip(documents, ids, distances):
            extracted = self.extraire_personal_detail(doc, user_query)
            if extracted:
                return f"📝 CV {doc_id} → {extracted}"

        # Sinon segment
        return (
            f"📌 Segment trouvé (CV {ids[0]}) :\n\n"
            f"{documents[0]}\n\n"
            f"🧠 Score : {distances[0]:.4f}"
        )
