import chromadb
from sentence_transformers import SentenceTransformer
from ollama import Client
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

        # Modèle d'embedding
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Client Llama
        self.llama_client = Client()

        # Configuration
        self.SEUIL_PERTINENCE = 0.7  # Score < 0.7 = pertinent
        self.USE_LLM = True  # Active/désactive Llama

    def get_system_prompt(self):
        """
        Prompt système optimisé pour l'analyse de CV
        """
        return """Tu es un assistant IA spécialisé dans l'analyse de CV et le recrutement.

🎯 **Ton rôle** :
- Analyser les CV de manière précise et structurée
- Extraire UNIQUEMENT les informations présentes dans le contexte fourni
- Répondre de façon concise et professionnelle
- Toujours citer le nom du candidat si mentionné

📋 **Format de réponse** :
1. **Pour les compétences/skills** :
   "**Compétences :**
   • Compétence 1
   • Compétence 2
   • Compétence 3"

2. **Pour l'expérience** :
   "**Expérience professionnelle :**
   • Poste | Entreprise | Période
   • Description succincte"

3. **Pour la formation** :
   "**Formation :**
   • Diplôme | Établissement | Année"

4. **Pour les informations de contact** :
   "**Contact :**
   📧 Email : ...
   📱 Téléphone : ...
   🔗 LinkedIn : ..."

⚠️ **Règles STRICTES** :
- Si l'information n'est PAS dans le contexte, réponds : "Cette information n'est pas mentionnée dans les CV "
- NE JAMAIS inventer ou supposer des informations
- Rester factuel et précis
- Utiliser des emojis pour améliorer la lisibilité
- Si plusieurs candidats correspondent, liste-les tous"""

    @staticmethod
    def extraire_personal_detail(segment, question):
        """
        Extraction simple pour les questions directes (email, phone, etc.)
        """
        question_clean = question.lower()
        lines = segment.splitlines()
        for line in lines:
            if question_clean in line.lower():
                parts = re.split(r":\s*", line, maxsplit=1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    def construire_contexte(self, documents, ids, distances, top_n=5):
        """
        Construit un contexte enrichi pour Llama avec les meilleurs segments
        """
        contexte_parts = []

        for i, (doc, doc_id, dist) in enumerate(zip(documents[:top_n], ids[:top_n], distances[:top_n])):
            if dist < self.SEUIL_PERTINENCE:
                contexte_parts.append(
                    f"--- CV {doc_id} (Score: {dist:.2f}) ---\n{doc}\n"
                )

        if not contexte_parts:
            return None

        return "\n".join(contexte_parts)

    def generer_reponse_llm(self, user_query, contexte):
        """
        Génère une réponse intelligente avec Llama
        """
        try:
            response = self.llama_client.chat(
                model="llama3.1:8b",
                messages=[
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": f"""Voici les segments pertinents extraits de CV :

{contexte}

Question de l'utilisateur : {user_query}

Réponds de manière structurée et professionnelle en te basant UNIQUEMENT sur le contexte fourni."""
                    }
                ]
            )

            return response["message"]["content"]

        except Exception as e:
            return f"❌ Erreur lors de la génération de réponse : {e}"

    def query(self, user_query, top_k=10):
        """
        Méthode principale de recherche avec IA
        """
        if not user_query or not user_query.strip():
            return "⚠️ La question est vide."

        # 1. Génération de l'embedding
        embedding = self.model.encode(user_query).tolist()

        # 2. Recherche vectorielle dans ChromaDB
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return "❌ Aucun résultat trouvé dans la base de données."

        # 3. Tentative d'extraction directe (pour questions simples)
        for doc, doc_id, dist in zip(documents, ids, distances):
            extracted = self.extraire_personal_detail(doc, user_query)
            if extracted and dist < 0.5:  # Seulement si très pertinent
                return f"✅ **Réponse trouvée** (CV {doc_id}) :\n\n{extracted}"

        # 4. Utiliser Llama pour une réponse intelligente
        if self.USE_LLM:
            contexte = self.construire_contexte(documents, ids, distances, top_n=5)

            if contexte:
                print("🧠 Génération de réponse avec Llama...")
                reponse_llm = self.generer_reponse_llm(user_query, contexte)
                return reponse_llm
            else:
                return (
                    f" Résultats peu pertinents.\n\n"
                    f"💡 Essayez de reformuler votre question de manière plus spécifique.\n"
                    f"Exemples : 'Quelles sont les compétences en Python ?', 'Quelle est l'expérience de [nom] ?'"
                )

        # 5. Fallback : retour du meilleur segment (si Llama désactivé)
        else:
            if distances[0] < self.SEUIL_PERTINENCE:
                return (
                    f"📌 **Segment pertinent** (CV {ids[0]}) :\n\n"
                    f"{documents[0]}\n\n"
                    f"🎯 Score : {distances[0]:.2f}"
                )
            else:
                return (
                    f"⚠️ **Résultat approximatif** (CV {ids[0]}) :\n\n"
                    f"{documents[0]}\n\n"
                    f"📊 Score : {distances[0]:.2f} (faible pertinence)"
                )
