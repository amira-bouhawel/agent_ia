import chromadb
from sentence_transformers import SentenceTransformer
import re

# ==========================
# 📌 Configuration ChromaDB
# ==========================
chroma_path = r"C:\Users\R I B\Desktop\agent_ia\chroma_db"
collection_name = "cv_segments"

try:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
except Exception as e:
    print(f"❌ Erreur lors de l'ouverture de la collection '{collection_name}': {e}")
    exit(1)

# ==========================
# 📌 Chargement du modèle embeddings
# ==========================
model = SentenceTransformer("all-MiniLM-L6-v2")
print("🧠 Modèle all-MiniLM-L6-v2 chargé avec succès.\n")

# ==========================
# 🔎 Boucle interactive RAG
# ==========================
print("🔎 Système RAG interactif pour tes CV")
print("Tape 'exit' pour quitter.\n")


def extraire_personal_detail(segment, question):
    """
    Extraction ciblée de la valeur correspondant à la question
    dans la section Personal Details.
    """
    question_clean = question.lower()
    segment_lines = segment.splitlines()
    for line in segment_lines:
        if question_clean in line.lower():
            # Extraire tout après le ":"
            match = re.split(r":\s*", line, maxsplit=1)
            if len(match) == 2:
                return match[1].strip()
    return None


while True:
    query = input("👉 Pose ta question : ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Sortie du programme. À bientôt !")
        break

    if not query:
        print("⚠️ La question ne peut pas être vide. Réessaie.\n")
        continue

    # ➤ Génération embedding pour la requête
    query_embedding = model.encode(query).tolist()

    # ➤ Recherche vectorielle top 10 résultats pour trouver le segment le plus pertinent
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]  # Score de similarité

    if not documents:
        print("❌ Aucun résultat trouvé pour cette question.\n")
        continue

    # ➤ Extraire la réponse exacte si la question correspond à Personal Details
    meilleur_score = -1
    meilleur_doc = ""
    meilleur_id = None
    valeur_extraite = None

    for doc, doc_id, dist in zip(documents, ids, distances):
        extracted = extraire_personal_detail(doc, query)
        if extracted:
            meilleur_score = dist
            meilleur_doc = doc
            meilleur_id = doc_id
            valeur_extraite = extracted
            break

    if valeur_extraite:
        print("\n📌 RÉSULTAT LE PLUS PRÉCIS :\n")
        print(f"(CV ID: {meilleur_id})")
        print(valeur_extraite)
        print(f"🧠 Score de similarité : {meilleur_score:.4f}")
        print("-" * 50)
    else:
        # Si pas d'extraction exacte → montrer le segment le plus proche
        print("\n📌 SEGMENT LE PLUS PERTINENT :\n")
        print(f"(CV ID: {ids[0]})")
        print(documents[0])
        print(f"🧠 Score de similarité : {distances[0]:.4f}")
        print("-" * 50)
