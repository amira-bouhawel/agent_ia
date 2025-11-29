import chromadb
from sentence_transformers import SentenceTransformer

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

    # ➤ Recherche vectorielle top 5 résultats
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not documents:
        print("❌ Aucun résultat trouvé pour cette question.\n")
        continue

    print("\n📌 RÉSULTATS TROUVÉS :\n")
    for i, (doc, doc_id) in enumerate(zip(documents, ids), start=1):
        print(f"--- Résultat {i} ---")
        print(f"(CV ID: {doc_id})")
        print(doc)
        print("-" * 50)
