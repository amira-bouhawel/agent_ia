import sqlite3
import chromadb

from sentence_transformers import SentenceTransformer

print("📥 Chargement des segments depuis rag_segments.db ...")

# Charger segments SQL
conn = sqlite3.connect("rag_segments.db")
c = conn.cursor()
c.execute("SELECT id, segment FROM rag_segments")
rows = c.fetchall()
conn.close()
print(f"👍 {len(rows)} segments trouvés\n")

# Charger modèle
print("🧠 Chargement modèle embeddings ...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Créer base persistante
client = chromadb.PersistentClient(
    path=r"C:\Users\R I B\Desktop\agent_ia\chroma_db"
)

collection = client.get_or_create_collection(
    name="cv_segments",
    metadata={"hnsw:space": "cosine"}
)

print("⚙️ Génération embeddings et insertion...")

for row_id, seg in rows:
    emb = model.encode(seg).tolist()
    collection.add(
        ids=[str(row_id)],
        embeddings=[emb],
        documents=[seg]
    )

print("🎉 Embeddings vectoriels générés et stockés dans cv_segments !")
print("💾 Persistence stockée automatiquement dans chroma_db/")
