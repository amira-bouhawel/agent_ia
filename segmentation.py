import sqlite3
import re

# ---------- Nettoyage avancé ------


def nettoyer_texte(txt):
    if not txt:
        return ""

    # Supprimer les caractères doublés : "MMaarruutthhii" → "Maruthi"
    txt = re.sub(r'(.)\1{1,}', r'\1', txt)

    # Remplacer les caractères bizarres
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)

    # Réduire les multiples espaces
    txt = re.sub(r'\s+', ' ', txt)

    return txt.strip()


# ---------- Segmentation robuste ----------
def segmenter_texte(txt):
    if not txt:
        return []

    # 1) segmentation sur 2 sauts de ligne
    segments = re.split(r"\n\s*\n", txt)

    # 2) segmentation par points, listes ou tirets si un seul segment
    if len(segments) == 1:
        segments = re.split(r"[•\-–]\s+|\. ", txt)

    # 3) découper toutes les 300-350 caractères si encore un seul segment
    if len(segments) == 1:
        taille = 350
        segments = [txt[i:i + taille] for i in range(0, len(txt), taille)]

    # 4) améliorer précision en regroupant sections logiques si possible
    sections_keywords = ["personal", "education", "employment", "skills", "project", "internship"]
    new_segments = []
    for s in segments:
        lower_s = s.lower()
        for kw in sections_keywords:
            if kw in lower_s:
                # découper plus finement autour des sections
                sub_segments = re.split(r"\n|\. ", s)
                new_segments.extend([ss.strip() for ss in sub_segments if len(ss.strip()) > 30])
                break
        else:
            new_segments.append(s.strip())

    # Nettoyer segments vides
    return [s for s in new_segments if len(s) > 30]


# ======================================================
#               CHARGER CVS.BD + CREER RAG
# ======================================================
print("📥 Chargement de cvs.db ...")
conn = sqlite3.connect("cvs.db")
c = conn.cursor()
c.execute("SELECT fichier_pdf, cv_texte FROM candidats")
rows = c.fetchall()
conn.close()

print(f"{len(rows)} CV trouvés.\n")

# Nouvelle BDD pour RAG
conn_rag = sqlite3.connect("rag_segments.db")
c_rag = conn_rag.cursor()

# table clean + segments
c_rag.execute("""
CREATE TABLE IF NOT EXISTS rag_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fichier TEXT,
    segment_index INTEGER,
    segment TEXT
)
""")

total_segments = 0

for fichier, texte in rows:
    texte_clean = nettoyer_texte(texte)
    segments = segmenter_texte(texte_clean)

    print(f"{fichier} → {len(segments)} segments")

    for i, seg in enumerate(segments):
        c_rag.execute(
            "INSERT INTO rag_segments (fichier, segment_index, segment) VALUES (?, ?, ?)",
            (fichier, i, seg)
        )

    total_segments += len(segments)

conn_rag.commit()
conn_rag.close()

print("\n🎉 rag_segments.db créé avec succès !")
print(f"Total segments générés : {total_segments}")
print("Cette base est maintenant 100% prête pour un moteur RAG.")
