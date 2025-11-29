import sqlite3
import os
import re
import pdfplumber

# Chemin vers le dossier contenant les PDF
folder_path = r"C:\Users\R I B\Desktop\agent_ia\data_pdf"

# Fonction pour extraire le texte d'un PDF


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text() is not None)


# Extraire les textes de tous les CV
cv_texts = {}
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        full_path = os.path.join(folder_path, filename)
        cv_texts[filename] = extract_text(full_path)

# Créer / connecter la base SQLite
conn = sqlite3.connect('cvs.db')
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS candidats (
    id INTEGER PRIMARY KEY,
    nom TEXT,
    email TEXT,
    cv_texte TEXT,
    fichier_pdf TEXT
)
''')

# Insérer chaque CV dans la base
for filename, texte in cv_texts.items():
    nom = filename.replace(".pdf", "")

    # Extraire l'email du texte du CV
    match = re.search(r'[\w\.-]+@[\w\.-]+', texte)
    email = match.group(0) if match else f"{nom.lower()}@email.com"

    c.execute(
        "INSERT INTO candidats (nom, email, cv_texte, fichier_pdf) VALUES (?, ?, ?, ?)",
        (nom, email, texte, filename)
    )

conn.commit()
conn.close()

print("Tous les CV ont été ajoutés à la base cvs.db avec succès !")
