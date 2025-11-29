import pdfplumber
import os


def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text() is not None)


# Chemin vers ton dossier contenant les PDF
folder_path = r"C:\Users\R I B\Desktop\agent_ia\data_pdf"

cv_texts = {}
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        full_path = os.path.join(folder_path, filename)
        cv_texts[filename] = extract_text(full_path)

# Optionnel : afficher les 500 premiers caractères de chaque CV
for name, texte in cv_texts.items():
    print(f"--- {name} ---")
    print(texte[:500])
    print("\n")
