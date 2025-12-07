from ollama import Client

client = Client()

# ============================================
# 📋 PROMPT SYSTÈME OPTIMISÉ POUR L'ANALYSE CV
# ============================================

SYSTEM_PROMPT = """Tu es un assistant IA spécialisé dans l'analyse de CV et le recrutement. 

🎯 **Ton rôle** :
- Analyser les CV de manière précise et structurée
- Extraire les informations demandées (compétences, expérience, formations, etc.)
- Répondre de façon concise et professionnelle
- Toujours citer le nom du candidat dans ta réponse

📋 **Format de réponse attendu** :
1. Si la question porte sur des compétences/skills :
   "**Compétences de [Nom du candidat] :**
   - Compétence 1
   - Compétence 2
   - Compétence 3"

2. Si la question porte sur l'expérience :
   "**Expérience professionnelle de [Nom du candidat] :**
   - Poste | Entreprise | Période
   - Description succincte"

3. Si la question porte sur la formation :
   "**Formation de [Nom du candidat] :**
   - Diplôme | Établissement | Année"

4. Si la question porte sur les informations de contact :
   "**Contact de [Nom du candidat] :**
   📧 Email : ...
   📱 Téléphone : ...
   🔗 LinkedIn : ..."

⚠️ **Règles importantes** :
- Si l'information n'est pas présente dans le CV, réponds : "Cette information n'est pas mentionnée dans le CV [Nom]"
- Ne jamais inventer ou supposer des informations
- Rester factuel et précis
- Utiliser des emojis pour améliorer la lisibilité
"""

# ============================================
# 🧪 TEST AVEC EXEMPLE
# ============================================

# Exemple de texte CV (vous remplacerez par les vrais segments de ChromaDB)
cv_context = """
Personal Details:
Name: Anuva Goyal
Email: anuvagoyal1@gmail.com
Phone: +91-9876543210
LinkedIn: linkedin.com/in/anuvagoyal
GitHub: github.com/AnuvaGoyal

Skills:
- Python (Advanced)
- Machine Learning (TensorFlow, PyTorch)
- Data Analysis (Pandas, NumPy)
- Deep Learning
- Natural Language Processing
- SQL & NoSQL Databases
- Git & GitHub
- Docker
- AWS Cloud Services

Experience:
Data Scientist | TechCorp India | 2022 - Present
- Développement de modèles ML pour la prédiction de churn
- Optimisation des algorithmes de recommandation
- Réduction de 30% du temps de traitement des données

ML Intern | StartupAI | 2021 - 2022
- Création de pipelines de données automatisés
- Implémentation de modèles NLP pour l'analyse de sentiment

Education:
Master in Computer Science | IIT Delhi | 2020 - 2022
Bachelor in Computer Science | Delhi University | 2016 - 2020
"""

# Question de l'utilisateur
user_question = "Quelles sont les compétences de Anuva ?"

# ============================================
# 📤 REQUÊTE À LLAMA
# ============================================

response = client.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""Voici le CV d'un candidat :

{cv_context}

Question : {user_question}

Réponds de manière structurée et professionnelle."""
        }
    ]
)

print("\n" + "=" * 60)
print("🤖 RÉPONSE DE L'AGENT IA")
print("=" * 60 + "\n")
print(response["message"]["content"])
print("\n" + "=" * 60)
