from flask import Flask, render_template, request, jsonify
from rag_engine import RAGEngine

app = Flask(__name__)

# Configuration
CHROMA_PATH = r"C:\Users\R I B\Desktop\agent_ia\chroma_db"
COLLECTION_NAME = "cv_segments"

# Initialiser RAG Engine
try:
    rag_engine = RAGEngine(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME)
    print("✅ RAG Engine initialisé avec succès")
except Exception as e:
    print(f"❌ Erreur initialisation RAG Engine: {e}")
    rag_engine = None


@app.route('/')
def index():
    """Page principale"""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])  # ✅ Changé de /query à /ask
def ask():
    """Endpoint pour traiter les questions"""
    if not rag_engine:
        return jsonify({
            'error': 'RAG Engine non initialisé',
            'answer': '❌ Erreur système'  # ✅ Changé 'response' en 'answer'
        }), 500

    data = request.get_json()
    user_query = data.get('question', '').strip()

    if not user_query:
        return jsonify({
            'error': 'Question vide',
            'answer': '⚠️ Veuillez poser une question'  # ✅ Changé 'response' en 'answer'
        }), 400

    try:
        # Obtenir la réponse du RAG
        response = rag_engine.query(user_query, top_k=10)

        return jsonify({
            'success': True,
            'answer': response,  # ✅ Changé 'response' en 'answer'
            'question': user_query
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'answer': f'❌ Erreur lors du traitement: {e}'  # ✅ Changé 'response' en 'answer'
        }), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
