from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os


load_dotenv()


app = Flask(__name__)
CORS(app)


# Initialize your AI components (same as before)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    "faiss_medical_index",
    embeddings,
    allow_dangerous_deserialization=True 
)


llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0.3,
    max_tokens=1000
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
)


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        
        # Translate AR → EN
        translated_query = GoogleTranslator(source='ar', target='en').translate(user_query)
        
        # Get response
        response = qa.run(translated_query)
        
        # Translate EN → AR
        translated_response = GoogleTranslator(source='en', target='ar').translate(response)
        
        return jsonify({
            "success": True,
            "response": translated_response
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)



