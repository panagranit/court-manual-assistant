from flask import Flask, request, jsonify, send_from_directory
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
from chat import retrieve_relevant_chunks, generate_answer

app = Flask(__name__, static_folder='public')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': "Missing 'message' field"}), 400
    relevant_chunks = retrieve_relevant_chunks(user_message, top_k=4)
    answer = generate_answer(user_message, relevant_chunks)
    return jsonify({'reply': answer})

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
