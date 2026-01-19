# api/index.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')

    reply = f"You said: {message}\nI'm alive!!"

    return jsonify({"reply": reply})

# Very important — this is what Vercel looks for
if __name__ == "__main__":
    app.run()
