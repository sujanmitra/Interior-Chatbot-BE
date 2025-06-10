from flask import Flask, request
import chat

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chatter():
    if request.is_json:
        data = request.get_json()
        return chat.start_chat(data.get('path'), data.get('prompt'))
    else:
        return "Unsupported content type", 400

if __name__ == '__main__':
    app.run(port=8080, debug=True)