from flask import Flask, render_template, request, jsonify
import importlib.util
import sys
import os

app = Flask(__name__)

# Dynamically import 'rag project' module because of the space in the filename
current_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_dir, "rag project.py")
spec = importlib.util.spec_from_file_location("rag_project", module_path)
rag_project = importlib.util.module_from_spec(spec)
sys.modules["rag_project"] = rag_project
spec.loader.exec_module(rag_project)

# Global chat history
chat_history = [{"role": "system", "content": "you are the chatbot of Royal Spice Resturant"}]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Append user message to history
    chat_history.append({"role": "user", "content": user_message})
    try:
        # Get response from the RAG model
        answer, source_docs = rag_project.get_bot_response(user_message, chat_history[-10:])
        
        # Append bot response to history
        chat_history.append({"role": "assistant", "content": answer})
        
        return jsonify({"response": answer})
    except Exception as e:
        print(f"Error during RAG generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
