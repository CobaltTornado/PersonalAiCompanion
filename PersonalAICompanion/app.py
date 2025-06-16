import os
import asyncio
import threading
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
from main_agent import ChiefArchitectAgent, ProgressManager


# Initialize Flask App and WebSocket
app = Flask(__name__)
app.config['SOCK_SERVER_OPTIONS'] = {'ping_interval': 25}
sock = Sock(app)

# --- Global State Management ---
progress_manager = ProgressManager()
agent_thread = None

# --- Agent Lifecycle Management ---
def run_agent_task(prompt_text, image_data, deep_reasoning):
    """Function to run in a separate thread."""
    agent = ChiefArchitectAgent(progress_manager)
    try:
        # asyncio.run() creates and manages the event loop for the coroutine.
        asyncio.run(agent.execute_task(prompt=prompt_text, image_data=image_data, deep_reasoning=deep_reasoning))
    except Exception as e:
        app.logger.error(f"Exception in agent thread: {e}", exc_info=True)


# --- HTTP Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Receives the chat message and starts the agent in a background thread."""
    global agent_thread
    data = request.json

    # --- OPTIONAL DEBUGGING LINE ---
    app.logger.info(
        f"Received data payload: { {key: value[:100] if isinstance(value, str) else value for key, value in data.items()} }")
    # --- END OF DEBUGGING LINE ---

    prompt = data.get('prompt')
    image_data = data.get('image_data')
    # Get the deep_reasoning flag from the request
    deep_reasoning = data.get('deep_reasoning', False)

    if not prompt and not image_data:
        return jsonify({"error": "Prompt or image is required."}), 400

    if agent_thread and agent_thread.is_alive():
        return jsonify({"status": "Agent is already running. Please wait."}), 429

    app.logger.info("Received chat request. Starting background agent thread.")
    # Pass the deep_reasoning flag to the agent task
    agent_thread = threading.Thread(target=run_agent_task, args=(prompt, image_data, deep_reasoning), daemon=True)
    agent_thread.start()

    return jsonify({"status": "Agent task has been started."})


# --- WebSocket Route ---
@sock.route('/ws')
def ws(socket):
    """Handles WebSocket connections for real-time updates."""
    app.logger.info("WebSocket client connected.")
    progress_manager.add_connection(socket)
    try:
        # Keep the connection alive by waiting for messages.
        while True:
            # The receive call will block until a message is received or the connection is closed.
            socket.receive(timeout=60 * 5) # Use a timeout to prevent blocking indefinitely
    except Exception:
        app.logger.info("WebSocket connection timed out or closed by client.")
    finally:
        progress_manager.remove_connection(socket)
        app.logger.info("WebSocket client removed from manager.")


# --- Main Execution ---
#if __name__ == '__main__':
    # Running without the reloader is more stable for threaded applications.
    #app.run(host='0.0.0.0', port=5000, debug=False)
