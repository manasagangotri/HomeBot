


from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import requests
from config import DB_CONFIG

app = Flask(__name__)
CORS(app)

# 🟢 User Registration
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    username = data.get('username')
    password = data.get('password')

    if not all([name, email, username, password]):
        return jsonify({"error": "All fields are required"}), 400
    
    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"error": "Username or Email already exists"}), 400
        
        cursor.execute("INSERT INTO users (name, email, username, password) VALUES (%s, %s, %s, %s)", 
                       (name, email, username, password))
        db.commit()

        cursor.close()
        db.close()

        return jsonify({"message": "User registered successfully"}), 201
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🟢 User Login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Both username and password are required"}), 400

    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()

        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        cursor.close()
        db.close()

        if user:
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"error": "Invalid credentials"}), 401

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")
        user_id = data.get("username")  # Make sure frontend sends this
        print(user_message)
        if not user_message or not user_id:
            return jsonify({"error": "Message and User ID are required"}), 400

        # 🟡 Send message to Rasa
        rasa_url = "http://localhost:5005/webhooks/rest/webhook"
        response = requests.post(rasa_url, json={"sender": user_id, "message": user_message})

       # print("response"+response)

        if response.status_code == 200:
            messages = response.json()
            print(messages)
            bot_reply = ""
            buttons = []
            
            # 🟡 Extract text and buttons
            for msg in messages:
                if "text" in msg:
                    bot_reply += msg["text"] + "\n"
                if "buttons" in msg:
                    buttons = msg["buttons"]
            
        
            # 🟢 Save to MySQL
            try:
                db = mysql.connector.connect(**DB_CONFIG)
                cursor = db.cursor()

                cursor.execute(
                    "INSERT INTO chat_history (user_id, message, response) VALUES (%s, %s, %s)",
                    (user_id, user_message, bot_reply.strip())
                )
                db.commit()

                cursor.close()
                db.close()

            except Exception as db_error:
                print("Failed to save chat history:", db_error)

            # ✅ Return response to frontend
            return jsonify({
                "reply": bot_reply.strip(),
                "buttons": buttons  # Just send, not save
            })

        return jsonify({"error": "Failed to connect to chatbot"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
'''

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")
        user_id = data.get("username")  # 🛑 Make sure your frontend sends user_id

        if not user_message or not user_id:
            return jsonify({"error": "Message and User ID are required"}), 400

        rasa_url = "http://localhost:5005/webhooks/rest/webhook"
        response = requests.post(rasa_url, json={"sender": "user", "message": user_message})

        if response.status_code == 200:
            messages = response.json()
            

    # Save to DB here if needed...
            if not messages:
                bot_reply = "I'm sorry, I couldn't understand that."
            else:
                bot_reply = " ".join(msg.get('text', '') for msg in messages)

            # ✅ Save user message and bot reply into the database
            try:
                db = mysql.connector.connect(**DB_CONFIG)
                cursor = db.cursor()

                cursor.execute(
                    "INSERT INTO chat_history (user_id, message, response) VALUES (%s, %s, %s)",
                    (user_id, user_message, bot_reply)
                )
                db.commit()

                cursor.close()
                db.close()

            except Exception as db_error:
                print("Failed to save chat history:", db_error)

            return jsonify({"reply": bot_reply})
        
        return jsonify({"error": "Failed to connect to chatbot"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
'''

'''@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message")
        user_id = data.get("username")

        # Optional feedback fields
        feedback_type = data.get("feedback_type")  # "like" or "dislike"
        suggestion = data.get("suggestion")

        if not user_message or not user_id:
            return jsonify({"error": "Message and User ID are required"}), 400

        rasa_url = "http://localhost:5005/webhooks/rest/webhook"
        response = requests.post(rasa_url, json={"sender": user_id, "message": user_message})

        if response.status_code == 200:
            messages = response.json()

            if not messages:
                bot_reply = "I'm sorry, I couldn't understand that."
                buttons = []
            else:
                bot_reply = " ".join(msg.get('text', '') for msg in messages if msg.get('text'))
                buttons = []
                for msg in messages:
                    if 'buttons' in msg:
                        buttons.extend(msg['buttons'])

            try:
                # Save chat history
                db = mysql.connector.connect(**DB_CONFIG)
                cursor = db.cursor()
                cursor.execute(
                    "INSERT INTO chat_history (user_id, message, response) VALUES (%s, %s, %s)",
                    (user_id, user_message, bot_reply)
                )

                # Save feedback if provided
                if feedback_type:
                    cursor.execute(
                        "INSERT INTO feedback1 (username, feedback_type, bot_reply, suggestion) VALUES (%s, %s, %s, %s)",
                        (user_id, feedback_type, bot_reply, suggestion)
                    )

                db.commit()
                cursor.close()
                db.close()
            except Exception as db_error:
                print("Failed to save chat or feedback:", db_error)

            return jsonify({
                "reply": bot_reply,
                "buttons": buttons
            })

        return jsonify({"error": "Failed to connect to chatbot"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
'''

@app.route('/feedback', methods=['POST'])
def receive_feedback():
    data = request.get_json()

    username = data.get("username")
    message = data.get("message")
    feedback_type = data.get("feedback")
    suggestion = data.get("suggestion", "")

    if not (username and message and feedback_type):
        return jsonify({"error": "Missing fields"}), 400

    try:
        db = mysql.connector.connect(**DB_CONFIG)
        cursor = db.cursor()
        cursor.execute(
            """
            INSERT INTO feedback1 (username, bot_reply, feedback_type, suggestion)
            VALUES (%s, %s, %s, %s)
            """,
            (username, message, feedback_type, suggestion),
        )
        db.commit()
        return jsonify({"message": "Feedback submitted"}), 200
    except Exception as e:
        print("Error inserting feedback:", e)
        return jsonify({"error": "Database error"}), 500


@app.route("/admin/feedback", methods=["GET"])
def get_feedback():
    try:
        db = mysql.connector.connect(**DB_CONFIG)

        cursor = db.cursor(dictionary=True)  # dictionary=True for JSON
        cursor.execute("SELECT * FROM feedback1 ORDER BY created_at DESC")
        feedback_data = cursor.fetchall()
        cursor.close()
        return jsonify(feedback_data), 200
    except Exception as e:
        print("Error fetching feedback:", e)
        return jsonify({"error": "Database error"}), 500

# 🏃 Start Flask Server
if __name__ == '__main__':
    app.run(debug=True)


