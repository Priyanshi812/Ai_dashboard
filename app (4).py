import streamlit as st
import sqlite3
from datetime import datetime

# Connect to SQLite database
conn = sqlite3.connect("ceramic_factory.db", check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS stock (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT,
    quantity INTEGER,
    last_updated TIMESTAMP
)
""")
conn.commit()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Chatbot"])

# -------------------- Dashboard Page --------------------
if page == "Dashboard":
    st.title("📊 Ceramic Factory Dashboard")

    # Add Stock
    st.subheader("Add Stock")
    item_name = st.text_input("Item Name")
    quantity = st.number_input("Quantity", min_value=0, step=1)

    if st.button("Add/Update Stock"):
        cursor.execute("SELECT * FROM stock WHERE item_name=?", (item_name,))
        result = cursor.fetchone()
        if result:
            cursor.execute(
                "UPDATE stock SET quantity=?, last_updated=? WHERE item_name=?",
                (quantity, datetime.now(), item_name)
            )
        else:
            cursor.execute(
                "INSERT INTO stock (item_name, quantity, last_updated) VALUES (?, ?, ?)",
                (item_name, quantity, datetime.now())
            )
        conn.commit()
        st.success("Stock updated successfully!")

    # Show Stock Table
    st.subheader("Current Stock")
    cursor.execute("SELECT * FROM stock")
    rows = cursor.fetchall()
    if rows:
        st.table(rows)
    else:
        st.write("No stock data available.")

# -------------------- Chatbot Page --------------------
elif page == "Chatbot":
    st.title("💬 Ceramic Factory Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        st.write(msg)

    # User input
    user_input = st.text_input("Ask me something about stock:")

    if st.button("Send"):
        if user_input:
            st.session_state["messages"].append(f"👨‍💼 You: {user_input}")

            # Simple chatbot logic
            if "stock" in user_input.lower():
                cursor.execute("SELECT item_name, quantity FROM stock")
                stock_data = cursor.fetchall()
                if stock_data:
                    reply = "📦 Current stock:\n"
                    for item, qty in stock_data:
                        reply += f"- {item}: {qty}\n"
                else:
                    reply = "No stock data available."
            else:
                reply = "I can help you with stock queries. Try asking 'What is my stock?'"

            st.session_state["messages"].append(f"🤖 Bot: {reply}")
