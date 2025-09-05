import sqlite3
import streamlit as st
import google.generativeai as genai

# -------------------------------
# Configure Gemini API
# -------------------------------
genai.configure(api_key="AIzaSyArdi2vVipU5Zi-xnxaJgcRvRN7G-_HKIk")  # replace with your key
model = genai.GenerativeModel("gemini-1.5-flash")

# -------------------------------
# Database Setup
# -------------------------------
def create_tables():
    conn = sqlite3.connect("ceramic_factory.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS inventory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product_name TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        location TEXT)''')
    conn.commit()
    conn.close()

def add_item(product_name, quantity, location):
    conn = sqlite3.connect("ceramic_factory.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO inventory (product_name, quantity, location) VALUES (?, ?, ?)",
                   (product_name, quantity, location))
    conn.commit()
    conn.close()

def get_inventory():
    conn = sqlite3.connect("ceramic_factory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inventory")
    rows = cursor.fetchall()
    conn.close()
    return rows

# -------------------------------
# Chatbot Function (Gemini)
# -------------------------------
def chatbot_response(user_input):
    prompt = f"""You are an AI assistant for a Ceramic Manufacturing Factory. 
    User asked: {user_input} 
    Answer clearly and in a helpful way."""
    
    response = model.generate_content(prompt)
    return response.text

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Ceramic Factory Dashboard", layout="wide")
    st.title("🏭 Ceramic Factory Smart Dashboard")

    # Sidebar Navigation
    menu = ["Inventory", "Add Inventory", "Chatbot"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Inventory Viewer
    if choice == "Inventory":
        st.subheader("📦 Current Inventory")
        rows = get_inventory()
        if rows:
            st.table(rows)
        else:
            st.info("No inventory data available.")

    # Add Inventory
    elif choice == "Add Inventory":
        st.subheader("➕ Add New Inventory")
        product_name = st.text_input("Product Name")
        quantity = st.number_input("Quantity", min_value=0)
        location = st.text_input("Location (e.g., Warehouse A)")

        if st.button("Add Item"):
            add_item(product_name, quantity, location)
            st.success(f"✅ {product_name} added successfully!")

    # Chatbot
    elif choice == "Chatbot":
        st.subheader("💬 AI Assistant")
        user_input = st.text_input("Ask me anything about your factory or stock")
        if st.button("Send"):
            if user_input.strip():
                response = chatbot_response(user_input)
                st.write("🤖 **AI:**", response)
            else:
                st.warning("Please enter a message.")

# Run app
if __name__ == "__main__":
    create_tables()
    main()
