import sqlite3
import os

# Make sure the database path matches your backend code
DB_PATH = r"C:\Users\viren\Desktop\Major Project phase 1\data\sessions.db"

# Ensure directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Create the sessions table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    duration INTEGER NOT NULL,
    focus_score INTEGER NOT NULL,
    subject TEXT DEFAULT 'Study Session'
)
""")

conn.commit()
conn.close()

print("✅ Database initialized successfully at:", DB_PATH)
