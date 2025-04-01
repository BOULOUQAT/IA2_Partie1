import sqlite3

def get_connection():
    """
    Crée une connexion à la base de données SQLite.
    """
    return sqlite3.connect("authentication.db")

def setup_database():
    """
    Crée la base de données et la table 'users' pour stocker les images des utilisateurs.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Créer la table des utilisateurs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            face_image BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialisation
setup_database()
