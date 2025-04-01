import sqlite3
import bcrypt
from database import get_connection
from face_auth import get_face_encoding  # Vous pouvez ajuster cette partie selon votre code

def register_user(username, email, password, captured_image):
    """
    Enregistre un utilisateur avec son mot de passe et son image faciale dans la base de données.
    """
    if check_user_exists(username, email):
        return "Erreur : Utilisateur ou email déjà existants."

    # Hacher le mot de passe
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    # Extraire l'encodage facial de l'image capturée (par exemple, en utilisant face_recognition)
    face_encoding = get_face_encoding(captured_image)  # Implémentez cette fonction selon votre logique
    if face_encoding is None:
        return "Erreur dans la détection du visage."

    # Connexion à la base de données et insertion des données
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, email, password, face_encoding) VALUES (?, ?, ?, ?)", 
                   (username, email, hashed_password.decode('utf-8'), face_encoding.tobytes()))
    conn.commit()
    conn.close()
    
    return f"Utilisateur {username} inscrit avec succès."

def check_user_exists(username, email):
    """
    Vérifie si l'utilisateur existe déjà dans la base de données.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username FROM users WHERE username = ? OR email = ?", (username, email))
    result = cursor.fetchone()
    conn.close()
    return result is not None
