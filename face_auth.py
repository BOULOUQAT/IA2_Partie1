import sqlite3
import numpy as np
import pickle
import face_recognition
from database import get_connection

def extract_facial_features(image_path):
    """
    Extrait les caractéristiques faciales d'une image.
    """
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques faciales : {e}")
        return None

def register_facial_features(username, image_path):
    """
    Associe des caractéristiques faciales à un utilisateur existant.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Extraction des caractéristiques faciales
    feature = extract_facial_features(image_path)
    if feature is not None:
        try:
            # Sérialisation des caractéristiques
            feature_blob = pickle.dumps(feature)
            cursor.execute("UPDATE users SET feature = ? WHERE username = ?", 
                           (feature_blob, username))
            conn.commit()
            return "Facial features registered successfully!"
        except sqlite3.Error as e:
            return f"Erreur lors de l'enregistrement des caractéristiques : {e}"
        finally:
            conn.close()
    return "No face detected in the image."

def login_with_face(image_path):
    """
    Authentifie un utilisateur via reconnaissance faciale.
    """
    # Extraction des caractéristiques faciales de l'image en direct
    query_feature = extract_facial_features(image_path)
    if query_feature is None:
        return "No face detected in the image."

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Récupération des caractéristiques faciales des utilisateurs
        cursor.execute("SELECT username, feature FROM users WHERE feature IS NOT NULL")
        users = cursor.fetchall()
    except sqlite3.Error as e:
        return f"Erreur lors de la récupération des utilisateurs : {e}"
    finally:
        conn.close()

    # Comparaison des caractéristiques
    for username, stored_feature_blob in users:
        stored_feature = pickle.loads(stored_feature_blob)
        distance = np.linalg.norm(stored_feature - query_feature)
        if distance < 0.6:  # Seuil de similarité
            return f"Facial login successful! Welcome, {username}."
    return "No matching face found."
