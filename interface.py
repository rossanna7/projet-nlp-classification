import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Charger le modèle et le tokenizer
try:
    model = load_model('best_model.keras') 
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle ou du tokenizer : {e}")

# Définir une fonction pour prédire la catégorie
def predict_category(text):
    max_length = 300  # Longueur maximale des séquences
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    categories = ['business', 'politics', 'sport', 'tech']
    probabilities = {categories[i]: float(prediction[0][i]) for i in range(len(categories))}
    category = categories[np.argmax(prediction)]
    confidence = np.max(prediction)
    return category, confidence,probabilities

# Personnaliser le style de la page
st.set_page_config(
    page_title="Document Classifier",
    page_icon="📰",
    layout="centered"
)

# En-tête de l'application
st.title("📰 Classification de Documents")
st.markdown("""
    Bienvenue dans l'application de classification de documents en Anglais. 
    Entrez un texte ci-dessous pour découvrir sa catégorie probable parmi :
    - **Tech**
    - **Business**
    - **Sport**
    - **Politics**
""")

# Ajouter une section pour l'entrée utilisateur
st.write("---")
st.subheader("💡 Entrez votre texte ci-dessous :")

# Champ de texte pour l'entrée utilisateur
user_input = st.text_area(
    "Votre texte :",
    placeholder="Tapez ou collez un texte ici, par exemple : The stock market saw a big rise today...",
    height=200
)

# Bouton de prédiction
if st.button("Prédire la catégorie"):
    if user_input.strip():
        if len(user_input.split()) > 300:
            st.error("⚠️ Votre texte est trop long. Veuillez saisir un texte de moins de 300 mots.")
        else:
            try:
                category, confidence,probabilities = predict_category(user_input)
                
                # Afficher les résultats
                st.write("### 📋 Résultat de la Classification")
                st.success(f"**Catégorie prédite : {category}**")
                st.write(f"🔍 **Confiance : {confidence * 100:.2f}%**")
                
                # Convertir `confidence` en float Python avant de l'utiliser dans st.progress
                st.progress(float(confidence))  # Conversion pour éviter l'erreur
                # Afficher un graphique interactif des probabilités
                fig = px.bar(
                    x=list(probabilities.keys()),
                    y=list(probabilities.values()),
                    labels={'x': 'Catégories', 'y': 'Probabilité'},
                    title="Distribution des Probabilités par Catégorie",
                    text_auto='.2s'
                )
                fig.update_traces(marker_color=['#4CAF50', '#FFC107', '#2196F3', '#FF5722'], textfont_size=12)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
    else:
        st.error("⚠️ Veuillez entrer un texte valide.")

# Ajouter un pied de page
st.write("---")
st.markdown("""
    **Développé par :** YNAA  
    🌟 Projet NLP - M2 HPC  
    🛠️ **Technologies utilisées :** TensorFlow, Streamlit
""")
