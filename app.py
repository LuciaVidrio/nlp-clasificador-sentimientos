import streamlit as st
import joblib
import re

# Cargar modelo, vectorizador y codificador
modelo = joblib.load("modelo_svm.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# Función para limpiar texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

# Función de predicción
def predecir_sentimiento(texto):
    texto_limpio = limpiar_texto(texto)
    texto_vect = vectorizer.transform([texto_limpio])
    pred = modelo.predict(texto_vect)
    return le.inverse_transform(pred)[0]

# Interfaz con Streamlit
st.title("Clasificador de Sentimiento de Reseñas")
st.write("Escribe una reseña de producto y te diremos si es positiva, negativa o neutra.")

# Entrada del usuario
entrada = st.text_area("Escribe tu reseña aquí:")

# Botón de predicción
if st.button("Clasificar"):
    if entrada.strip() == "":
        st.warning("Por favor, escribe una reseña.")
    else:
        resultado = predecir_sentimiento(entrada)
        st.success(f"Sentimiento detectado: **{resultado.upper()}**")
