import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------- CARGAR Y PROCESAR DATOS ----------
st.title("🧬 Predicción de *Helicobacter spp.* en Animales de Granja")

st.markdown("""
Este modelo predice qué especie de *Helicobacter* puede encontrarse en un animal de granja
según el país, animal, tipo de muestra y año de detección.
""")


csv_path = os.path.join(os.path.dirname(__file__), "helicobacter_data.csv")
df = pd.read_csv(csv_path, sep=";")

#df = pd.read_csv("helicobacter_data.csv",encoding="windows-1252", sep=";")


# Codificar columnas
le_animal = LabelEncoder()
le_pais = LabelEncoder()
le_muestra = LabelEncoder()
le_especie = LabelEncoder()

df["Animal_encoded"] = le_animal.fit_transform(df["Animal"])
df["País_encoded"] = le_pais.fit_transform(df["País"])
df["Muestra_encoded"] = le_muestra.fit_transform(df["Muestra"])
df["Especie_encoded"] = le_especie.fit_transform(df["Especie"])

# Entrenar modelo
X = df[["Animal_encoded", "Pais_encoded", "Muestra_encoded", "Año"]]
y = df["Especie_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Guardar objetos
joblib.dump(modelo, "modelo_helicobacter.pkl")
joblib.dump(le_animal, "le_animal.pkl")
joblib.dump(le_pais, "le_pais.pkl")
joblib.dump(le_muestra, "le_muestra.pkl")
joblib.dump(le_especie, "le_especie.pkl")

# ---------- FORMULARIO PARA PREDICCIÓN ----------
st.header("🔍 Ingresar datos para predicción")

animal = st.selectbox("Animal", le_animal.classes_)
pais = st.selectbox("País", le_pais.classes_)
muestra = st.selectbox("Tipo de muestra", le_muestra.classes_)
anio = st.slider("Año", 2000, 2025, 2025)

if st.button("Predecir especie"):
    # Codificar entrada
    animal_enc = le_animal.transform([animal])[0]
    pais_enc = le_pais.transform([pais])[0]
    muestra_enc = le_muestra.transform([muestra])[0]

    entrada = np.array([[animal_enc, pais_enc, muestra_enc, anio]])
    pred = modelo.predict(entrada)
    especie_predicha = le_especie.inverse_transform(pred)[0]

    st.success(f"✅ Especie predicha: **{especie_predicha}**")

# ---------- VISUALIZAR MATRIZ DE CONFUSIÓN ----------
st.header("📊 Desempeño del Modelo")

y_pred = modelo.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True, target_names=le_especie.classes_)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le_especie.classes_, yticklabels=le_especie.classes_, cmap="Blues", ax=ax)
plt.ylabel("Real")
plt.xlabel("Predicho")
st.pyplot(fig)
