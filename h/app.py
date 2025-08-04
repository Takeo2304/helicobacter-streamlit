import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import os
import unicodedata

# ---------- CARGAR Y PROCESAR DATOS ----------
st.title("Predicción de la distribucion de *Helicobacter spp.* en Animales de Granja")

st.markdown("""
Este modelo predice qué especie de *Helicobacter* puede encontrarse en un animal de granja
según el país, animal, tipo de muestra y año de detección.
""")
st.markdown("""
Priorizar diagnósticos en laboratorios veterinarios.

Detectar patrones epidemiológicos para vigilancia tipo One Health.

Aplicar IA en microbiología, automatizando análisis repetitivos.

Escalar el sistema incluyendo genética, ambiente o metadatos clínicos.
""")

# Ruta al archivo Excel
excel_path = os.path.join(os.path.dirname(__file__), "helicobacter_data.xlsx")
df = pd.read_excel(excel_path, engine='openpyxl')


# Función para limpiar nombres de columnas
def normalizar_columna(col):
    col = ''.join((c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn'))
    return col.strip().capitalize()


df.columns = [normalizar_columna(col) for col in df.columns]
df.rename(columns={"Ano": "Año"}, inplace=True)

columnas_necesarias = ["Animal", "Pais", "Muestra", "Año", "Especie"]
faltantes = [col for col in columnas_necesarias if col not in df.columns]

if faltantes:
    st.error(f"❌ Faltan columnas en el archivo Excel: {faltantes}")
    st.stop()

# ---------- CODIFICACIÓN ----------
le_animal = LabelEncoder()
le_pais = LabelEncoder()
le_muestra = LabelEncoder()
le_especie = LabelEncoder()

df["Animal_encoded"] = le_animal.fit_transform(df["Animal"])
df["Pais_encoded"] = le_pais.fit_transform(df["Pais"])
df["Muestra_encoded"] = le_muestra.fit_transform(df["Muestra"])
df["Especie_encoded"] = le_especie.fit_transform(df["Especie"])

# ---------- ENTRENAMIENTO ----------
X = df[["Animal_encoded", "Pais_encoded", "Muestra_encoded", "Año"]]
y = df["Especie_encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

joblib.dump(modelo, "modelo_helicobacter.pkl")
joblib.dump(le_animal, "le_animal.pkl")
joblib.dump(le_pais, "le_pais.pkl")
joblib.dump(le_muestra, "le_muestra.pkl")
joblib.dump(le_especie, "le_especie.pkl")

# ---------- FORMULARIO PARA PREDICCIÓN ----------
st.header("Ingresar datos para predicción")

animal = st.selectbox("Animal", le_animal.classes_)
pais = st.selectbox("País", le_pais.classes_)
muestra = st.selectbox("Tipo de muestra", le_muestra.classes_)
anio = st.slider("Año", 2000, 2025, 2025)

if st.button("Predecir especie"):
    animal_enc = le_animal.transform([animal])[0]
    pais_enc = le_pais.transform([pais])[0]
    muestra_enc = le_muestra.transform([muestra])[0]
    entrada = np.array([[animal_enc, pais_enc, muestra_enc, anio]])

    pred = modelo.predict(entrada)
    especie_predicha = le_especie.inverse_transform(pred)[0]

    # Mostrar probabilidades
    proba = modelo.predict_proba(entrada)[0]
    especies = le_especie.inverse_transform(np.arange(len(proba)))
    proba_df = pd.DataFrame({
        "Especie": especies,
        "Probabilidad": proba
    }).sort_values(by="Probabilidad", ascending=False)

    st.success(f"✅ Especie predicha: **{especie_predicha}**")
    st.write(" Probabilidades de predicción:")
    st.dataframe(proba_df)

    fig_proba = px.bar(proba_df, x="Especie", y="Probabilidad", title="Probabilidades por especie")
    st.plotly_chart(fig_proba)

# ---------- DESEMPEÑO DEL MODELO ----------
y_pred_test = modelo.predict(X_test)
report = classification_report(y_test, y_pred_test, target_names=le_especie.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()

#st.subheader("Desempeño del Modelo")
#st.dataframe(df_report)

# ---------- MATRIZ DE CONFUSIÓN INTERACTIVA ----------
mat = confusion_matrix(y_test, y_pred_test)
z = mat.tolist()
etiquetas = le_especie.classes_.tolist()

fig_cm = ff.create_annotated_heatmap(
    z=z,
    x=etiquetas,
    y=etiquetas,
    colorscale='Blues',
    showscale=True,
    annotation_text=[[str(val) for val in row] for row in z],
    font_colors=["black"]
)

fig_cm.update_layout(
    title="🔷 Matriz de Confusión Interactiva",
    xaxis_title="Especie predicha",
    yaxis_title="Especie real"
)

st.plotly_chart(fig_cm)