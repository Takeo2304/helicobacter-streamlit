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
import unicodedata

# ---------- CARGAR Y PROCESAR DATOS ----------
st.title("🧬 Predicción de *Helicobacter spp.* en Animales de Granja")

st.markdown("""
Este modelo predice qué especie de *Helicobacter* puede encontrarse en un animal de granja
según el país, animal, tipo de muestra y año de detección.
""")

# Ruta al archivo Excel
excel_path = os.path.join(os.path.dirname(__file__), "helicobacter_data.xlsx")
df = pd.read_excel(excel_path, engine='openpyxl')

# Función para limpiar nombres de columnas
def normalizar_columna(col):
    col = ''.join((c for c in unicodedata.normalize('NFD', col) if unicodedata.category(c) != 'Mn'))  # quita tildes
    return col.strip().capitalize()

df.columns = [normalizar_columna(col) for col in df.columns]

# Renombrar "Ano" a "Año" explícitamente si aparece
df.rename(columns={"Ano": "Año"}, inplace=True)

# Mostrar columnas para depuración
#st.write("🧾 Columnas detectadas:", df.columns.tolist())

# Verificar que estén todas las columnas necesarias
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

# Guardar modelo y codificadores
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
    animal_enc = le_animal.transform([animal])[0]
    pais_enc = le_pais.transform([pais])[0]
    muestra_enc = le_muestra.transform([muestra])[0]

    entrada = np.array([[animal_enc, pais_enc, muestra_enc, anio]])
    pred = modelo.predict(entrada)
    especie_predicha = le_especie.inverse_transform(pred)[0]

    st.success(f"✅ Especie predicha: **{especie_predicha}**")

y_pred_test = modelo.predict(X_test)
report = classification_report(y_test, y_pred_test, target_names=le_especie.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.subheader("🔎 Desempeño del Modelo")
st.dataframe(df_report)

# ---------- VISUALIZAR MATRIZ DE CONFUSIÓN ----------
st.header("Desempeño del Modelo")


#y_pred = modelo.predict(X_test)
#reporte = classification_report(y_test, y_pred, target_names=le_especie.classes_, output_dict=True)
#df_reporte = pd.DataFrame(reporte).transpose()
#st.dataframe(df_reporte.style.format(precision=2))

#cm = confusion_matrix(y_test, y_pred)
#fig, ax = plt.subplots()
#sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
 #           xticklabels=le_especie.classes_,
  #          yticklabels=le_especie.classes_,
   #         ax=ax)
#plt.ylabel("Especie real")
#plt.xlabel("Especie predicha")
#st.pyplot(fig)

mat = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots()
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_especie.classes_,
            yticklabels=le_especie.classes_)
plt.xlabel("Especie predicha")
plt.ylabel("Especie real")
st.pyplot(fig)