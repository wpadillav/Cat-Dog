import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Parámetros ===
directorio_pruebas = os.path.join(os.path.dirname(__file__), "pruebas")
modelo_path = "modelo_gatos_perros.keras"
altura, ancho = 150, 150

# === Verificar existencia del modelo ===
if not os.path.exists(modelo_path):
    print(f"❌ Modelo no encontrado: {modelo_path}")
    exit(1)

# === Cargar el modelo ===
modelo = load_model(modelo_path)
print(f"✅ Modelo cargado: {modelo_path}")

# === Buscar imágenes en el directorio de pruebas ===
imagenes = sorted([
    f for f in os.listdir(directorio_pruebas)
    if f.lower().endswith(".jpg")
])

if not imagenes:
    print(f"❌ No se encontraron imágenes .jpg en {directorio_pruebas}")
    exit(1)

# === Predecir y mostrar resultados ===
resultados = []

for nombre_img in imagenes:
    ruta = os.path.join(directorio_pruebas, nombre_img)

    try:
        imagen = load_img(ruta, target_size=(altura, ancho))
        imagen = img_to_array(imagen)
        imagen = imagen / 255.0  # Normalizar
        imagen = np.expand_dims(imagen, axis=0)

        prediccion = modelo.predict(imagen, verbose=0)[0][0]
        etiqueta = "Perro 🐶" if prediccion > 0.5 else "Gato 🐱"

        resultados.append((nombre_img, etiqueta))
    except Exception as e:
        resultados.append((nombre_img, f"Error: {str(e)}"))

# === Mostrar resumen ===
print("\n📋 Resultados de predicción:\n")
for archivo, pred in resultados:
    print(f"{archivo} → {pred}")
