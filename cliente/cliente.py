import cv2
import numpy as np
import tensorflow as tf

# Cargar modelo entrenado
modelo = tf.keras.models.load_model('modelo_gatos_perros.keras')
print("✅ Modelo cargado")

# Tamaño esperado por el modelo
altura, ancho = 150, 150

# Abrir la cámara (0 = cámara principal)
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("🎥 Cámara abierta. Presiona 'q' para salir.")

while True:
    ret, frame = camara.read()
    if not ret:
        print("❌ Error al capturar frame.")
        break

    # Mostrar el frame original
    cv2.imshow("Entrada", frame)

    # Preprocesar la imagen
    imagen = cv2.resize(frame, (ancho, altura))
    imagen = imagen.astype('float32') / 255.0
    imagen = np.expand_dims(imagen, axis=0)

    # Hacer predicción
    prediccion = modelo.predict(imagen, verbose=0)[0][0]
    etiqueta = "🐶 Perro" if prediccion >= 0.5 else "🐱 Gato"
    confianza = f"{prediccion:.2f}" if prediccion >= 0.5 else f"{1 - prediccion:.2f}"

    # Mostrar resultado en pantalla
    resultado = f"{etiqueta} ({confianza})"
    cv2.putText(frame, resultado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Clasificacion", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()
