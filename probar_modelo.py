import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import argparse
import os
import sys

# Parámetros del modelo (deben coincidir con los del entrenamiento)
IMG_WIDTH = 150
IMG_HEIGHT = 150
CLASSES = ["Gato", "Perro"]
MODEL_PATH = "modelo_gatos_perros.keras"

def preprocesar_frame(frame):
    """
    Convierte un frame BGR de OpenCV a RGB, lo redimensiona y normaliza.
    """
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def inicializar_camara(tipo="auto"):
    """
    Inicializa la cámara dependiendo del entorno o del parámetro forzado.
    """
    if tipo == "pi":
        print("Usando Pi Camera (CSI)...")
        return cv2.VideoCapture(0, cv2.CAP_V4L2)
    elif tipo == "usb":
        print("Usando cámara USB o laptop...")
        return cv2.VideoCapture(0)
    else:  # auto
        if os.uname().machine.startswith("arm") and os.path.exists("/dev/video0"):
            print("Detectada Raspberry Pi: usando Pi Camera (CSI)...")
            return cv2.VideoCapture(0, cv2.CAP_V4L2)
        else:
            print("Entorno estándar: usando cámara USB o laptop...")
            return cv2.VideoCapture(0)

def main():
    # Argumentos desde línea de comandos
    parser = argparse.ArgumentParser(description="Clasificador en tiempo real Gato/Perro")
    parser.add_argument("--cam", choices=["auto", "usb", "pi"], default="auto", help="Tipo de cámara a usar")
    args = parser.parse_args()

    # Cargar modelo
    print("Cargando modelo...")
    try:
        modelo = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        sys.exit(1)

    # Inicializar cámara
    cap = inicializar_camara(args.cam)
    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        sys.exit(1)

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar imagen de la cámara.")
            break

        # Clasificar frame
        entrada = preprocesar_frame(frame)
        pred = modelo.predict(entrada)[0][0]
        clase = CLASSES[1] if pred >= 0.5 else CLASSES[0]
        confianza = pred if pred >= 0.5 else 1 - pred
        texto = f"{clase} ({confianza:.2%})"

        # Mostrar resultado
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Clasificador Gato/Perro", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Cerrando cámara...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
