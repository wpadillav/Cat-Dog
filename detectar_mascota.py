import os
import sys
import cv2
import numpy as np

# Verificar versión de Python
print(f"Python versión: {sys.version}")

try:
    import tensorflow as tf
    print(f"TensorFlow versión: {tf.__version__}")
except ImportError:
    print("Error: TensorFlow no está instalado.")
    print("Ejecuta: pip install tensorflow-macos tensorflow-metal")
    sys.exit(1)

# Verificar versión de TensorFlow
if tf.__version__ < "2.0":
    print("Error: Se requiere TensorFlow 2.x")
    print(f"Versión actual: {tf.__version__}")
    sys.exit(1)

# Configuración para GPU en Mac
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU detectada y configurada correctamente")
    else:
        print("No se detectó GPU, usando CPU")
except Exception as e:
    print(f"Error al configurar GPU: {str(e)}")
    print("Continuando con CPU...")

from collections import Counter

def preparar_imagen(frame, altura=150, ancho=150):
    """Prepara la imagen para la predicción"""
    # Convertir a RGB ya que OpenCV usa BGR
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Redimensionar manteniendo el aspecto
    img = cv2.resize(img, (altura, ancho))
    # Normalizar
    img = img.astype(np.float32) / 255.0
    # Expandir dimensiones para el modelo
    img = np.expand_dims(img, axis=0)
    return img

def mostrar_prediccion(frame, prediccion, contadores):
    """Muestra la predicción en el frame"""
    # Obtener probabilidad
    prob = prediccion[0][0]
    
    # Ajustar umbrales de detección
    UMBRAL_CONFIANZA = 0.85  # Aumentamos el umbral de confianza
    
    if prob < 0.3:  # Es más probable que sea gato
        texto = "Gato"
        confianza = 1 - prob
        color = (0, 255, 0)  # Verde
    elif prob > 0.7:  # Es más probable que sea perro
        texto = "Perro"
        confianza = prob
        color = (0, 0, 255)  # Rojo
    else:
        texto = "No detectado"
        confianza = max(prob, 1-prob)
        color = (128, 128, 128)  # Gris
    
    # Actualizar contadores solo si la confianza es alta
    if confianza > UMBRAL_CONFIANZA:
        contadores[texto] += 1
    
    # Mostrar información en pantalla
    cv2.putText(frame, f"{texto}: {confianza:.2%}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Mostrar contadores
    cv2.putText(frame, f"Gatos: {contadores['Gato']}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Perros: {contadores['Perro']}", 
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Dibujar rectángulo solo si hay detección clara
    if confianza > UMBRAL_CONFIANZA:
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, height), color, 2)

def main():
    # Verificar si existe el modelo
    modelo_path = 'modelo_gatos_perros.h5'
    if not os.path.exists(modelo_path):
        print("Error: No se encuentra el modelo entrenado.")
        print("Primero ejecuta entrenar_modelo.py")
        return

    # Inicializar contadores
    contadores = Counter({'Gato': 0, 'Perro': 0})

    try:
        # Cargar el modelo con manejo de errores específico
        print("Cargando modelo...")
        try:
            modelo = tf.keras.models.load_model(modelo_path, compile=False)
            modelo.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            print("Reentrenando el modelo...")
            # Si el modelo no se puede cargar, necesitas reentrenarlo
            print("Por favor, ejecuta primero: python entrenar_modelo.py")
            return

        # Inicializar la cámara
        print("Iniciando cámara...")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("Error: No se puede acceder a la cámara")
            return

        print("Detector iniciado. Presiona 'ESC' para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se puede obtener el frame")
                break
            
            # Preparar la imagen
            img = preparar_imagen(frame)
            
            # Realizar predicción con menos verbosidad
            prediccion = modelo.predict(img, verbose=0)
            
            # Mostrar resultado
            mostrar_prediccion(frame, prediccion, contadores)
            
            # Mostrar frame
            cv2.imshow('Detector de Mascotas', frame)
            
            # Esperar más tiempo entre frames (30ms) y detectar ESC
            tecla = cv2.waitKey(30) & 0xFF
            if tecla == 27:  # 27 es el código ASCII de ESC
                print("\nCerrando el detector...")
                break

        # Mostrar resultados finales
        print("\nResultados finales:")
        print(f"Total de gatos detectados: {contadores['Gato']}")
        print(f"Total de perros detectados: {contadores['Perro']}")

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
    
    finally:
        # Liberar recursos
        print("Liberando recursos...")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()