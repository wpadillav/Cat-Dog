import os
import sys

# Verificar todas las dependencias necesarias
dependencias = {
    'tensorflow': None,
    'matplotlib': None,
    'PIL': None,
    'cv2': None,
    'scipy': None
}

try:
    import tensorflow as tf
    dependencias['tensorflow'] = tf.__version__
    print(f"TensorFlow versión: {tf.__version__}")
    
    import matplotlib
    dependencias['matplotlib'] = matplotlib.__version__
    print(f"Matplotlib versión: {matplotlib.__version__}")
    
    from PIL import Image
    dependencias['PIL'] = Image.__version__
    print(f"Pillow versión: {Image.__version__}")
    
    import cv2
    dependencias['cv2'] = cv2.__version__
    print(f"OpenCV versión: {cv2.__version__}")
    
    import scipy
    dependencias['scipy'] = scipy.__version__
    print(f"SciPy versión: {scipy.__version__}")
    
except ImportError as e:
    print(f"\nError: Falta instalar dependencias - {str(e)}")
    print("\nEjecuta los siguientes comandos:")
    print("pip install matplotlib==3.7.1")
    print("pip install pillow==10.0.0")
    print("pip install opencv-python==4.8.0.74")
    print("pip install scipy==1.11.1")
    sys.exit(1)

# Actualizar versiones mínimas
versiones_minimas = {
    'tensorflow': '2.13.0',
    'matplotlib': '3.7.0',
    'PIL': '10.0.0',
    'cv2': '4.8.0',
    'scipy': '1.11.0'
}

for lib, version in dependencias.items():
    if version < versiones_minimas[lib]:
        print(f"\nAdvertencia: La versión de {lib} ({version}) es menor que la recomendada ({versiones_minimas[lib]})")
        print(f"Esto podría causar problemas de compatibilidad.")

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def verificar_imagenes(directorio):
    """Verifica y cuenta las imágenes válidas en el directorio."""
    imagenes_validas = 0
    imagenes_invalidas = 0
    for imagen_path in glob.glob(os.path.join(directorio, "*.jpg")):
        try:
            with Image.open(imagen_path) as img:
                img.verify()
                imagenes_validas += 1
        except:
            print(f"Imagen inválida encontrada: {imagen_path}")
            imagenes_invalidas += 1
            try:
                os.remove(imagen_path)
                print(f"Imagen inválida eliminada: {imagen_path}")
            except:
                print(f"No se pudo eliminar la imagen: {imagen_path}")
    return imagenes_validas, imagenes_invalidas

def main():
    # Configurar rutas
    directorio_base = os.path.join(os.path.dirname(__file__), "dataset")

    # Verificar si existen las carpetas necesarias
    if not os.path.exists(os.path.join(directorio_base, "cat")) or \
       not os.path.exists(os.path.join(directorio_base, "dog")):
        print("Error: No se encuentran las carpetas cat y dog en dataset/")
        exit()

    print("Verificando imágenes...")
    cat_valid, cat_invalid = verificar_imagenes(os.path.join(directorio_base, "cat"))
    dog_valid, dog_invalid = verificar_imagenes(os.path.join(directorio_base, "dog"))

    print(f"Gatos - Imágenes válidas: {cat_valid}, Imágenes inválidas: {cat_invalid}")
    print(f"Perros - Imágenes válidas: {dog_valid}, Imágenes inválidas: {dog_invalid}")

    # Configurar parámetros
    altura, ancho = 150, 150
    batch_size = 32
    num_epochs = 15

    # Preparar generador de datos con validación
    datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    print("Cargando imágenes de entrenamiento...")

    try:
        # Crear generadores
        train_generator = datagen.flow_from_directory(
            directorio_base,
            target_size=(altura, ancho),
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            directorio_base,
            target_size=(altura, ancho),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )

        print("Creando modelo...")

        # Crear el modelo usando Input
        inputs = layers.Input(shape=(altura, ancho, 3))
        x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        modelo = models.Model(inputs=inputs, outputs=outputs)

        # Compilar el modelo
        modelo.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("Iniciando entrenamiento...")

        # Entrenar el modelo
        history = modelo.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=num_epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )

        print("Guardando modelo...")

        # Guardar el modelo de manera compatible
        modelo.save('modelo_gatos_perros.h5', save_format='h5')

        print("¡Entrenamiento completado!")
        
        try:
            # Graficar resultados
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Entrenamiento')
            plt.plot(history.history['val_accuracy'], label='Validación')
            plt.title('Precisión del modelo')
            plt.xlabel('Época')
            plt.ylabel('Precisión')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Entrenamiento')
            plt.plot(history.history['val_loss'], label='Validación')
            plt.title('Pérdida del modelo')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.savefig('grafico_entrenamiento.png')
            print("Gráfico guardado en grafico_entrenamiento.png")

        except Exception as e:
            print(f"Advertencia: No se pudieron mostrar las gráficas: {str(e)}")
            print("El modelo se ha guardado correctamente de todas formas.")

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        exit(1)

    # Verificar si existe el modelo
    modelo_path = 'modelo_gatos_perros.h5'
    if not os.path.exists(modelo_path):
        print("Error: No se encuentra el modelo entrenado.")
        print("Primero ejecuta: python entrenar_modelo.py")
        sys.exit(1)

if __name__ == "__main__":
    main()