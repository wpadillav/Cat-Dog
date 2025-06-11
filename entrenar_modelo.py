import os
import sys
import glob
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Permite generar gráficos sin entorno gráfico (ideal para servidores)
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar configuración desde config.json
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except Exception as e:
    print(f"Error al cargar config.json: {e}")
    sys.exit(1)

# Validar claves esperadas
claves_esperadas = [
    "image_height", "image_width", "batch_size", "num_epochs",
    "validation_split", "dataset_dir", "classes",
    "model_output_path", "plot_output_path"
]
faltantes = [clave for clave in claves_esperadas if clave not in config]
if faltantes:
    print(f"Error: Faltan claves en config.json: {faltantes}")
    sys.exit(1)

# Extraer parámetros del JSON
altura = config["image_height"]
ancho = config["image_width"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
validation_split = config["validation_split"]
dataset_dir = config["dataset_dir"]
clases = config["classes"]
model_output_path = config["model_output_path"]
plot_output_path = config["plot_output_path"]

# Mostrar configuración cargada
print("Configuración cargada:")
for clave, valor in config.items():
    print(f"  {clave}: {valor}")

# Verificación de dependencias clave
try:
    import tensorflow as tf
    import PIL
    import matplotlib

    print(f"TensorFlow versión: {tf.__version__}")
    print(f"Pillow versión: {PIL.__version__}")
    print(f"Matplotlib versión: {matplotlib.__version__}")
except ImportError as e:
    print(f"\nError: Falta instalar una dependencia: {str(e)}")
    sys.exit(1)

def verificar_imagenes(directorio):
    """
    Verifica y elimina imágenes .jpg corruptas del directorio.
    """
    imagenes_validas = 0
    imagenes_invalidas = 0

    for imagen_path in glob.glob(os.path.join(directorio, "*.jpg")):
        try:
            with Image.open(imagen_path) as img:
                img.verify()
            imagenes_validas += 1
        except Exception as e:
            print(f"Imagen inválida: {imagen_path} ({e})")
            imagenes_invalidas += 1
            try:
                os.remove(imagen_path)
                print(f"Imagen eliminada: {imagen_path}")
            except Exception as e:
                print(f"No se pudo eliminar la imagen: {imagen_path} ({e})")

    return imagenes_validas, imagenes_invalidas

def main():
    # Obtener ruta absoluta al dataset
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    directorio_base = os.path.join(script_dir, dataset_dir)

    # Verificar estructura esperada del dataset
    if not os.path.exists(os.path.join(directorio_base, "cat")) or \
       not os.path.exists(os.path.join(directorio_base, "dog")):
        print("Error: No se encuentran las carpetas cat y dog en el dataset.")
        sys.exit(1)

    print("Verificando imágenes...")
    cat_valid, cat_invalid = verificar_imagenes(os.path.join(directorio_base, "cat"))
    dog_valid, dog_invalid = verificar_imagenes(os.path.join(directorio_base, "dog"))

    print(f"Gatos - Válidas: {cat_valid}, Inválidas: {cat_invalid}")
    print(f"Perros - Válidas: {dog_valid}, Inválidas: {dog_invalid}")

    # Preparación de los generadores de datos
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=validation_split,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        directorio_base,
        target_size=(altura, ancho),
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        classes=clases
    )

    validation_generator = datagen.flow_from_directory(
        directorio_base,
        target_size=(altura, ancho),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        classes=clases
    )

    print("Creando modelo...")

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

    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Configurando callbacks...")

    callback_early = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    callback_checkpoint = callbacks.ModelCheckpoint(
        filepath=model_output_path,
        monitor='val_loss',
        save_best_only=True
    )

    print("Entrenando modelo...")

    history = modelo.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[callback_early, callback_checkpoint]
    )

    print("Entrenamiento finalizado.")

    try:
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

        plt.savefig(plot_output_path)
        print(f"Gráfico guardado en {plot_output_path}")

    except Exception as e:
        print(f"Advertencia: No se pudieron generar las gráficas: {str(e)}")

if __name__ == "__main__":
    main()
