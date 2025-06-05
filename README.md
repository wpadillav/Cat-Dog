# Clasificador de Imágenes de Gatos y Perros 🐱🐶

Este proyecto entrena un modelo de redes neuronales convolucionales (CNN) utilizando TensorFlow y Keras para clasificar imágenes entre **gatos** y **perros**. El modelo puede ser entrenado con una estructura básica de carpetas y luego utilizado para predicción.

---

## 📁 Estructura del Proyecto

```

.
├── dataset/
│   ├── cat/                        # Imágenes de gatos
│   └── dog/                        # Imágenes de perros
├── entrenar\_modelo.py             # Script principal para entrenamiento del modelo
├── grafico\_entrenamiento.png      # Resultados visuales del entrenamiento (precisión y pérdida)
├── README.md                       # Documentación del proyecto
└── requirements.txt                # Dependencias del entorno

````

---

## 🚀 ¿Qué hace `entrenar_modelo.py`?

Este script realiza las siguientes tareas:

1. **Verificación de dependencias**  
   Comprueba la versión de las principales bibliotecas utilizadas y muestra advertencias si no coinciden con las versiones recomendadas.

2. **Validación de imágenes**  
   Recorre el directorio `dataset/cat` y `dataset/dog` para verificar imágenes corruptas (por ejemplo, truncadas) y las elimina automáticamente.

3. **Carga y preprocesamiento de datos**  
   Usa `ImageDataGenerator` para aplicar aumentos de datos y dividir automáticamente los datos en entrenamiento (80%) y validación (20%).

4. **Definición del modelo CNN**  
   Crea un modelo secuencial con capas convolucionales, de pooling y completamente conectadas para la clasificación binaria.

5. **Entrenamiento**  
   Entrena el modelo durante 15 épocas, registrando precisión y pérdida en cada época.

6. **Guardado del modelo**  
   Guarda el modelo entrenado en formato `.h5` (`modelo_gatos_perros.h5`) y genera un gráfico visual (`grafico_entrenamiento.png`).

---

## 🛠️ Instalación

### 1. Crear entorno virtual (opcional pero recomendado)

```bash
python3 -m venv lib_pip
source lib_pip/bin/activate
````

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧪 Requisitos de las librerías

Contenido del archivo `requirements.txt`:

```
numpy                  # Manipulación eficiente de arreglos y datos numéricos
tensorflow[and-cuda]   # Framework de deep learning con soporte para GPU (cuDNN/CUDA)
pillow                 # Carga y validación de imágenes (usado por PIL)
opencv-python          # Procesamiento de imágenes (opcional, para visualización avanzada)
matplotlib             # Visualización de resultados (precisión y pérdida)
scipy                  # Funciones científicas adicionales utilizadas por algunas bibliotecas
```

---

## 📈 Resultados del modelo

* **Precisión en entrenamiento**: \~85.8%
* **Precisión en validación**: \~86.4%
* **Gráfico**: Se guarda automáticamente como `grafico_entrenamiento.png`.

---

## 🔮 Futuros cambios

* Reemplazar el guardado en `.h5` por el formato moderno de Keras (`.keras`).
* Actualizar compatibilidad con Keras 3 y eliminar argumentos como `save_format`.
* Incorporar `EarlyStopping` y `ModelCheckpoint` para mejorar el control del entrenamiento.

---

## 📌 Requisitos del dataset

* Las carpetas `dataset/cat/` y `dataset/dog/` deben contener imágenes `.jpg` bien formateadas.
* No se recomienda el uso de imágenes TIFF ni RAW.

---
