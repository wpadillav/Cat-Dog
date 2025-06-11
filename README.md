# 🐱🐶 Clasificador de Imágenes de Gatos y Perros

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)
![GPU Support](https://img.shields.io/badge/GPU-Supported-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Maintained](https://img.shields.io/badge/maintained-yes-brightgreen)

📝 Este proyecto está licenciado bajo los términos de la [Licencia MIT](LICENSE).

Este proyecto entrena una red neuronal convolucional (CNN) utilizando TensorFlow y Keras para clasificar imágenes como **gato** o **perro**. El entrenamiento aprovecha la GPU (CUDA) si está disponible y correctamente configurada.

---

## 📁 Estructura del Proyecto

```

.
├── dataset/                      # Contiene las imágenes .jpg de entrenamiento
│   ├── cat/                      # Imágenes de gatos
│   └── dog/                      # Imágenes de perros
├── entrenar\_modelo.py           # Script principal de entrenamiento
├── config.json                  # Parámetros de configuración del modelo
├── grafico\_entrenamiento.png    # Gráfico generado tras el entrenamiento
├── requirements.txt             # Dependencias necesarias
├── .gitignore                   # Archivos excluidos del repositorio
└── README.md                    # Documentación del proyecto

````

---

## 🚀 ¿Qué hace `entrenar_modelo.py`?

Este script realiza las siguientes tareas:

1. **Carga configuración desde `config.json`** para definir hiperparámetros y rutas.
2. **Verifica versiones de librerías críticas**: TensorFlow, Pillow, Matplotlib.
3. **Valida imágenes `.jpg`**: elimina automáticamente las corruptas.
4. **Carga los datos y los separa** en 80% entrenamiento y 20% validación mediante `ImageDataGenerator`.
5. **Define una CNN sencilla** para clasificación binaria (gato vs perro).
6. **Entrena el modelo usando GPU si está disponible**, con `EarlyStopping` y `ModelCheckpoint`.
7. **Guarda el mejor modelo entrenado** en formato `.keras`.
8. **Genera y guarda un gráfico** de precisión y pérdida por época.

---

## ⚙️ Configuración (`config.json`)

Ejemplo de archivo `config.json`:

```json
{
  "image_height": 150,
  "image_width": 150,
  "batch_size": 32,
  "num_epochs": 50,
  "validation_split": 0.2,
  "dataset_dir": "dataset",
  "classes": ["cat", "dog"],
  "model_output_path": "modelo_gatos_perros.keras",
  "plot_output_path": "grafico_entrenamiento.png"
}
````

Puedes ajustar cualquier parámetro sin modificar el código Python directamente.

---

## 📦 Instalación

### 1. Crear entorno virtual (opcional pero recomendado)

```bash
python3 -m venv lib_pip
source lib_pip/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧪 Dataset

Asegúrate de contar con la siguiente estructura de carpetas:

```
dataset/
├── cat/
│   ├── imagen1.jpg
│   └── ...
└── dog/
    ├── imagen1.jpg
    └── ...
```

> 📝 Todas las imágenes deben tener extensión `.jpg`. Las imágenes inválidas o corruptas serán detectadas y eliminadas automáticamente al iniciar el entrenamiento.

---

## 📋 `requirements.txt`

```txt
tensorflow[and-cuda]==2.19.0
pillow>=11.2.1
matplotlib>=3.10.3
scipy>=1.15.3
packaging>=24.0
```

> ⚠️ **`numpy` y `opencv-python` no son necesarios actualmente.**

---

## 📈 Resultados Esperados

* Precisión de validación estimada: **85% a 87%**
* Modelo entrenado: `modelo_gatos_perros.keras`
* Gráfico guardado: `grafico_entrenamiento.png`

---

## 🛑 `.gitignore` recomendado

Ejemplo de contenido:

```
*.keras
*.h5
.env
__pycache__/
*.pyc
dataset/
grafico_entrenamiento.png
```

---

## 📌 Repositorio

Repositorio en GitHub:
🔗 [https://github.com/wpadillav/Cat-Dog](https://github.com/wpadillav/Cat-Dog)

