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
│   ├── cat/                     # Imágenes de gatos
│   └── dog/                     # Imágenes de perros
├── entrenar\_modelo.py          # Script principal de entrenamiento
├── grafico\_entrenamiento.png   # Gráfico con precisión y pérdida
├── requirements.txt            # Dependencias necesarias
├── .gitignore                  # Archivos excluidos del repositorio
└── README.md                   # Documentación del proyecto

````

---

## 🚀 ¿Qué hace `entrenar_modelo.py`?

Este script realiza las siguientes tareas:

1. **Verifica versiones de librerías críticas**: TensorFlow, Pillow, Matplotlib.
2. **Valida imágenes** `.jpg`: elimina automáticamente las corruptas.
3. **Carga los datos**: separa 80% para entrenamiento y 20% para validación usando `ImageDataGenerator`.
4. **Define una CNN sencilla** para clasificación binaria.
5. **Entrena el modelo con soporte GPU** si está disponible.
6. **Aplica EarlyStopping y guarda el mejor modelo** en formato `.keras`.
7. **Genera un gráfico** de precisión y pérdida por época.

---

## 📦 Instalación

### 1. Crear entorno virtual (opcional)

```bash
python3 -m venv lib_pip
source lib_pip/bin/activate
````

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🧪 Dataset

Se requiere la siguiente estructura:

```
dataset/
├── cat/
│   ├── imagen1.jpg
│   └── ...
└── dog/
    ├── imagen1.jpg
    └── ...
```

* Todas las imágenes deben tener extensión `.jpg`.
* Se eliminarán automáticamente las imágenes corruptas o inválidas.

---

## 📋 requirements.txt

```
tensorflow[and-cuda]==2.19.0
pillow>=11.2.1
matplotlib>=3.10.3
scipy>=1.15.3
packaging>=24.0
```

> ⚠️ **`numpy` y `opencv-python` no se requieren actualmente.**

---

## 📈 Resultados esperados

* Precisión de validación esperada: entre **85% y 87%**
* Modelo entrenado: `modelo_gatos_perros.keras`
* Gráfico guardado: `grafico_entrenamiento.png`

---

## 🛑 Ignorados por Git

Archivo `.gitignore`:

```
modelo_gatos_perros.h5
modelo_gatos_perros.keras
.env
dataset/
```

---

## 📌 Repositorio

Repositorio en GitHub:
🔗 [https://github.com/wpadillav/Cat-Dog](https://github.com/wpadillav/Cat-Dog)

---