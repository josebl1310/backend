# API de Clasificación de Imágenes con FastAPI y PyTorch

Este proyecto expone un modelo de deep learning (ResNet) a través de una API REST utilizando FastAPI. Permite enviar imágenes, procesarlas y obtener predicciones, incluyendo visualizaciones como Grad-CAM si están activadas.

---

## 🧰 Requisitos

- Python 3.8+
- pip
- (Opcional) [Conda](https://docs.conda.io/en/latest/)

---

## 📦 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/josebl1310/backend.git
cd backend

Crear entorno virtual:

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

Opcional en conda:
conda create -n tu_entorno python=3.9
conda activate entorno

Instalar dependencias:
pip install -r requirements.txt

Levantar la API:
uvicorn main:app --reload

El servidor se iniciará en http://127.0.0.1:8000

