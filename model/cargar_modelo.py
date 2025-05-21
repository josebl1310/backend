# test_model.py
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import os # Importar os para construir la ruta al modelo

# --- 1. CONFIGURACIÓN DEL MODELO Y CARGA DE PESOS ---
model = models.resnet18()
# Asegúrate de que esta capa final sea IDÉNTICA a como la definiste en el entrenamiento
model.fc = nn.Linear(model.fc.in_features, 4) # 4 clases (0, 1, 2, 3)

# Ruta a tu archivo de pesos del modelo
# !!! ASEGÚRATE DE QUE ESTA RUTA ES CORRECTA !!!
model_path = r'D:\Documentos\Documentos - copia\Uni\TFG\backend\model\mejor_modelo.pth'

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"Pesos cargados correctamente desde: {model_path}")
except FileNotFoundError:
    print(f"ERROR: Archivo de pesos no encontrado en {model_path}. Por favor, verifica la ruta.")
    exit() # Salir si no se encuentra el modelo
except Exception as e:
    print(f"ERROR al cargar pesos: {e}")
    exit() # Salir si hay otro error de carga

model.eval() # Poner el modelo en modo evaluación (crucial para la inferencia)

# --- 2. DEFINICIÓN DE LAS TRANSFORMACIONES DEL MODELO ---
# !!! ESTAS TRANSFORMACIONES DEBEN SER IDÉNTICAS A LAS USADAS DURANTE EL ENTRENAMIENTO !!!
# ESPECIALMENTE RESIZE Y NORMALIZE (medias y std).
model_transform = transforms.Compose([
    transforms.Resize((224, 224)), # Tamaño esperado por ResNet18
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Medias de ImageNet
                         std=[0.229, 0.224, 0.225])   # Stds de ImageNet
])

# --- 3. CARGAR Y PROCESAR UNA IMAGEN DE PRUEBA (CLASE 'pituitary') ---
# !!! CAMBIA ESTA RUTA A UNA IMAGEN REAL DE TU CLASE 'pituitary' !!!
# Elige una imagen de la que estés seguro que es 'pituitary'.
image_to_test_path = r'D:\Documentos\Documentos - copia\Uni\TFG\datasetNUEVO\Testing\pituitary\Te-pi_0035.jpg' # Ejemplo, ¡AJUSTA ESTO!

# Puedes añadir un mapeo de clases para imprimir los nombres
class_names = {
    0: 'glioma',
    1: 'meningioma',
    2: 'notumor',
    3: 'pituitary'
}

print(f"\n--- Evaluando imagen: {image_to_test_path} ---")

try:
    # Cargar la imagen
    image = Image.open(image_to_test_path).convert('RGB')
    
    # Preprocesar la imagen para el modelo
    input_tensor = model_transform(image).unsqueeze(0) # Añadir dimensión de batch

    # Realizar la inferencia
    with torch.no_grad(): # Desactivar el cálculo de gradientes para la inferencia (más rápido y menos memoria)
        output_logits = model(input_tensor)
        output_probs = torch.softmax(output_logits, dim=1) # Convertir logits a probabilidades
        pred_class_idx = output_probs.argmax(dim=1).item() # Obtener el índice de la clase predicha

    # Imprimir resultados
    print(f"Probabilidades por clase (orden 0,1,2,3): {output_probs.squeeze().tolist()}")
    print(f"Índice de Clase Predicha: {pred_class_idx}")
    print(f"Nombre de Clase Predicha: {class_names.get(pred_class_idx, 'Desconocida')}")
    print(f"La clase real de la imagen es: {class_names.get(3, 'pituitary')}") # Confirmamos que esperamos la clase 3

except FileNotFoundError:
    print(f"ERROR: Imagen de prueba no encontrada en {image_to_test_path}. Verifica la ruta.")
except Exception as e:
    print(f"ERROR al procesar la imagen: {e}")