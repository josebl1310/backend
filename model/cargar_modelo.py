import torch
from torchvision import models
import torch.nn as nn

# Cargar el modelo preentrenado
model = models.resnet18()

# Modificar la capa final para 4 clases (como en tu API)
model.fc = nn.Linear(model.fc.in_features, 4)

# Imprimir la estructura del modelo
print(model)

# Opcional: Puedes intentar cargar los pesos si quieres ver si la estructura coincide
# try:
#     model.load_state_dict(torch.load(
#         r'D:\Documentos\Documentos - copia\Uni\TFG\backend\model\mejor_modelo.pth',
#         map_location=torch.device('cpu')
#     ))
#     print("\nPesos cargados correctamente.")
# except FileNotFoundError:
#     print("\nArchivo de pesos no encontrado.")
# except Exception as e:
#     print(f"\nError al cargar los pesos: {e}")