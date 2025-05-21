from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageEnhance
import numpy as np
import io
import random
import torch
from torchvision import transforms, models
import torch.nn as nn
from scipy.ndimage import zoom
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import ast

# --- Cargar modelo preentrenado y ajustar capa final ---
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases

try:
    model.load_state_dict(torch.load(
        r'D:\Documentos\Documentos - copia\Uni\TFG\backend\model\mejor_modelo.pth',
        map_location=torch.device('cpu')
    ))
    print("Pesos cargados correctamente en la API.")
except FileNotFoundError:
    print("Archivo de pesos no encontrado para la API. Asegúrate de que 'mejor_modelo.pth' esté en la ruta correcta.")
except Exception as e:
    print(f"Error al cargar pesos en la API: {e}")

model.eval() # Asegúrate de que el modelo esté en modo evaluación

# --- Crear app FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse("frontend/index.html")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# --- Modelo para parámetros de imagen ---
class ImageRequest(BaseModel):
    grayscale: bool
    method: str
    resize: bool
    target_size: tuple
    keep_aspect_ratio: bool
    interpolation: str
    normalize: bool
    normalize_method: str
    range_minmax: tuple
    flip: str
    rotation: int
    brightness_range: tuple
    zoom_range: tuple
    noise: str
    apply_prob: float

# --- Funciones auxiliares (se mantienen de la versión anterior) ---
def apply_grayscale(image: Image.Image, grayscale: bool, method: str) -> Image.Image:
    if not grayscale:
        return image
    if method == 'average':
        return image.convert('L').convert('RGB')
    elif method == 'luminosity':
        np_img = np.array(image)
        lum = np_img[..., 0] * 0.21 + np_img[..., 1] * 0.72 + np_img[..., 2] * 0.07
        lum_img = np.stack([lum, lum, lum], axis=-1).astype(np.uint8)
        return Image.fromarray(lum_img)
    elif method == 'desaturation':
        np_img = np.array(image)
        max_val = np.max(np_img, axis=2)
        min_val = np.min(np_img, axis=2)
        desat = ((max_val + min_val) / 2).astype(np.uint8)
        desat_img = np.stack([desat, desat, desat], axis=-1)
        return Image.fromarray(desat_img)
    return image

def apply_resize_augmentation(image: Image.Image, resize: bool, target_size: tuple, keep_aspect_ratio: bool,
                            interpolation: str) -> Image.Image:
    if not resize:
        return image
    interp_methods = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    interp = interp_methods.get(interpolation, Image.BILINEAR)
    if keep_aspect_ratio:
        image.thumbnail(target_size, interp)
        new_img = Image.new("RGB", target_size)
        paste_pos = ((target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2)
        new_img.paste(image, paste_pos)
        return new_img
    else:
        return image.resize(target_size, interp)

def apply_normalize_augmentation(image: Image.Image, normalize: bool, normalize_method: str, range_minmax: tuple) -> Image.Image:
    if not normalize:
        return image
    np_img = np.array(image).astype(np.float32)
    if normalize_method == 'minmax':
        min_val, max_val = range_minmax
        if max_val - min_val == 0:
            np_img = np.zeros_like(np_img)
        else:
            np_img = (np_img - min_val) / (max_val - min_val)
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
    elif normalize_method == 'zscore':
        mean = np.mean(np_img, axis=(0, 1), keepdims=True)
        std = np.std(np_img, axis=(0, 1), keepdims=True)
        np_img = (np_img - mean) / (std + 1e-7)
        np_img = ((np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-7) * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def augment_image(image: Image.Image, flip: str, rotation: int,
                  brightness_range: tuple, zoom_range: tuple,
                  noise: str, apply_prob: float) -> Image.Image:
    if flip != 'none' and random.random() < apply_prob:
        if flip == 'horizontal':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip == 'vertical':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip == 'both':
            image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    if rotation != 0 and random.random() < apply_prob:
        image = image.rotate(rotation)

    if brightness_range != (1.0, 1.0) and random.random() < apply_prob:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(brightness_range[0], brightness_range[1])
        image = enhancer.enhance(factor)

    if zoom_range != (1.0, 1.0) and random.random() < apply_prob:
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        w, h = image.size
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        new_w = max(1, new_w)
        new_h = max(1, new_h)

        image = image.resize((new_w, new_h), Image.LANCZOS)
        if zoom_factor > 1:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            left = max(0, min(left, new_w - w))
            top = max(0, min(top, new_h - h))
            image = image.crop((left, top, left + w, top + h))
        else:
            new_image = Image.new("RGB", (w, h), (0, 0, 0))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            new_image.paste(image, (left, top))
            image = new_image

    if noise != 'none' and random.random() < apply_prob:
        np_img = np.array(image).astype(np.float32)
        if noise == 'gaussian':
            mean = 0
            var = 0.01
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, np_img.shape) * 255
            noisy = np_img + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy)
        elif noise == 'salt_and_pepper':
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(np_img)
            num_salt = np.ceil(amount * np_img.size * s_vs_p).astype(int)
            coords = [np.random.randint(0, dim - 1, num_salt) for dim in np_img.shape]
            out[tuple(coords)] = 255
            num_pepper = np.ceil(amount * np_img.size * (1. - s_vs_p)).astype(int)
            coords = [np.random.randint(0, dim - 1, num_pepper) for dim in np_img.shape]
            out[tuple(coords)] = 0
            image = Image.fromarray(out.astype(np.uint8))
    return image

def transform_image_for_model_and_augment(image: Image.Image, params: ImageRequest):
    # Crear una copia de la imagen para aplicar aumentos sin modificar la original
    model_input_image = image.copy() 

    # Aplicar los aumentos de datos seleccionados por el usuario
    model_input_image = apply_grayscale(model_input_image, params.grayscale, params.method)
    model_input_image = apply_resize_augmentation(model_input_image, params.resize, params.target_size, params.keep_aspect_ratio, params.interpolation)
    model_input_image = apply_normalize_augmentation(model_input_image, params.normalize, params.normalize_method, params.range_minmax)
    model_input_image = augment_image(model_input_image, params.flip, params.rotation, params.brightness_range,
                                      params.zoom_range, params.noise, params.apply_prob)

    # Definir las transformaciones ESTÁNDAR que el modelo SIEMPRE espera (redimensionamiento y normalización de ImageNet)
    model_standard_transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet18 espera 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Normalización ImageNet
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Aplicar las transformaciones estándar
    image_tensor = model_standard_transform(model_input_image).unsqueeze(0)  # Añadir batch dimension
    return image_tensor

# --- Clase para manejar Grad-CAM manualmente ---
class ManualGradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None

        # Registrar hooks para la capa objetivo
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                module.register_forward_hook(self._save_activation)
                module.register_backward_hook(self._save_gradient)
                break
        else:
            raise ValueError(f"Capa '{target_layer_name}' no encontrada en el modelo.")

    def _save_activation(self, module, input, output):
        self.activations = output
    
    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output es una tupla (grad_wrt_output, grad_wrt_weights, ...)
        # Queremos los gradientes de la salida de la capa, que es el primer elemento.
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_class_idx):
        self.model.zero_grad() # Limpiar gradientes anteriores
        
        # Realizar pasada hacia adelante
        output_logits = self.model(input_tensor)
        
        # Calcular gradientes para la clase objetivo
        # Asegurarse de que el target_class_idx sea válido
        if target_class_idx is None:
            # Si no se especifica, usa la clase con el logit más alto
            target_class_idx = output_logits.argmax(dim=1).item()
        
        # Seleccionar el logit de la clase objetivo y aplicar backward
        # output_logits[0, target_class_idx] es el logit para la clase específica en el lote 0
        output_logits[0, target_class_idx].backward(retain_graph=True) 

        # Calcular los pesos de las características (alpha_k)
        # Global Average Pooling de los gradientes
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3]) # Promedio sobre altura y ancho

        # Multiplicar las activaciones por los pesos de los gradientes
        # Asegurarse de que las dimensiones coincidan para la multiplicación
        # pooled_gradients: [batch_size, num_features]
        # self.activations: [batch_size, num_features, H, W]
        for i in range(len(pooled_gradients)): # Iterar sobre el batch (en nuestro caso, solo 1)
            # Expandir pooled_gradients para que pueda multiplicarse por self.activations espacialmente
            # [num_features] -> [num_features, 1, 1]
            if i == 0: # Para el primer elemento del lote (y único en nuestro caso)
                weighted_activations = self.activations[i] * pooled_gradients[i].unsqueeze(-1).unsqueeze(-1)
            else:
                # Si tuvieras un lote real, aquí iría la lógica para los elementos subsiguientes
                pass 

        # Sumar los mapas de características ponderados y aplicar ReLU
        # Sumar sobre la dimensión de características para obtener el mapa de calor final
        cam = torch.sum(weighted_activations, dim=0) # Sumar a través de las características
        cam = torch.relu(cam) # Aplicar ReLU

        return cam

# --- Inicializar la implementación manual de Grad-CAM ---
# Necesitas pasar el modelo y el nombre de la capa objetivo.
# Para ResNet18, 'layer4' es la última capa convolucional antes del Global Average Pooling.
grad_cam_manual_extractor = ManualGradCAM(model, target_layer_name='layer4')

# --- Endpoint principal ---
@app.post("/procesar")
async def procesar_imagen(
        file: UploadFile = File(...),
        grayscale: bool = Form(False),
        method: str = Form("average"),
        resize: bool = Form(False),
        target_size: str = Form("(224,224)"),
        keep_aspect_ratio: bool = Form(True),
        interpolation: str = Form("bilinear"),
        normalize: bool = Form(False),
        normalize_method: str = Form("minmax"),
        range_minmax: str = Form("(0,255)"),
        flip: str = Form("none"),
        rotation: int = Form(0),
        brightness_range: str = Form("(1.0,1.0)"),
        zoom_range: str = Form("(1.0,1.0)"),
        noise: str = Form("none"),
        apply_prob: float = Form(0.0)
):
    # Convertir strings a tipos correctos
    target_size = ast.literal_eval(target_size)
    range_minmax = ast.literal_eval(range_minmax)
    brightness_range = ast.literal_eval(brightness_range)
    zoom_range = ast.literal_eval(zoom_range)

    # Cargar imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB') # La imagen original para la superposición

    # Construir objeto params
    params = ImageRequest(
        grayscale=grayscale,
        method=method,
        resize=resize,
        target_size=target_size,
        keep_aspect_ratio=keep_aspect_ratio,
        interpolation=interpolation,
        normalize=normalize,
        normalize_method=normalize_method,
        range_minmax=range_minmax,
        flip=flip,
        rotation=rotation,
        brightness_range=brightness_range,
        zoom_range=zoom_range,
        noise=noise,
        apply_prob=apply_prob
    )
    
    # Preparar el tensor de entrada para el modelo, aplicando los aumentos si se solicitan
    input_tensor = transform_image_for_model_and_augment(image.copy(), params) # Pasamos una copia
    
    # La inferencia del modelo no necesita que input_tensor.requires_grad sea True
    # hasta que hagamos el backward, pero es bueno que lo tengamos listo.
    input_tensor.requires_grad_(True) 
    
    # Realizar la inferencia
    model.eval() # Aseguramos que el modelo está en modo evaluación antes de la inferencia
    output_logits = model(input_tensor) 
    
    # Calcula las probabilidades y la clase predicha
    output_probs = torch.softmax(output_logits, dim=1)
    pred_class = output_probs.argmax(dim=1).item() # La clase predicha por el modelo

    # --- Generar Grad-CAM usando la implementación manual ---
    # Pasamos el input_tensor (ya que los hooks necesitan que haya habido una pasada forward)
    # y la clase predicha.
    activation_map = grad_cam_manual_extractor(input_tensor, pred_class)
    
    # Convertir a numpy para procesamiento de imagen
    activation_map = activation_map.squeeze().detach().cpu().numpy()

    # Redimensionar mapa para la imagen original
    # Usamos las dimensiones de la imagen original (no la 224x224 del tensor de entrada)
    map_resized = zoom(activation_map, (image.height / activation_map.shape[0], image.width / activation_map.shape[1]),
                       order=1)
    map_resized = (map_resized - map_resized.min()) / (map_resized.max() - map_resized.min() + 1e-8)  # Normalizar a 0-1

    # Crear imagen de heatmap (colores rojos)
    heatmap = np.zeros((map_resized.shape[0], map_resized.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = (map_resized * 255).astype(np.uint8)  # rojo
    heatmap[..., 1] = 0
    heatmap[..., 2] = 0

    heatmap_img = Image.fromarray(heatmap)

    # Superponer heatmap con imagen original
    heatmap_img = heatmap_img.resize(image.size)
    overlay = Image.blend(image, heatmap_img, alpha=0.5)

    # Guardar resultado temporal
    output_path = "frontend/grad_cam_result.jpg"
    overlay.save(output_path)

    return {
        "imagen_url": "/frontend/grad_cam_result.jpg",
        "prediccion": pred_class
    }