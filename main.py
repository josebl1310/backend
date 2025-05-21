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
from torchcam.methods import GradCAM
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import ast

# --- Cargar modelo preentrenado y ajustar capa final ---
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 clases
model.load_state_dict(torch.load(
    r'D:\Documentos\Documentos - copia\Uni\TFG\backend\model\mejor_modelo.pth',
    map_location=torch.device('cpu')
))
model.eval()

# --- Inicializar Grad-CAM aquí, después de cargar el modelo ---
cam_extractor = GradCAM(model, target_layer='layer4')  # Usamos layer4 como en el notebook

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

# --- Funciones auxiliares ---

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

def apply_resize(image: Image.Image, resize: bool, target_size: tuple, keep_aspect_ratio: bool,
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

def apply_normalize(image: Image.Image, normalize: bool, normalize_method: str, range_minmax: tuple) -> Image.Image:
    if not normalize:
        return image
    np_img = np.array(image).astype(np.float32)
    if normalize_method == 'minmax':
        min_val, max_val = range_minmax
        np_img = (np_img - min_val) / (max_val - min_val)
        np_img = np.clip(np_img, 0, 1)
        np_img = (np_img * 255).astype(np.uint8)
    elif normalize_method == 'zscore':
        mean = np.mean(np_img, axis=(0, 1), keepdims=True)
        std = np.std(np_img, axis=(0, 1), keepdims=True)
        np_img = (np_img - mean) / (std + 1e-7)
        np_img = ((np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def augment_image(image: Image.Image, flip: str, rotation: int,
                  brightness_range: tuple, zoom_range: tuple,
                  noise: str, apply_prob: float) -> Image.Image:
    # Flip
    if flip != 'none' and random.random() < apply_prob:
        if flip == 'horizontal':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip == 'vertical':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip == 'both':
            image = image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    # Rotation
    if rotation != 0 and random.random() < apply_prob:
        image = image.rotate(rotation)

    # Brightness
    if brightness_range != (1.0, 1.0) and random.random() < apply_prob:
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(brightness_range[0], brightness_range[1])
        image = enhancer.enhance(factor)

    # Zoom
    if zoom_range != (1.0, 1.0) and random.random() < apply_prob:
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        w, h = image.size
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        if zoom_factor > 1:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            image = image.crop((left, top, left + w, top + h))
        else:
            new_image = Image.new("RGB", (w, h))
            left = (w - new_w) // 2
            top = (h - new_h) // 2
            new_image.paste(image, (left, top))
            image = new_image

    # Noise
    if noise != 'none' and random.random() < apply_prob:
        np_img = np.array(image)
        if noise == 'gaussian':
            mean = 0
            var = 0.01
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, np_img.shape)
            noisy = np_img + gauss * 255
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
            image = Image.fromarray(noisy)
        elif noise == 'salt_and_pepper':
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(np_img)
            # Salt
            num_salt = np.ceil(amount * np_img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in np_img.shape]
            out[tuple(coords)] = 255
            # Pepper
            num_pepper = np.ceil(amount * np_img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in np_img.shape]
            out[tuple(coords)] = 0
            image = Image.fromarray(out)
    return image

def transform_image(image: Image.Image, params: ImageRequest):
    # Aplicar preprocesos y aumentos
    image = apply_grayscale(image, params.grayscale, params.method)
    image = apply_resize(image, params.resize, params.target_size, params.keep_aspect_ratio, params.interpolation)
    image = apply_normalize(image, params.normalize, params.normalize_method, params.range_minmax)
    image = augment_image(image, params.flip, params.rotation, params.brightness_range,
                          params.zoom_range, params.noise, params.apply_prob)
    # Transformar a tensor normalizado para modelo
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)  # Añadir batch dimension
    return image_tensor

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
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

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

    # Procesar imagen y transformar a tensor
    input_tensor = transform_image(image, params)
    input_tensor.requires_grad_(True)

    # Poner el modelo en modo de entrenamiento temporalmente para los gradientes
    model.train()

    # Obtener la salida del modelo
    output = model(input_tensor)
    output.requires_grad_(True)  # Aseguramos que la salida también requiera gradientes
    pred_class = output.argmax(dim=1).item()

    # Obtener Grad-CAM
    activation_map = cam_extractor(pred_class, output)[0].squeeze().detach().cpu().numpy()

    # Volver a poner el modelo en modo de evaluación
    model.eval()

    # Redimensionar mapa para la imagen original
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




