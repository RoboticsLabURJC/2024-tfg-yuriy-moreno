import sys
import carla
import numpy as np
import json
import open3d as o3d
import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import signal
import pygame
import os
import cv2

from scipy.spatial import cKDTree

####### Control PID ########
# Imports seguir linea
import argparse
import cv2
# --- Variables PID globales ---
previous_error = 0.0
integral = 0.0
rgb_image = None
visual_error = 0
segmentation_model = None
####### Control PID ########

# Workaround para pandas en Python 3.8
if sys.version_info < (3, 9):
    try:
        import backports.zoneinfo
        sys.modules['zoneinfo'] = backports.zoneinfo
    except ImportError:
        print("Falta backports.zoneinfo. Instálalo con pip.")
        raise

VIRIDIS = np.array(plt.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Variables globales
actor_list = []
manual_mode = False

# Mapa de colores para etiquetas semánticas
SEMANTIC_COLOR_MAP = {
    -1: (255, 255, 255),  # puntos cósmicos(blanco)
    0: (0, 0, 0),         # Ningún objeto (negro)
    1: (128, 64, 128),    # Carretera
    2: (244, 35, 232),    # Acera
    3: (70, 70, 70),      # Edificios
    4: (102, 102, 156),  # Paredes
    5: (190, 153, 153),   # Cercas
    6: (153, 153, 153),   # Poste
    7: (250, 170, 30),    # Semáforos
    8: (220, 220, 0),    # Señal de tráfico
    9: (107, 142, 35),    # Vegetación (verde)
    10: (152, 251, 152),      # Terreno 
    11: (70, 130, 180),  # Cielo
    12: (220, 20, 60),     # Peatón (rojo)
    13: (255, 0, 0),     # Rider (rojo)
    14: (0, 0, 142),      # Autos (azul)
    15: (0, 0, 70),      # Camion (azul)
    16: (0, 60, 100),      # Bus 
    17: (0, 80, 100),      # Tren
    18: (0, 0, 230),      # Motocicleta
    19: (119, 11, 32),    # Bicicletas
    20: (110, 190, 160),    # Static
    21: (170, 120, 50),    # Dynamic
    22: (55, 90, 80),       # Otro
    23: (45, 60, 150),      # Agua
    24: (157, 234, 50),     # Linea de carretera
    25: (81, 0, 81),        # Asfalto
    26: (150, 100, 100),    # Puente
    27: (230, 150, 140),    # Rail Track
    28: (180, 165, 180),    # Guard Rail
    29: (138, 149, 151),    # Rocks
}


# Obtención del color según la etiqueta semántica
def get_color_from_semantic(semantic_tag):
    return SEMANTIC_COLOR_MAP.get(int(semantic_tag), (255, 255, 255))  # Color blanco si no está en la lista

def add_cosmic_noise_points(points, semantic_tags, max_range, 
                            hfov, upper_fov, lower_fov,
                            rate=0.001):
    """
    Inserta puntos falsos ('ruido cósmico') en el mismo marco que los datos del LiDAR.
    Como los puntos del LiDAR ya están en coordenadas globales, 
    aquí generamos directamente en mundo sin transformaciones extra.
    """
    N = points.shape[0]
    n_fake = int(N * rate)
    if n_fake == 0:
        return points, semantic_tags

    # Generar coordenadas XYZ aleatorias en el rango del LiDAR
    az = np.radians(np.random.uniform(-hfov/2, hfov/2, n_fake))
    el = np.radians(np.random.uniform(lower_fov, upper_fov, n_fake))
    r  = np.random.uniform(0.1, max_range, n_fake)

    x = -r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    fake_points_world = np.stack([x, y, z], axis=1)

    # Etiquetas de ruido cósmico (-1)
    fake_tags = np.full(n_fake, -1, dtype=np.int32)

    points_new = np.vstack([points, fake_points_world])
    semantic_tags_new = np.concatenate([semantic_tags, fake_tags])

    return points_new, semantic_tags_new


def calculate_ring_id(points, channels, lower_fov, upper_fov):
    """
    Asigna a cada punto un índice de anillo (haz) según su elevación.
    - points: (N,3) en el mismo sistema que ya usas (con tu flip X aplicado)
    - channels: nº de haces del LiDAR
    - lower_fov_deg, upper_fov_deg: límites verticales del sensor (grados)
    """
    # elevación en grados
    xy = np.hypot(points[:, 0], points[:, 1]) # sqrt(x^2 + y^2), distancia al eje Z
    elevation = np.degrees(np.arctan2(points[:, 2], xy))  # elevación en grados

    # normaliza al rango [0, 1] usando el FOV vertical real
    vertical_fov = (upper_fov - lower_fov)
    # evitar divisiones raras si el FOV es 0
    vertical_fov = vertical_fov if vertical_fov != 0 else 1e-6
    t = (elevation - lower_fov) / vertical_fov

    # cuantiza a [0, channels-1]
    ring = np.round(t * (channels - 1)).astype(np.int32)
    ring = np.clip(ring, 0, channels - 1)
    return ring

def subsample_by_ring_id(points, semantic_tags, ring_id, step=2):
    """
    Conserva canales completos: mantiene los puntos cuyo ring_id % step == 0.
    """
    mask = (ring_id % step) == 0
    return points[mask], semantic_tags[mask], mask  # devolvemos mask por si quieres aplicarla a intensities, etc.

def subsample_by_rays(points, semantic_tags, ring_id, step_ray=2):
    """
    Reduce la resolución horizontal: conserva 1 de cada `step_ray` puntos
    dentro de cada haz (ring) ordenando por azimut.
    Devuelve (points_sub, tags_sub, mask_sub).
    """
    N = len(points)
    if N == 0:
        mask = np.zeros(0, dtype=bool)
        return points, semantic_tags, mask

    # Azimutal en grados
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    az = (az + 360.0) % 360.0   # [0, 360)

    mask = np.zeros(N, dtype=bool)
    unique_rings = np.unique(ring_id)

    for r in unique_rings:
        idx = np.where(ring_id == r)[0]
        if idx.size == 0:
            continue
        order = np.argsort(az[idx])          # Ordena los puntos de ese haz por azimutal
        keep  = idx[order][::step_ray]       # 1 de cada step_ray
        mask[keep] = True

    return points[mask], semantic_tags[mask], mask


# 1. Cargar datos originales de GOOSE
with open("/home/yuriy/Universidad/2024-tfg-yuriy-moreno/scripts/Calcular Atenuacion/atenuacion_global.json") as f:
    goose_stats = json.load(f)

# 2. Mapeo manual de etiquetas GOOSE → CARLA
GOOSE_TO_CARLA_LABELS = {
    "0": 0,   # undefined
    "23": 1,   # asphalt → road
    "21": 2,   # sidewalk
    "38": 3,   # building
    "39": 4,   # wall
    "41": 5,   # fence
    "45": 6,   # pole
    "19": 7,   # traffic light
    "46": 8,   # traffic sign
    "17": 9,   # bush → vegetation
    "50": 10,  # low grass → terrain
    "53": 11,  # sky
    "14": 12,  # pedestrian
    "32": 13,  # rider
    "12": 14,  # car
    "34": 15,  # truck
    "15": 16,  # bus
    "35": 17,  # on_rail → train
    "20": 18,  # motorcycle
    "13": 19,  # bicycle
    "4": 20,  # obstacle → static
    "4": 21,  # dynamic
    "4": 22,  #  other       
    "54": 23,   # water
    "11": 24,  # road line
    "31": 25,  # soil → ground
    "43": 26,  # bridge
    "26": 27,  # rail track
    "42": 28  # guard rail
}

# 3. Crear diccionario final con etiquetas CARLA
ATTENUATION_CARLA = {}
for goose_label, carla_label in GOOSE_TO_CARLA_LABELS.items():
    if goose_label in goose_stats:
        ATTENUATION_CARLA[carla_label] = {
            "mean": goose_stats[goose_label]["media"],
            "std": goose_stats[goose_label]["std"]
        }

def custom_intensity(points: np.ndarray, semantic_tags: np.ndarray, attenuation_dict: dict, I0: float = 255.0, add_noise: bool = True) -> np.ndarray:
    """
    Calcula intensidades simuladas usando modelos de atenuación por clase.
    
    I = I₀ · exp(−α · d), donde α puede tener una desviación aleatoria si `add_noise` es True.

    Args:
        points (np.ndarray): Puntos XYZ (Nx3).
        semantic_tags (np.ndarray): Etiquetas semánticas por punto (N,).
        attenuation_dict (dict): Diccionario {class_id: {"mean": μ, "std": σ}} con los coeficientes α por clase.
        I0 (float): Intensidad máxima o ideal.
        add_noise (bool): Si True, añade ruido Gaussiano a los α.

    Returns:
        np.ndarray: Intensidades simuladas (N,).
    """

    distances = np.linalg.norm(points, axis=1)
    alphas = np.zeros_like(distances, dtype=np.float32)

    for label, stats in attenuation_dict.items():
        mask = semantic_tags == label
        if not np.any(mask):
            continue

        mu = stats["mean"]
        sigma = stats["std"] if add_noise else 0.0
        alpha_values = np.random.normal(mu, sigma, size=np.count_nonzero(mask)) if add_noise else np.full(np.count_nonzero(mask), mu)
        #alpha_values = np.clip(alpha_values, 0.0, None)
        alphas[mask] = alpha_values

    intensities = I0 * np.exp(-alphas * distances)
    #intensities /= I0  # Pasa a rango [0, 1]
    #intensities = I0 * np.exp(-1 * distances)
    return intensities


# Callback para procesar los datos del sensor LiDAR
def lidar_callback(lidar_data, downsampled_point_cloud, frame,lidar, lidar_range, hfov, upper_fov, lower_fov , noise_std=0.1, attenuation_coefficient=0.1, output_dir = 'dataset/lidar', output_bin_dir='dataset/lidar_bin'):
    """
    Procesa los datos del LiDAR obtenidos en cada frame.
    - Guarda una copia de la nube de puntos original (sin modificaciones).
    - Aplica ruido gaussiano y pérdidas de puntos.
    - Asigna colores basados en etiquetas semánticas.
    - Guarda la nube de puntos procesada.

    Args:
        lidar_data: carla.LidarMeasurement - Datos crudos del LiDAR.
        point_cloud: open3d.geometry.PointCloud - Nube de puntos procesada.
        raw_point_cloud: open3d.geometry.PointCloud - Nube de puntos original.
        frame: int - Número de frame actual.
        noise_std: float - Desviación estándar del ruido gaussiano aplicado.
        attenuation_coefficient: float - Coeficiente de atenuación para calcular la intensidad
        output_dir: Dirección de salida de la nube du puntos
    """
    # 1) Datos originales (sin modificaciones)
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))  # Ahora cada fila tiene 6 valores

    # Reflejar los datos en el eje X para mantener coherencia con el sistema de coordenadas de CARLA
    data[:, 0] = -data[:, 0]

    # Extraer las coordenadas XYZ y los valores de intensidad
    points = data[:, :3]
    intensities = data[:, -1]
    # Extraer etiquetas semánticas
    #semantic_tags = data[:, 5].view(np.uint32)  # Convertir los datos a enteros


    print("Sensor pose (lidar.get_transform):", lidar.get_transform())
    print("First LiDAR point (from raw_data):", points[0])


    print(f"Antes de las pérdidas: {len(points)} puntos")
    

    ######### Aplicar ruido y perdidas ############
    # Calcular la distancia de cada punto al sensor (suponiendo que el sensor está en el origen)
    #distances = np.linalg.norm(points, axis=1)  # Distancia euclidiana

    # Calcular la intensidad para cada punto utilizando la fórmula I = e^(-a * d)
    #intensities = np.exp(-attenuation_coefficient * distances)
    # 2) ring_id
    #channels = int(lidar.attributes['channels'])
    #ring_id = calculate_ring_id(points, channels,
    #                      lower_fov=lower_fov,
    #                      upper_fov=upper_fov)

    # 3) Submuestreo por haces (canales)
    #points, semantic_tags,mask = subsample_by_ring_id(points, semantic_tags, ring_id, step=4) # 32 canales
    #ring_id = ring_id[mask]

    # 4) submuestreo por rayos
    #points, semantic_tags, mask_ray = subsample_by_rays(points, semantic_tags, ring_id, step_ray=10)
    #ring_id = ring_id[mask_ray]

    #points, semantic_tags = add_cosmic_noise_points(
    #    points, semantic_tags, rate=0.01,
    #    max_range=lidar_range, hfov=hfov,
    #    upper_fov=upper_fov, lower_fov=lower_fov,
    #)

    #intensities = custom_intensity(points, semantic_tags, ATTENUATION_CARLA)


    # Mostrar el número de puntos eliminados con intensidad cero
    #print(f"Se eliminaron {zero_intensity_removed} puntos con intensidad cero.")
    print(f"Después de las pérdidas: {len(points)} puntos")


# 📌 Etapa 3: Puntos después del submuestreo
    #downsampled_colors = np.array([get_color_from_semantic(tag) for tag in semantic_tags]) / 255.0

    ########Antigua forma de crear el point cloud########
    # downsampled_point_cloud.points = o3d.utility.Vector3dVector(points)
    # downsampled_point_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors) # Colores RGB normalizados (Nx3)
    # # Guardar las etiquetas semánticas en el campo "normals"
    # downsampled_point_cloud.normals = o3d.utility.Vector3dVector(np.c_[semantic_tags, semantic_tags, semantic_tags])

    downsampled_point_cloud.point.positions = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)  # Puntos XYZ
    #downsampled_point_cloud.point.colors = o3d.core.Tensor(downsampled_colors, dtype=o3d.core.Dtype.Float32)  # Colores RGB normalizados (Nx3)
    #downsampled_point_cloud.point.labels = o3d.core.Tensor(semantic_tags.reshape(-1,1), dtype=o3d.core.Dtype.Int32)  # Etiquetas semánticas
    downsampled_point_cloud.point.intensities = o3d.core.Tensor(intensities.reshape(-1,1), dtype=o3d.core.Dtype.Float32)  # Intensidades normalizadas (Nx1)


    # 📂 Guardar el point cloud cada 20 frames
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_bin_dir):
        os.makedirs(output_bin_dir)

    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"lidar_points_{frame:04d}.pcd")
        print(f"Guardando archivo {filename}...")
        o3d.t.io.write_point_cloud(filename, downsampled_point_cloud,write_ascii=False)

        filename_bin = os.path.join(output_bin_dir, f"lidar_points_{frame:04d}.bin")
        
        points = downsampled_point_cloud.point["positions"].numpy()     # (N,3)
        intens = downsampled_point_cloud.point["intensities"].numpy()   # (N,1)
        points_4d = np.hstack([points, intens])                         # (N,4)

        print(f"Guardando archivo {filename_bin}...")
        points_4d.astype(np.float32).tofile(filename_bin)

################################Segmentación Lidar####################################
import argparse
import json

import detectionmetrics.utils.lidar as ul
from mmdet3d.datasets.transforms import (
    LoadPointsFromFile,
    LoadAnnotations3D,
    Pack3DDetInputs,
)
from mmengine.registry import FUNCTIONS
import numpy as np
import open3d as o3d
import torch
from torchvision.transforms import Compose

COLLATE_FN = FUNCTIONS.get("pseudo_collate")
np.random.seed(42)


def get_sample(points: np.ndarray, has_intensity=True):



    #points = sample["points"]
    n_feats = 4 if has_intensity else 3

    assert points.shape[1] >= n_feats, \
        f"Esperaba al menos {n_feats} features, tengo {points.shape[1]}"

    # Recortamos a las dimensiones que quiera el modelo (x,y,z / x,y,z,intensidad)
    points = points[:, : n_feats].astype(np.float32)

    sample = {
        "points": torch.from_numpy(points).float(), # Convertir a tensor
        "pts_semantic_mask_path": None,
        "sample_id": None,
        "sample_idx": None,
        "num_pts_feats": n_feats,
        "lidar_path": None,
    }

    # Lista de transforms (sin LoadPointsFromFile, porque ya tenemos el array)
    transforms = []
    # Esto es lo que Pack3DDetInputs espera: sample["points"]


    if sample["pts_semantic_mask_path"] is not None:
        transforms.append(
            LoadAnnotations3D(
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype="np.uint32",
                seg_offset=65536,
                dataset_type="semantickitti",
            )
        )
    transforms.append(
        Pack3DDetInputs(
            keys=["points", "pts_semantic_mask"],
            meta_keys=["sample_idx", "lidar_path", "num_pts_feats", "sample_id"],
        )
    )

    pipeline = Compose(transforms)
    sample = pipeline(sample)

    return sample


def get_lut(ontology):
    max_idx = max(class_data["idx"] for class_data in ontology.values())
    lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for class_data in ontology.values():
        lut[class_data["idx"]] = class_data["rgb"]
    return lut

def inference(
    model,
    points: np.ndarray,
    has_intensity: bool = False,
    ontology: dict | None = None,
    device: str = "cuda",
):
    """
    points: np.array (N,4) o (N,3)
    """
    # 1) Preparamos sample como antes, pero desde el array
    sample = get_sample(points, has_intensity)
    sample = COLLATE_FN([sample])
    sample = model.data_preprocessor(sample, training=False)
    inputs, data_samples = sample["inputs"], sample["data_samples"]

    # 2) Inferencia
    output = model(inputs, data_samples, mode="predict")[0]
    pred = output.pred_pts_seg.pts_semantic_mask.squeeze().cpu().numpy()  # (N,)

    # 3) Ontología / colores
    unique_labels = np.unique(pred)
    if ontology is None:
        ontology = {
            str(int(label)): {
                "idx": int(label),
                "rgb": np.random.randint(0, 256, size=3).tolist(),
            }
            for label in unique_labels
        }

    lut = get_lut(ontology)
    colors = lut[pred] / 255.0  # (N,3) en [0,1]

    return pred, colors, ontology

def load_segmentation_model_lidar(model_path: str, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device)
    return model.to(device).eval()
######################################################################################


# Función para crear y configurar el vehículo con sensores
def spawn_vehicle_lidar_camera_segmentation(world, bp, traffic_manager, delta):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 📌 LiDAR SEMÁNTICO (Densidad completa)
    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '1000000')
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
    lidar_bp.set_attribute('upper_fov', '30')  # Incrementar el límite superior
    lidar_bp.set_attribute('lower_fov', '-30')  # Reducir el límite inferior
    lidar_bp.set_attribute('horizontal_fov', '180')
    #lidar_bp.set_attribute('noise_stddev', '0.05')
    lidar_position = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)

    # Configuración de cámara RGB
    camera_bp = bp.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    #camera_transform = carla.Transform(carla.Location(x=-4.0, z=2.5))
    camera_transform = carla.Transform(carla.Location(x=1, z=1.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)



    vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle, lidar, camera

def set_camera_view(viz, third_person):
    ctr = viz.get_view_control()
    if third_person:
        # Configuración de cámara en tercera persona
        ctr.set_zoom(0.06)
        ctr.set_front([1.0, 0.0, 0.3])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([0, 0, 1])
    else:
        # Configuración de cámara en primera persona (ajusta según necesidad)
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([-1, 0, 0])

def camera_callback(image, display_surface, frame):
    global rgb_image

    output_dir = 'dataset/rgb'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convertir la imagen al formato numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa

    # Guardar la imagen como PNG
    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"rgb_{frame:04d}.png")
        print(f"Guardando imagen RGB en {filename}")
        #image_to_save = array[:, :, ::-1]  # Convertir de BGRA a RGB
        cv2.imwrite(filename, array)



    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    rgb_image = array.copy()  # Guardar la imagen RGB globalmente

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    #pygame.display.update(display_surface.get_rect())

import onnxruntime as ort
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
COMMON_ONTOLOGY_ORIGINAL = {
    "void": {"idx": 0, "rgb": [0, 0, 0]},
    "water": {"idx": 1, "rgb": [0, 0, 128]},
    "obstacle": {"idx": 2, "rgb": [255, 0, 0]},
    "nondrivable_vegetation": {"idx": 3, "rgb": [0, 128, 0]},
    "drivable_vegetation": {"idx": 4, "rgb": [0, 255, 0]},
    "unstable_terrain": {"idx": 5, "rgb": [128, 64, 32]},
    "stable_terrain": {"idx": 6, "rgb": [128, 128, 128]},
    "sky": {"idx": 7, "rgb": [128, 128, 255]},
}

##########################Segmentación visualización##########################
with open("/home/yuriy/Universidad/2024-tfg-yuriy-moreno/scripts/dataset/pinar_train/ontology.json", "r") as f:
    COMMON_ONTOLOGY = json.load(f)
    
    
def ontology_to_lut(ontology):
    """Convert ontology to look-up table."""
    max_idx = max(v["idx"] for v in ontology.values())
    lut = np.zeros((max_idx + 1, 3), dtype=np.uint8)
    for v in ontology.values():
        lut[v["idx"]] = v["rgb"]
    return lut

def load_segmentation_model(model_path, device="cuda"):
    print(f"Cargando modelo de segmentación desde {model_path}...")
    model = torch.load(model_path, map_location=device)
    model = model.to(device).eval()
    print("Modelo cargado correctamente.")
    return model

def run_segmentation_model(model, device="cuda", inference_mode= "mmsegmentation"):
    global rgb_image
    image = rgb_image
    lut = ontology_to_lut(COMMON_ONTOLOGY)
    
    image = Image.fromarray(image)
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    tensor = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)

    #model = model.to("cuda").eval()
    tensor = tensor.cuda()

    if inference_mode == "torch":
        pred = model(tensor).cpu()
    elif inference_mode == "mmsegmentation":
        pred = model.inference(
            tensor,
            [
                dict(
                    ori_shape=tensor.shape[2:],
                    img_shape=tensor.shape[2:],
                    pad_shape=tensor.shape[2:],
                    padding_size=[0, 0, 0, 0],
                )
            ],
        ).cpu()
    else:
        raise ValueError("Invalid inference mode provided.")
    
    if inference_mode != "mmsegmentation":
        pred = F.interpolate(pred, tensor.shape[2:], mode="bilinear")

    pred = torch.argmax(pred, dim=1).squeeze().numpy()
    pred_rgb = lut[pred]

    pred_rgb = Image.fromarray(pred_rgb)

    return pred_rgb

def segmentation_callback(image, display_surface, frame):
    global visual_error, rgb_image, segmentation_model

     # Asegurarse de que hay imagen RGB disponible
    if rgb_image is None:
        print("No hay imagen RGB disponible aún.")
        return
    

    output_dir = 'dataset/segmentation'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_raw_dir = 'dataset/segmentation_idx'
    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)


    # Ejecutar el modelo de segmentación neuronal
    pred_rgb = run_segmentation_model(segmentation_model,inference_mode="torch")

    # Convertir el resultado a numpy
    seg_array = np.array(pred_rgb)
    seg_array = np.ascontiguousarray(seg_array)

    ####### Control PID ########
     # --- Buscar color del terreno ---
    lower = np.array([70, 0, 70], dtype=np.uint8)
    upper = np.array([90, 20, 90], dtype=np.uint8)
    #lower = np.array([0, 0, 60], dtype=np.uint8)
    #upper = np.array([10, 10, 70], dtype=np.uint8)
    mask = cv2.inRange(seg_array, lower, upper)

    # --- Calcular centroide del terreno ---
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x_center = seg_array.shape[1] // 2
        visual_error =  cX - x_center

        # Visual debug
        cv2.circle(seg_array, (cX, cY), 10, (0, 255, 0), -1)
        cv2.line(seg_array, (x_center, 0), (x_center, seg_array.shape[0]), (0, 255, 255), 2)
    else:
        visual_error = 0  # si no detecta terreno, error nulo
      # Mostrar imagen combinada
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((seg_array, mask_rgb))
    surface = pygame.surfarray.make_surface(combined.swapaxes(0, 1))

    ####### Control PID ########
    
    # Guardar la imagen como PNG
    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"segmentation_{frame:04d}.png")
        print(f"Guardando imagen de segmentación en {filename}")

        cv2.imwrite(filename, seg_array[:, :, ::-1])

    # Crear la superficie de Pygame y mostrarla en la sección inferior de la pantalla
    display_surface.blit(surface, (0, 0))

####################################################################################

# Control del vehículo manual o automático
def vehicle_control(vehicle, max_speed_mps = 10):
    global manual_mode, previous_error, integral, visual_error, last_time
    control = carla.VehicleControl()  # Crear un control en blanco
    vehicle.set_autopilot(False)
    velocity = vehicle.get_velocity()
    speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

    # --- Medir tiempo entre iteraciones ---
    current_time = time.time()
    dt = current_time - last_time if 'last_time' in globals() else 0.05  # valor inicial por defecto
    last_time = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cleanup()
            sys.exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            manual_mode = not manual_mode
            #vehicle.set_autopilot(False)  # desactiva autopilot de CARLA
            mode = "manual" if manual_mode else "automático"
            print(f"\nCoche en modo {mode}.")
            time.sleep(0.3)  # Evitar múltiples activaciones por pulsación rápida

    # Aplicar controles si estamos en modo manual
    keys = pygame.key.get_pressed()
    if manual_mode:


        if keys[pygame.K_w]:
            if speed < max_speed_mps:
                control.throttle = 1.0
            else:
                control.throttle = 0.0
        if keys[pygame.K_s]:
            control.brake = 1.0
        if keys[pygame.K_a]:
            control.steer = -0.3
        elif keys[pygame.K_d]:
            control.steer = 0.3
        else:
            control.steer = 0.0
            
        vehicle.apply_control(control)
        return
    # Si no, estamos en modo automático
    # --- Modo automático: control visual PID ---
    Kp = 0.004
    Ki = 0.001
    Kd = 0.004

    error = visual_error
    integral += error*dt
    derivative = (error - previous_error)/dt
    steer_cmd = Kp * error + Ki * integral + Kd * derivative
    steer_cmd = np.clip(steer_cmd, -1.0, 1.0)
    previous_error = error

    # Velocidad adaptativa (más despacio al girar)
    v_max, v_min = 8.0, 1.0
    throttle_cmd = max(v_max - 2.0 * abs(steer_cmd), v_min)

    if speed < max_speed_mps:
        control.throttle = throttle_cmd / 10.0
    else:
        control.throttle = 0.0
    control.steer = steer_cmd
    control.brake = 0.0

    vehicle.apply_control(control)


# Configuración y ejecución del simulador
def main():
    pygame.init()
    # Configurar Pygame sin usar OpenGL explícitamente
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    width, height = 800, 600 # Aumentamos el tamaño vertical para RGB y segmentación

    # Crear la ventana principal de Pygame y dos subventanas como superficies
    screen = pygame.display.set_mode((width, height))  # Doble ancho para mostrar ambas vistas en paralelo
    pygame.display.set_caption("CARLA - RGB")

    #screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    #pygame.display.set_caption("CARLA Vehículo Control")

    rgb_surface = pygame.Surface((width, height))          # Subventana para la cámara RGB
    #segmentation_surface = pygame.Surface((width, height)) # Subventana para la segmentación

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    import time
    time.sleep(2.0)

    # Configurar el clima
    weather = carla.WeatherParameters(
        cloudiness=10.0,                # 0–100 → Nubosidad (0 = cielo despejado)
        precipitation=50.0,             # 0–100 → Lluvia (0 = sin lluvia, 100 = lluvia intensa)
        precipitation_deposits=50.0,    # 0–100 → Charcos en el suelo
        wind_intensity=50.0,            # 0–100 → Viento
        sun_azimuth_angle=90.0,         # 0–360° → Dirección del sol (horizontal)
        sun_altitude_angle=90.0,        # -90 a 90° → Altura del sol (90 = mediodía)
        fog_density=0.0,               # 0–100 → Grosor de la niebla
        fog_distance=10.0,              # metros → Inicio de la niebla
        wetness=50.0,                 # 0–100 → Humedad del asfalto
        fog_falloff=1.0,               # >0.0 → Caída de visibilidad con la distancia
        scattering_intensity=0.0,      # 0–∞ → Contribución de la luz en niebla volumétrica
        mie_scattering_scale=0.0,      # 0–∞ → Efecto de partículas grandes (humo, polvo)
        rayleigh_scattering_scale=0.0, # 0–∞ → Efecto de partículas pequeñas (moléculas del aire)
        dust_storm=0.0                # 0–1 → Intensidad de tormenta de polvo
    )
    #world.set_weather(weather)


    blueprint_library = world.get_blueprint_library()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    settings = world.get_settings()
    delta = 0.03
    settings.fixed_delta_seconds = delta
    settings.synchronous_mode = True
    world.apply_settings(settings)

    global actor_list, third_person_view
    vehicle, lidar, camera = spawn_vehicle_lidar_camera_segmentation(world, blueprint_library, traffic_manager, delta)

    ########################Segmentación neuronal########################
    print("Cargando modelo de segmentación neuronal...")
    global segmentation_model
    #segmentation_model = load_segmentation_model(
        #"/home/yuriy/Downloads/segformer_mit_b2_8xb1.pt"
        #"/home/yuriy/Universidad/2024-tfg-yuriy-moreno/models/prueba-epoch=epoch=98-step=step=1089-val_miou=val_miou=0.14.pt"
    #    "/home/yuriy/Universidad/2024-tfg-yuriy-moreno/models/pruebaGrande-epoch=epoch=43-step=step=10032-val_miou=val_miou=0.12.pt"
    #)   
    ####################################################################
    ########################Segmentación lidar########################
    lidar_seg_model = load_segmentation_model_lidar("/home/yuriy/Repositorios/proyecto-GAIA/Perception/scripts/lidar/Example/rellis_20-minkunet-epoch=17-step=70200-val_miou=0.37.pt")
    ####################################################################
    
    # Obtener atributos del LiDAR
    attrs = lidar.attributes  # diccionario de strings
    lidar_range    = float(attrs['range'])
    upper_fov      = float(attrs['upper_fov'])
    lower_fov      = float(attrs['lower_fov'])
    horizontal_fov = float(attrs['horizontal_fov'])


    actor_list.append(vehicle)
    actor_list.append(lidar)
    #actor_list.append(lidar_low_2)
    #actor_list.append(lidar_low_3)
    actor_list.append(camera)

    frame = 0 # Contador de frames

    # Llamada al callback de cámara RGB
    camera.listen(lambda image: camera_callback(image, rgb_surface, frame))

    # Llamada al callback de segmentación
    #def process_segmentation():
    #    segmentation_callback(None, segmentation_surface, frame)

    
    
    downsampled_point_cloud = o3d.t.geometry.PointCloud()  # Nube submuestreada
    
    original = o3d.geometry.PointCloud()
    segmented = o3d.geometry.PointCloud()  # Legacy solo para visualización



    lidar.listen(lambda data: lidar_callback(data, 
            downsampled_point_cloud, 
            frame,
            lidar,
            noise_std=0.1,
            output_dir='dataset/lidar',
            lidar_range=lidar_range,
            hfov=horizontal_fov,
            upper_fov=upper_fov,
            lower_fov=lower_fov))

    # Utilizar VisualizerWithKeyCallback
 # Crear dos visualizadores SEPARADOS
 
    viz_original = o3d.visualization.Visualizer() # Puntos después del submuestreo
    viz_segmented = o3d.visualization.Visualizer()
    

    viz_original.create_window(window_name="Lidar Original", width=960, height=540, left=1100, top=100)
    viz_segmented.create_window(window_name="Lidar Segmentado", width=960, height=540, left=1100, top=100)

    
    for viz in [viz_original, viz_segmented]:
        viz.get_render_option().background_color = [0.05, 0.05, 0.05]
        viz.get_render_option().point_size = 0.7
        viz.get_render_option().show_coordinate_frame = True

    third_person_view = True

    # Añadir el point cloud al visualizador en la primera iteración
    lidar_data_received = False

    # Configurar el callback de tecla 'v'
    def toggle_camera_view(_):
        global third_person_view
        third_person_view = not third_person_view
        set_camera_view(viz_original, third_person_view)
        set_camera_view(viz_segmented, third_person_view)
        print("Cambiando a tercera persona" if third_person_view else "Cambiando a primera persona")
        return True  # Devolver True para continuar el evento de renderizado

    # Asignar el callback al visualizador
    
    #viz_raw.register_key_callback(ord("V"), toggle_camera_view)
    #viz.register_key_callback(ord("V"), toggle_camera_view)

    # dt0 = datetime.now()

    while True:
        dt0 = datetime.now() # Inicio de medición de FPS

        world.tick()  # Asegurar sincronización

        #process_segmentation()
        vehicle_control(vehicle)

        # 📌 Calcular y mostrar velocidad
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        font = pygame.font.SysFont("Arial", 20)
        speed_text = font.render(f"Velocidad: {speed * 3.6:.1f} km/h", True, (255, 255, 255))
        rgb_surface.blit(speed_text, (10, 10))

        # Dibujar ambas subventanas en la ventana principal
        screen.blit(rgb_surface, (0, 0))                    # Poner RGB a la izquierda
        #screen.blit(segmentation_surface, (width, 0))       # Poner Segmentación a la derecha
        pygame.display.flip()

        pygame.event.pump()  # Procesar eventos Pygame

        ######nuevo########
        #downsampled_point_cloud_legacy = downsampled_point_cloud.to_legacy()

        if frame == 5 and not lidar_data_received:
  
            viz_segmented.add_geometry(segmented)
            viz_original.add_geometry(original) 
            ########

            #viz_downsampled.add_geometry(downsampled_point_cloud)# Nube con submuestreo
            lidar_data_received = True
            print("Geometry added to the visualizer")
            set_camera_view(viz_original, third_person_view)
            set_camera_view(viz_segmented, third_person_view)

        ################################333
        try:
            #labels = downsampled_point_cloud.point["labels"].numpy().flatten()
            positions = downsampled_point_cloud.point["positions"].numpy()
            intensities = downsampled_point_cloud.point["intensities"].numpy()  # (N,1)
        except RuntimeError:
            print("⚠️ Frame descartado por conflicto de acceso.")
            continue
        ###############Segmentación Lidar####################
        if positions.shape[0] == 0:
            frame += 1
            continue
        points_4d = np.hstack([positions, intensities])  # (N,4)

        pred_labels, seg_colors, _ = inference(
            lidar_seg_model,
            points_4d,
            has_intensity=False,
            ontology=COMMON_ONTOLOGY,  # o pasa tu ontología fija si la tienes
        )
        ############################################################



        segmented.points = o3d.utility.Vector3dVector(positions)
        segmented.colors = o3d.utility.Vector3dVector(seg_colors)

        original.points = o3d.utility.Vector3dVector(positions)
        
        # Normalizar intensidades a [0,1]
        int_flat = intensities.reshape(-1)
        int_norm = (int_flat - int_flat.min()) / (int_flat.ptp() + 1e-6)
        orig_colors = np.stack([int_norm, int_norm, int_norm], axis=1)  # (N,3)

        original.colors = o3d.utility.Vector3dVector(orig_colors)

        #########################################
        #viz_downsampled.update_geometry(downsampled_point_cloud)
        viz_segmented.update_geometry(segmented)
        viz_original.update_geometry(original)
        #viz_downsampled.poll_events()


        #viz_downsampled.update_renderer()

        # time.sleep(0.03)
        #world.tick()


        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            fps = 1.0 / process_time.total_seconds()
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        #dt0 = datetime.now()
        frame += 1

        if not viz_segmented.poll_events() or not viz_original.poll_events():
            print("Exiting visualization")
            break
        viz_segmented.update_renderer()
        viz_original.update_renderer
    cleanup()

def cleanup():
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()

