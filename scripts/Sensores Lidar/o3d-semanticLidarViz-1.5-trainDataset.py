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
visual_error = 0.0
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

def add_noise_to_lidar(points, std_dev):
    """
    Agrega ruido gaussiano a la nube de puntos LiDAR.
    Args:
        points: np.array (N, 3) - Coordenadas XYZ del LiDAR semántico
        std_dev: float - Desviación estándar del ruido
    Returns:
        np.array (N, 3) - Nube de puntos con ruido
    """
    noise = np.random.normal(0, std_dev, points.shape)  # Ruido Gaussiano
    noisy_points = points + noise
    return noisy_points

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


def drop_points(points, semantic_tags, intensities, drop_rate=0.45, intensity_limit=0.8, zero_intensity_drop=0.4, low_intensity_threshold=0.01 ):
    """
    Aplica pérdidas de puntos con la misma lógica que el LiDAR de CARLA.
    - Primero aplica pérdida aleatoria.
    - Luego protege puntos con intensidad alta.
    - Luego elimina puntos con intensidad cero con probabilidad extra.

    Args:
    - points: np.array (N, 3) - Coordenadas XYZ del LiDAR (sin información adicional como colores o etiquetas).
    - semantic_tags: np.array (N,) - Etiquetas semánticas de cada punto.
    - intensities: np.array (N,) - Intensidades de los puntos.
    - drop_rate: float - Probabilidad de eliminar un punto de forma aleatoria.
    - intensity_limit: float - Umbral por encima del cual no se eliminan puntos (protección de puntos con alta intensidad).
    - zero_intensity_drop: float - Probabilidad de eliminar puntos con intensidad cero.
    - low_intensity_threshold: float - Umbral bajo de intensidad a partir del cual se eliminan los puntos.

    Returns:
        np.array (M, 3) - Nube de puntos con menos puntos
        np.array (M,) - Etiquetas semánticas filtradas
    """

    num_points = points.shape[0]

    # Máscara de eliminación aleatoria
    mask_drop_random = np.random.rand(num_points) < drop_rate

    # Restaurar puntos con alta intensidad (> intensity_limit)
    mask_keep_high_intensity = intensities > intensity_limit
    mask_drop_random[mask_keep_high_intensity] = False  # No eliminamos estos puntos

    # Verificar cuántos puntos tienen intensidad muy baja
    num_low_intensity = np.sum(intensities < low_intensity_threshold)
    print(f"Cantidad de puntos con intensidad menor a {low_intensity_threshold}: {num_low_intensity}")

    # Máscara para eliminar puntos con intensidad baja (por debajo del umbral)
    mask_drop_low_intensity = (intensities < low_intensity_threshold) & (np.random.rand(num_points) < zero_intensity_drop)


    # Contamos cuántos puntos con intensidad cero se eliminan
    zero_intensity_dropped = np.sum(mask_drop_low_intensity)
    
    # Combinamos todas las eliminaciones
    final_mask = ~mask_drop_random & ~mask_drop_low_intensity

    return points[final_mask],semantic_tags[final_mask],intensities[final_mask], zero_intensity_dropped

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
def lidar_callback(lidar_data, downsampled_point_cloud, frame,lidar, lidar_range, hfov, upper_fov, lower_fov , noise_std=0.1, attenuation_coefficient=0.1, output_dir = 'dataset/lidar'):
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
    data = np.reshape(data, (int(data.shape[0] / 6), 6))  # Ahora cada fila tiene 6 valores

    # Reflejar los datos en el eje X para mantener coherencia con el sistema de coordenadas de CARLA
    data[:, 0] = -data[:, 0]

    # Extraer las coordenadas XYZ y los valores de intensidad
    points = data[:, :3]

    # Extraer etiquetas semánticas
    semantic_tags = data[:, 5].view(np.uint32)  # Convertir los datos a enteros


    print("Sensor pose (lidar.get_transform):", lidar.get_transform())
    print("First LiDAR point (from raw_data):", points[0])


    print(f"Antes de las pérdidas: {len(points)} puntos")
    

    ######### Aplicar ruido y perdidas ############
    # Calcular la distancia de cada punto al sensor (suponiendo que el sensor está en el origen)
    #distances = np.linalg.norm(points, axis=1)  # Distancia euclidiana

    # Calcular la intensidad para cada punto utilizando la fórmula I = e^(-a * d)
    #intensities = np.exp(-attenuation_coefficient * distances)

    # 2) ring_id
    channels = int(lidar.attributes['channels'])
    ring_id = calculate_ring_id(points, channels,
                          lower_fov=lower_fov,
                          upper_fov=upper_fov)

    # 3) Submuestreo por haces (canales)
    #points, semantic_tags,mask = subsample_by_ring_id(points, semantic_tags, ring_id, step=4) # 32 canales
    #ring_id = ring_id[mask]

    # 4) submuestreo por rayos
    #points, semantic_tags, mask_ray = subsample_by_rays(points, semantic_tags, ring_id, step_ray=10)
    #ring_id = ring_id[mask_ray]

    # Aplicar ruido a los puntos
    #points = add_noise_to_lidar(points, noise_std)

    points, semantic_tags = add_cosmic_noise_points(
        points, semantic_tags, rate=0.01,
        max_range=lidar_range, hfov=hfov,
        upper_fov=upper_fov, lower_fov=lower_fov,
    )

    intensities = custom_intensity(points, semantic_tags, ATTENUATION_CARLA)

    # Aplicar pérdidas de puntos según las reglas del LiDAR
    #points, semantic_tags, intensities, zero_intensity_removed = drop_points(points, semantic_tags, intensities)

    # Mostrar el número de puntos eliminados con intensidad cero
    #print(f"Se eliminaron {zero_intensity_removed} puntos con intensidad cero.")
    print(f"Después de las pérdidas: {len(points)} puntos")


# 📌 Etapa 3: Puntos después del submuestreo
    downsampled_colors = np.array([get_color_from_semantic(tag) for tag in semantic_tags]) / 255.0

    ########Antigua forma de crear el point cloud########
    # downsampled_point_cloud.points = o3d.utility.Vector3dVector(points)
    # downsampled_point_cloud.colors = o3d.utility.Vector3dVector(downsampled_colors) # Colores RGB normalizados (Nx3)
    # # Guardar las etiquetas semánticas en el campo "normals"
    # downsampled_point_cloud.normals = o3d.utility.Vector3dVector(np.c_[semantic_tags, semantic_tags, semantic_tags])

    downsampled_point_cloud.point.positions = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)  # Puntos XYZ
    downsampled_point_cloud.point.colors = o3d.core.Tensor(downsampled_colors, dtype=o3d.core.Dtype.Float32)  # Colores RGB normalizados (Nx3)
    downsampled_point_cloud.point.labels = o3d.core.Tensor(semantic_tags.reshape(-1,1), dtype=o3d.core.Dtype.Int32)  # Etiquetas semánticas
    downsampled_point_cloud.point.intensities = o3d.core.Tensor(intensities.reshape(-1,1), dtype=o3d.core.Dtype.Float32)  # Intensidades normalizadas (Nx1)


    # 📂 Guardar el point cloud cada 20 frames
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"lidar_points_{frame:04d}.pcd")
        print(f"Guardando archivo {filename}...")
        o3d.t.io.write_point_cloud(filename, downsampled_point_cloud,write_ascii=False)

# Función para crear y configurar el vehículo con sensores
def spawn_vehicle_lidar_camera_segmentation(world, bp, traffic_manager, delta):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 📌 LiDAR SEMÁNTICO (Densidad completa)
    lidar_bp = bp.find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '1000000')
    # lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
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

    # Configuración de cámara de segmentación semántica
    segmentation_bp = bp.find('sensor.camera.semantic_segmentation')
    segmentation_bp.set_attribute('image_size_x', '800')
    segmentation_bp.set_attribute('image_size_y', '600')
    segmentation_bp.set_attribute('fov', '90')
    segmentation_transform = carla.Transform(carla.Location(x=1, z=1.5))
    segmentation_camera = world.spawn_actor(segmentation_bp, segmentation_transform, attach_to=vehicle)


    vehicle.set_autopilot(True, traffic_manager.get_port())
    return vehicle, lidar, camera, segmentation_camera

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
    output_dir = 'dataset/rgb'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convertir la imagen al formato numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa

    # Guardar la imagen como PNG
    if frame % 5 == 0:
        filename = os.path.join(output_dir, f"rgb_{frame:04d}.png")
        print(f"Guardando imagen RGB en {filename}")
        #image_to_save = array[:, :, ::-1]  # Convertir de BGRA a RGB
        cv2.imwrite(filename, array)



    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    #pygame.display.update(display_surface.get_rect())

def segmentation_callback(image, display_surface, frame):
    global visual_error

    output_dir = 'dataset/segmentation'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_raw_dir = 'dataset/segmentation_idx'
    if not os.path.exists(output_raw_dir):
        os.makedirs(output_raw_dir)

    ########### Obtener etiquetas crudas ###########
    # Convertir los datos de la imagen a un array numpy
    raw_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw_array = np.reshape(raw_array, (image.height, image.width, 4))
    # Extraer los canales RGB y convertir de BGRA a RGB
    seg_raw_array = np.copy(raw_array[:, :, :3])  # copiar para hacerla editable

    raw_labels = seg_raw_array[:, :, 2]
    ########### Obtener etiquetas crudas ###########

    # Convertir la imagen de segmentación directamente usando CityScapesPalette
    image.convert(carla.ColorConverter.CityScapesPalette)

    # Convertir los datos de la imagen a un array numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))

    # Extraer los canales RGB y convertir de BGRA a RGB
    seg_array = np.copy(array[:, :, :3])  # copiar para hacerla editable
    seg_array = seg_array[:, :, ::-1]
    seg_array.setflags(write=1)  # Asegurarse de que el array es escribible
    seg_array = np.ascontiguousarray(seg_array)  # 👈 garantiza layout correcto

    ####### Control PID ########
     # --- Buscar color del terreno ---
    lower = np.array([70, 0, 70], dtype=np.uint8)
    upper = np.array([90, 20, 90], dtype=np.uint8)
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
    if frame % 5 == 0:
        filename = os.path.join(output_dir, f"segmentation_{frame:04d}.png")
        raw_filename = os.path.join(output_raw_dir, f"segmentation_idx_{frame:04d}.png")

        print(f"Guardando imagen de segmentación en {filename}")
        cv2.imwrite(filename, seg_array[:, :, ::-1])  # Convertir de RGB a BGR para OpenCV
        cv2.imwrite(raw_filename, raw_labels)


    # Crear la superficie de Pygame y mostrarla en la sección inferior de la pantalla
    #surface = pygame.surfarray.make_surface(seg_array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))

    # Llamar a la función para mostrar las etiquetas
    #display_labels(display_surface)



    
    # Actualizar solo el área de la pantalla donde se muestra la segmentación
    #pygame.display.update(pygame.Rect(0, 600, image.width, image.height))



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
    width, height = 832, 608 # Aumentamos el tamaño vertical para RGB y segmentación

    # Crear la ventana principal de Pygame y dos subventanas como superficies
    screen = pygame.display.set_mode((width * 2, height))  # Doble ancho para mostrar ambas vistas en paralelo
    pygame.display.set_caption("CARLA - RGB y Segmentación")

    #screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    #pygame.display.set_caption("CARLA Vehículo Control")

    rgb_surface = pygame.Surface((width, height))          # Subventana para la cámara RGB
    segmentation_surface = pygame.Surface((width, height)) # Subventana para la segmentación

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
    vehicle, lidar, camera, segmentation_camera = spawn_vehicle_lidar_camera_segmentation(world, blueprint_library, traffic_manager, delta)

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
    actor_list.append(segmentation_camera)

    # Llamada al callback de cámara RGB
    camera.listen(lambda image: camera_callback(image, rgb_surface, frame))

    # Llamada al callback de segmentación
    segmentation_camera.listen(lambda image: segmentation_callback(image, segmentation_surface, frame))

    
    
    downsampled_point_cloud = o3d.t.geometry.PointCloud()  # Nube submuestreada
    downsampled_point_cloud_legacy = o3d.geometry.PointCloud()  # Legacy solo para visualización

    frame = 0 # Contador de frames

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
 
    viz_downsampled = o3d.visualization.Visualizer() # Puntos después del submuestreo

    

    viz_downsampled.create_window(window_name="Lidar Con Submuestreo", width=960, height=540, left=1100, top=100)

    
    for viz in [viz_downsampled]:
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
        set_camera_view(viz_downsampled, third_person_view)
        print("Cambiando a tercera persona" if third_person_view else "Cambiando a primera persona")
        return True  # Devolver True para continuar el evento de renderizado

    # Asignar el callback al visualizador
    
    #viz_raw.register_key_callback(ord("V"), toggle_camera_view)
    #viz.register_key_callback(ord("V"), toggle_camera_view)

    # dt0 = datetime.now()

    while True:
        dt0 = datetime.now() # Inicio de medición de FPS

        world.tick()  # Asegurar sincronización

        vehicle_control(vehicle)

        # 📌 Calcular y mostrar velocidad
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        font = pygame.font.SysFont("Arial", 20)
        speed_text = font.render(f"Velocidad: {speed * 3.6:.1f} km/h", True, (255, 255, 255))
        rgb_surface.blit(speed_text, (10, 10))

        # Dibujar ambas subventanas en la ventana principal
        screen.blit(rgb_surface, (0, 0))                    # Poner RGB a la izquierda
        screen.blit(segmentation_surface, (width, 0))       # Poner Segmentación a la derecha
        pygame.display.flip()

        pygame.event.pump()  # Procesar eventos Pygame

        ######nuevo########
        #downsampled_point_cloud_legacy = downsampled_point_cloud.to_legacy()

        if frame == 5 and not lidar_data_received:
  
            viz_downsampled.add_geometry(downsampled_point_cloud_legacy) 
            ########

            #viz_downsampled.add_geometry(downsampled_point_cloud)# Nube con submuestreo
            lidar_data_received = True
            print("Geometry added to the visualizer")
            set_camera_view(viz_downsampled, third_person_view)



        #pcd_legacy.points = o3d.utility.Vector3dVector(np.asarray(downsampled_point_cloud.point["positions"]))
        #downsampled_point_cloud_legacy.points = o3d.utility.Vector3dVector(downsampled_point_cloud.point["positions"].numpy())

        # Opcional: colores si tienes etiquetas
        # colors = np.array([get_color_from_semantic(int(t)) for t in downsampled_point_cloud.point["labels"].numpy().flatten()]) / 255.0
        # downsampled_point_cloud_legacy.colors = o3d.utility.Vector3dVector(colors)

        ################################333
        try:
            labels = downsampled_point_cloud.point["labels"].numpy().flatten()
            positions = downsampled_point_cloud.point["positions"].numpy()
        except RuntimeError:
            print("⚠️ Frame descartado por conflicto de acceso.")
            continue

        if labels.shape[0] == positions.shape[0]:
            colors = np.array([get_color_from_semantic(int(t)) for t in labels]) / 255.0
            downsampled_point_cloud_legacy.points = o3d.utility.Vector3dVector(positions)
            downsampled_point_cloud_legacy.colors = o3d.utility.Vector3dVector(colors)
        else:
            print(f"❌ Mismatch de etiquetas/puntos: {labels.shape[0]} vs {positions.shape[0]}")
            continue


        #########################################
        #viz_downsampled.update_geometry(downsampled_point_cloud)
        viz_downsampled.update_geometry(downsampled_point_cloud_legacy)

        viz_downsampled.poll_events()


        viz_downsampled.update_renderer()

        # time.sleep(0.03)
        #world.tick()


        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            fps = 1.0 / process_time.total_seconds()
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        #dt0 = datetime.now()
        frame += 1

        if not viz_downsampled.poll_events():
            print("Exiting visualization")
            break

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

