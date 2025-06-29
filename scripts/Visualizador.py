import open3d as o3d
import numpy as np

# Cargar archivo .pcd con atributos
pcd_t = o3d.t.io.read_point_cloud("dataset/lidar/lidar_points_0020.pcd")
print("Atributos disponibles:", pcd_t.point)

# Extraer posiciones y colores personalizados
points = pcd_t.point["positions"].numpy()

if "labels" in pcd_t.point:
    labels = pcd_t.point["labels"].numpy().flatten()
    # Función para colorear por etiquetas (usa tu mapa o asigna colores únicos)
    def get_color_from_label(label):
        np.random.seed(label)
        return np.random.rand(3)  # RGB aleatorio fijo por etiqueta
    colors = np.array([get_color_from_label(l) for l in labels])
elif "intensities" in pcd_t.point:
    intensities = pcd_t.point["intensities"].numpy().flatten()
    intensities = np.clip(intensities, 0, 1)  # Normalizar si es necesario
    colors = np.tile(intensities[:, None], (1, 3))  # Escala de grises
else:
    colors = np.zeros_like(points)

# Crear nube legacy para visualizar
pcd_legacy = o3d.geometry.PointCloud()
pcd_legacy.points = o3d.utility.Vector3dVector(points)
pcd_legacy.colors = o3d.utility.Vector3dVector(colors)

# Mostrar con GUI tradicional (mejor compatibilidad)
o3d.visualization.draw_geometries([pcd_legacy], window_name="PCD Viewer")

