import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Configuración
lidar_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train"
label_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train"
escenario = "2022-07-22_flight"
clase_diana = 41  # coche
limite_frames = 200  # para pruebas

# Rutas
carpeta_lidar = os.path.join(lidar_root, escenario)
carpeta_label = os.path.join(label_root, escenario)
bin_files = sorted(glob(os.path.join(carpeta_lidar, "*.bin")))

# Acumuladores
todas_distancias = []
todas_remisiones = []

for idx, bin_path in enumerate(bin_files[:limite_frames]):
    base = os.path.basename(bin_path).replace("_vls128.bin", "")
    label_path = os.path.join(carpeta_label, base + "_goose.label")
    if not os.path.exists(label_path):
        continue

    try:
        scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        puntos = scan[:, :3]
        remission = scan[:, 3]
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    except Exception as e:
        print(f"Error en frame {idx}: {e}")
        continue

    distancias = np.linalg.norm(puntos, axis=1)
    mask = (remission > 0.01) & (distancias > 0.1) & (labels == clase_diana)

    todas_distancias.append(distancias[mask])
    todas_remisiones.append(remission[mask])

# Concatenar todo
todas_distancias = np.concatenate(todas_distancias)
todas_remisiones = np.concatenate(todas_remisiones)

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(todas_distancias, todas_remisiones, s=2, alpha=0.3)
plt.xlabel("Distancia (m)")
plt.ylabel("Remisión")
plt.title(f"Scatter Remisión vs Distancia - Clase {clase_diana}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"scatter_clase_{clase_diana}.png", dpi=300)
plt.show()

