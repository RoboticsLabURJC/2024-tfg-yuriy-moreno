import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURACION ---
BIN_PATH = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train/2023-05-17_neubiberg_sunny/2023-05-17_neubiberg_sunny__0366_1684329686270285947_vls128.bin"  # Reemplaza con tu ruta real
LABEL_PATH = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train/2023-05-17_neubiberg_sunny/2023-05-17_neubiberg_sunny__0366_1684329686270285947_goose.label"  # Reemplaza

# --- Cargar datos ---
scan = np.fromfile(BIN_PATH, dtype=np.float32).reshape((-1, 4))
puntos = scan[:, :3]
remission = scan[:, 3]

# --- Calcular distancia ---
distancias = np.linalg.norm(puntos, axis=1)

# --- Cargar etiquetas ---
labels = np.fromfile(LABEL_PATH, dtype=np.uint32) & 0xFFFF

# --- Filtro básico ---
mask_valid = (remission > 0.01) & (distancias > 0.1)
distancias = distancias[mask_valid]
remission = remission[mask_valid]
labels = labels[mask_valid]

# --- Visualización ---
plt.figure(figsize=(10, 6))
plt.scatter(distancias, remission, c=labels, cmap='tab20', s=1, alpha=0.6)
plt.xlabel("Distancia (m)")
plt.ylabel("Remisión")
plt.title("Relación distancia vs remisión (una imagen LiDAR)")
plt.grid(True)
plt.tight_layout()
plt.show()
