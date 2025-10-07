import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# --- Configuración ---
lidar_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train"
label_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train"
escenario = "2022-08-18_putzbrunn_feldwege"

clase_objetivo = 4              # Clase a inspeccionar
bin_size = 10                  # Tamaño del bin en metros
min_muestras_por_bin = 10      # Mínimo de puntos para considerar promedio válido

# --- Binning ---
bin_edges = np.arange(0, 200 + bin_size, bin_size)
bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]}" for i in range(len(bin_edges) - 1)]

# --- Acumuladores ---
intensidades_por_bin = {r: [] for r in bin_labels}
conteo_por_bin = {r: 0 for r in bin_labels}

carpeta_lidar = os.path.join(lidar_root, escenario)
carpeta_label = os.path.join(label_root, escenario)
bin_files = sorted(glob(os.path.join(carpeta_lidar, "*.bin")))

print(f"Procesando {len(bin_files)} archivos de {escenario} para clase {clase_objetivo}")

for bin_path in bin_files:
    base = os.path.basename(bin_path).replace("_vls128.bin", "")
    label_path = os.path.join(carpeta_label, base + "_goose.label")
    if not os.path.exists(label_path):
        continue

    try:
        scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
        puntos = scan[:, 0:3]
        remission = scan[:, 3]
        labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
    except:
        continue

    distancias = np.linalg.norm(puntos, axis=1)
    mask = (remission > 0.01) & (distancias > 0.1) & (labels == clase_objetivo)

    remission = remission[mask]
    distancias = distancias[mask]

    for i in range(len(bin_edges) - 1):
        bmin, bmax = bin_edges[i], bin_edges[i+1]
        in_bin = (distancias > bmin) & (distancias <= bmax)
        bin_key = f"{bmin}-{bmax}"
        intensidades_por_bin[bin_key].extend(remission[in_bin])
        conteo_por_bin[bin_key] += np.sum(in_bin)

# --- Cálculo final ---
medias = []
conteos = []

for key in bin_labels:
    datos = intensidades_por_bin[key]
    n = conteo_por_bin[key]
    if n >= min_muestras_por_bin:
        medias.append(np.mean(datos))
    else:
        medias.append(0.0)
    conteos.append(n)

# --- Visualización ---
x = np.arange(len(bin_labels))

fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.bar(x, medias, width=0.4, align='center', color='tab:blue', label="Intensidad promedio")
ax1.set_ylabel("Intensidad promedio", color='tab:blue')
ax1.set_xlabel("Bin de distancia (m)")
ax1.set_xticks(x)
ax1.set_xticklabels(bin_labels, rotation=90)

ax2 = ax1.twinx()
ax2.plot(x, conteos, color='tab:orange', marker='o', label="Número de puntos")
ax2.set_ylabel("Número de puntos", color='tab:orange')

plt.title(f"Clase {clase_objetivo} – Intensidad vs distancia")
plt.tight_layout()
plt.grid(True)
plt.show()
