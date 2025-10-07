import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from glob import glob

# Ruta a los archivos
lidar_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train/2022-07-27_hoehenkirchner_forst"
label_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train/2022-07-27_hoehenkirchner_forst"

# --- Recolectar todos los archivos .bin
bin_files = sorted(glob(os.path.join(lidar_root, "*.bin")))

# --- Inicializar acumulador por clase
acumulador_por_clase = defaultdict(list)

I0 = 255.0  # Remisión ideal para normalización

for bin_path in bin_files:
    base = os.path.basename(bin_path).replace("_vls128.bin", "")
    label_path = os.path.join(label_root, base + "_goose.label")

    if not os.path.exists(label_path):
        print(f"Advertencia: No se encontró el archivo de etiquetas para {bin_path}.")
        continue

    # --- Paso 1: Leer puntos e intensidades

    scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    puntos = scan[:, :3]
    remission = scan[:, 3]

    # --- Paso 2: Leer etiquetas semánticas
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_label = labels & 0xFFFF

    # Distancia euclidiana por punto desde el origen del sensor
    distancias = np.linalg.norm(puntos, axis=1)

    # Evita división por 0 o log(0)
    mask_valid = (remission > 0.1) & (distancias > 0.1)

    if np.count_nonzero(mask_valid) < 100:  # Evitar nubes vacías o corruptas
        continue

    # Aplicamos la fórmula de atenuación:
    atenuaciones = -np.log(remission[mask_valid] / I0) / distancias[mask_valid]


    # Paso 1: Extraer solo los valores válidos
    valid_alphas = atenuaciones
    valid_labels = sem_label[mask_valid]


    # Paso 2: Calcular percentiles para filtrar extremos
    p1, p99 = np.percentile(valid_alphas, [1, 99])

    # Paso 3: Crear máscara que filtre valores "centrales"
    mask_range = (valid_alphas > p1) & (valid_alphas < p99)

    # Paso 4: Asignar a cada clase los valores filtrados
    for alpha, label in zip(valid_alphas[mask_range], valid_labels[mask_range]):
        acumulador_por_clase[label].append(alpha)



# --- Calcular estadísticas finales
estadisticas_por_clase = {
    int(clase): {
        "media": float(np.mean(valores)),
        "mediana": float(np.median(valores)),
        "std": float(np.std(valores)),
        "n": int(len(valores))
    }
    for clase, valores in acumulador_por_clase.items()
}

# --- Ordenar las clases por número de etiqueta
estadisticas_por_clase_ordenadas = dict(sorted(estadisticas_por_clase.items()))

# Guardar en un archivo JSON
with open("atenuacion_por_clase.json", "w") as f:
    json.dump(estadisticas_por_clase_ordenadas, f, indent=2)


print(f"\n✅ Proceso completado. Clases procesadas: {len(estadisticas_por_clase)}")