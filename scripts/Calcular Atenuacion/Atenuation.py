import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Simula que ya cargaste los datos desde los archivos .bin y .label
# puntos.shape = (N, 3), remission.shape = (N,), sem_label.shape = (N,)
# Supón que ya tienes:
# puntos, remission, sem_label
# Ruta a tus archivos
bin_file = "/home/yuriy/Descargas/GOOSE/goose_3d_val/lidar/val/2022-08-30_siegertsbrunn_feldwege/2022-08-30_siegertsbrunn_feldwege__0528_1661860582736903436_vls128.bin"
label_file = "/home/yuriy/Descargas/GOOSE/goose_3d_val/labels/val/2022-08-30_siegertsbrunn_feldwege/2022-08-30_siegertsbrunn_feldwege__0528_1661860582736903436_goose.label"

# --- Paso 1: Leer puntos e intensidades
scan = np.fromfile(bin_file, dtype=np.float32).reshape((-1, 4))
puntos = scan[:, :3]
remission = scan[:, 3]

# --- Paso 2: Leer etiquetas semánticas
labels = np.fromfile(label_file, dtype=np.uint32)
sem_label = labels & 0xFFFF

I0 = 255.0  # Remisión ideal para normalización

# Distancia euclidiana por punto desde el origen del sensor
distancias = np.linalg.norm(puntos, axis=1)

# Evita división por 0 o log(0)
mask_valid = (remission > 0.1) & (distancias > 0.1)

# Aplicamos la fórmula de atenuación:
atenuaciones = np.zeros_like(remission)
atenuaciones = -np.log(remission[mask_valid] / I0) / distancias[mask_valid]


# Paso 1: Extraer solo los valores válidos
valid_alphas = atenuaciones
valid_labels = sem_label[mask_valid]


# Paso 2: Calcular percentiles para filtrar extremos
p1, p99 = np.percentile(valid_alphas, [1, 99])

# Paso 3: Crear máscara que filtre valores "centrales"
mask_range = (valid_alphas > p1) & (valid_alphas < p99)

# Paso 4: Asignar a cada clase los valores filtrados
atenuacion_por_clase = defaultdict(list)
for alpha, label in zip(valid_alphas[mask_range], valid_labels[mask_range]):
    atenuacion_por_clase[label].append(alpha)


# Agrupar por etiqueta semántica
# atenuacion_por_clase = defaultdict(list)
# for alpha, label in zip(atenuaciones, sem_label):
#     if alpha > 0 and alpha < 1.5:  # Descarta valores extremos o inválidos
#         atenuacion_por_clase[label].append(alpha)

# Mostrar resumen por clase
for clase, valores in atenuacion_por_clase.items():
    valores_np = np.array(valores)
    print(f"Clase {clase:02d}: Media={valores_np.mean():.3f}, Std={valores_np.std():.3f}, n={len(valores_np)}")


# Calcular medias por clase
media_por_clase = {int(k): float(np.mean(v)) for k, v in atenuacion_por_clase.items()}

# Calcular medias y desviaciones estándar por clase
estadisticas_por_clase = {
    int(clase): {
        "media": float(np.mean(valores)),
        "std": float(np.std(valores)),
        "n": int(len(valores))
    }
    for clase, valores in atenuacion_por_clase.items()
}

# Guardar en un archivo JSON
with open("atenuacion_por_clase.json", "w") as f:
    json.dump(estadisticas_por_clase, f, indent=2)


# Visualización opcional
plt.figure(figsize=(12, 6))
for clase, valores in sorted(atenuacion_por_clase.items()):
    plt.hist(valores, bins=50, alpha=0.5, label=f"Clase {clase:02d}")
plt.title("Distribución de atenuación por clase")
plt.xlabel("Atenuación (-log(remission / I0) / distancia)")
plt.ylabel("Frecuencia")
plt.legend()
plt.tight_layout()
plt.show()
