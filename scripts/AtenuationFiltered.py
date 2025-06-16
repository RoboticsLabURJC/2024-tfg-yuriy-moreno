import os
import numpy as np
import json
from glob import glob
from collections import defaultdict

escenarios_validos = {
    "2022-07-22_flight",
    "2022-07-27_hoehenkirchner_forst",
    "2022-08-18_putzbrunn_feldwege",
    "2022-08-30_siegersbrunn_feldwege",
    "2022-09-14_garching_uebungsplatz",
    "2022-09-21_garching_uebungsplatz_2",
    # "2022-10-12_sollninden_waldwege",         # Nublado
    "2022-10-14_hohenbrunn_feldwege_waldwege",
    # "2022-11-04_campus_rain",                 # Lluvia
    "2022-11-11_aying",                       
    # "2022-11-29_campus_rain_2",               #  Lluvia
    # "2022-12-07_aying_hills",                 #  Nieve
    # "2023-01-20_aying_mangfall_2",            #  Nieve
    # "2023-01-20_campus_snow",                 #  Nieve
    # "2023-02-13_touareg_neuperlach",          # Nublado
    # "2023-02-23_campus_roads",                # Nublado
    "2023-03-02_garching",
    "2023-03-03_garching_2",                  
    "2023-04-05_flight_with_tiguan",          
    # "2023-04-20_campus",                      # Nublado y restos de lluvia
    # "2023-05-15_neubiberg_rain",              #  Lluvia
    "2023-05-17_neubiberg_sunny",
    # "2023-05-24_neubiberg_cloudy"             #  Nublado
}

# Raíces de datos LiDAR y etiquetas
lidar_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train"
label_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train"

acumulador_por_clase = defaultdict(list)
I0 = 255.0  # Remisión ideal


# --- Recorrer todos los escenarios válidos
for escenario in sorted(escenarios_validos):
    print(f"Procesando escenario: {escenario}")
    carpeta_lidar = os.path.join(lidar_root, escenario)
    carpeta_label = os.path.join(label_root, escenario)

    bin_files = sorted(glob(os.path.join(carpeta_lidar, "*.bin")))

    for idx, bin_path in enumerate(bin_files):
        print(f"[{idx + 1}/{len(bin_files)}] Procesando archivo: {os.path.basename(bin_path)}")
        # Extraer base y construir ruta del .label correspondiente
        rel_path = os.path.relpath(bin_path, lidar_root)
        base = os.path.basename(bin_path).replace("_vls128.bin", "")
        label_path = os.path.join(label_root, os.path.dirname(rel_path), base + "_goose.label")

        if not os.path.exists(label_path):
            print(f"⚠️ Etiqueta no encontrada para {bin_path}")
            continue

        try:
            scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
            puntos = scan[:, :3]
            remission = scan[:, 3]
            labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF

        except Exception as e:
            print(f"❌ Error procesando {bin_path}: {e}")
            continue

        distancias = np.linalg.norm(puntos, axis=1)
        mask_valid = (remission > 0.01) & (distancias > 0.1)
        if np.count_nonzero(mask_valid) < 100:
            continue

        alphas = -np.log(remission[mask_valid] / I0) / distancias[mask_valid]
        valid_labels = labels[mask_valid]

        if remission[mask_valid].shape == distancias[mask_valid].shape == labels[mask_valid].shape:
            print(f"✅ {len(alphas)} puntos válidos en {os.path.basename(bin_path)}")
        
        # Filtrado robusto por percentiles
        p1, p99 = np.percentile(alphas, [1, 99])
        mask_range = (alphas > p1) & (alphas < p99)

        for alpha, label in zip(alphas[mask_range], valid_labels[mask_range]):
            acumulador_por_clase[int(label)].append(alpha)

# --- Estadísticas finales
estadisticas_por_clase = {
    clase: {
        "media": float(np.mean(valores)),
        "mediana": float(np.median(valores)),
        "std": float(np.std(valores)),
        "n": int(len(valores))
    }
    for clase, valores in sorted(acumulador_por_clase.items())
}

# Guardar
with open("atenuacion_global.json", "w") as f:
    json.dump(estadisticas_por_clase, f, indent=2)

print(f"✅ {len(estadisticas_por_clase)} clases guardadas en 'atenuacion_global.json'")
