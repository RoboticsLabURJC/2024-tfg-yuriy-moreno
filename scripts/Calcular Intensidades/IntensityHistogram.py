# histograma_atenuacion.py
import os
import numpy as np
import json
from glob import glob
from collections import defaultdict
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# --- Configuracion ---
lidar_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/lidar/train"
label_root = "/home/yuriy/Descargas/GOOSE/goose_3d_train/labels/train"
escenarios_validos = {
    "2022-07-22_flight",
    # "2022-07-27_hoehenkirchner_forst",
    # "2022-08-18_putzbrunn_feldwege",
    # "2022-08-30_siegersbrunn_feldwege",
    # "2022-09-14_garching_uebungsplatz",
    # "2022-09-21_garching_uebungsplatz_2",
    # "2022-10-14_hohenbrunn_feldwege_waldwege",
    # "2022-11-11_aying",
    #"2023-03-02_garching",
    #"2023-03-03_garching_2",
    # "2023-04-05_flight_with_tiguan",
    #"2023-05-17_neubiberg_sunny",
}

bin_edges = np.arange(0, 200 + 10, 10)

#bin_edges = np.digitize(distancias[mask_valid], bins) - 1

histograma = defaultdict(lambda: defaultdict(list))
conteo_puntos = defaultdict(lambda: defaultdict(int))
I0 = 255.0

# --- Procesamiento ---
for escenario in sorted(escenarios_validos):
    print(f"Procesando escenario: {escenario}")
    carpeta_lidar = os.path.join(lidar_root, escenario)
    carpeta_label = os.path.join(label_root, escenario)
    bin_files = sorted(glob(os.path.join(carpeta_lidar, "*.bin")))

    for idx, bin_path in enumerate(bin_files):
        print(f"[{idx + 1}/{len(bin_files)}] Procesando archivo: {os.path.basename(bin_path)}")
        base = os.path.basename(bin_path).replace("_vls128.bin", "")
        label_path = os.path.join(carpeta_label, base + "_goose.label")
        if not os.path.exists(label_path):
            continue

        try:
            #scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
            scan = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
            puntos = scan[:, 0:3]
            remission = scan[:, 3]
            labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        except:
            continue

        distancias = np.linalg.norm(puntos, axis=1)
        mask_valid = (remission > 0.01) & (distancias > 0.1)

        remission = remission[mask_valid]
        distancias = distancias[mask_valid]
        labels = labels[mask_valid]

        for clase in np.unique(labels):
            clase_mask = labels == clase
            for i in range(len(bin_edges) - 1):
                in_bin = (distancias > bin_edges[i]) & (distancias <= bin_edges[i+1]) & clase_mask
                r_vals = remission[in_bin].tolist()
                histograma[int(clase)][f"{bin_edges[i]}-{bin_edges[i+1]}"] += r_vals
                conteo_puntos[int(clase)][f"{bin_edges[i]}-{bin_edges[i+1]}"] += len(r_vals)

# --- Media por bin ---
tabla_final = {
    clase: {
        rango: float(np.mean(valores)) if valores else 0.0
        for rango, valores in bins.items()
    }
    for clase, bins in histograma.items()
}

with open("intensidad_histograma_clase.json", "w") as f:
    json.dump(tabla_final, f, indent=2)

with open("intensidad_histograma_conteo.json", "w") as f:
    json.dump(conteo_puntos, f, indent=2)

print(f"✅ Histograma generado para {len(tabla_final)} clases.")


clases_mostrar = [0, 4, 6, 12, 17, 23, 38, 41]  # Por ejemplo

plt.figure(figsize=(12, 6))
for clase in clases_mostrar:
    bins = tabla_final[clase]  # <- quitar el str()
    dist_labels = list(bins.keys())
    medias = list(bins.values())
    plt.plot(dist_labels, medias, label=f"Clase {clase}")

plt.xticks(rotation=45)
plt.ylabel("Media de remisión")
plt.xlabel("Distancia (m)")
plt.title("Remisión media por clase según distancia")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig("remision_filtrada.png")
plt.show()

