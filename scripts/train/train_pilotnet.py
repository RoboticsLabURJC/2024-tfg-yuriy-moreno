# train_pilotnet.py
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import torchvision.transforms as T

from model_pilotnet import PilotNetVW

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="runs/pilotnet")

import pandas as pd
import numpy as np



# -------- Configuración --------
#DATA_ROOT = "/home/yuriy/Repositorios/Follow-Line-Combine-Dataset/"  # carpeta base donde están las imágenes y los csv
DATA_ROOT= "/home/yuriy/Universidad/2024-tfg-yuriy-moreno/dataset"
#CSV_PATHS = [
#    "adjustment_data.csv",  # pon aquí tu primer csv
#    "train.csv",  # y aquí el segundo
#]
CSV_PATHS = ["controlFinal2.csv"]

BATCH_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
MODEL_PATH = "carlaFinal2.pth"
RANDOM_SEED = 42
PATIENCE = 10    # si 5 epochs seguidas no mejora → parar
MIN_DELTA = 5e-4
# -------------------------------

df = pd.read_csv("/home/yuriy/Universidad/2024-tfg-yuriy-moreno/dataset/controlFinal2.csv")
# Bin de w
bins = np.linspace(-1, 1, 21)  # 20 bins    
df["bin"] = np.digitize(df["w"], bins)

# Objetivo total
TARGET_TOTAL = 40000

unique_bins = df["bin"].unique()
target_per_bin = TARGET_TOTAL // len(unique_bins)

balanced = []
#max_per_bin = 1500  # límite por curva/recta

# Calculamos el centro de cada bin
bin_centers = (bins[:-1] + bins[1:]) / 2

for b in unique_bins:
    group = df[df["bin"] == b]

    # centro de este bin (-1..1 aprox)
    center = bin_centers[b - 1]

    if len(group) == 0:
        continue
    # Si hay pocas → upsample (duplicar con replacement)
    if len(group) < target_per_bin:
        group = group.sample(target_per_bin, replace=True, random_state=42)
    # Si hay muchas → downsample
    else:
        #group = group.sample(max_per_bin, random_state=42)
        group = group.sample(target_per_bin, replace=False, random_state=42)
    balanced.append(group)

df_balanced = pd.concat(balanced).sample(frac=1).reset_index(drop=True)
df_balanced.to_csv("/home/yuriy/Universidad/2024-tfg-yuriy-moreno/dataset/control_balancedFinal2.csv", index=False)

print("Original:", len(df), "Balanced:", len(df_balanced))
CSV_PATHS = ["control_balancedFinal2.csv"]

class CarDataset(Dataset):
    def __init__(self, csv_paths, root_dir="", transform=None):
        """
        csv_paths: lista de rutas de csv
        root_dir: carpeta raíz para las imágenes
        """
        dfs = [pd.read_csv(os.path.join(root_dir, p)) for p in csv_paths]
        self.data = pd.concat(dfs, ignore_index=True)
        self.root_dir = root_dir
        self.transform = transform

        self.hflip = T.RandomHorizontalFlip(p=0.5)

        # Barajamos el dataframe para mezclar circuitos
        self.data = self.data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image"])
        image = Image.open(img_path).convert("RGB")

        v = float(row["v"])
        w = float(row["w"])

        flipped = False
        image, flipped = self.apply_hflip(image)

        if flipped:
            w = -w   # ← invertimos la dirección
        if self.transform:
            image = self.transform(image)

        target = torch.tensor([v, w], dtype=torch.float32)
        return image, target
    
    def apply_hflip(self, img):
        """
        Aplica flip horizontal aleatorio y retorna si se flippeó.
        """
        if random.random() < 0.5:
            return T.functional.hflip(img), True
        return img, False


def main():
    # Para reproducibilidad
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    transform = T.Compose([
        # Aquí podrías meter un recorte si quieres:
        # T.CenterCrop((crop_h, crop_w)),
        T.Resize((66, 200)),

        # ----- Augmentación -----
        #T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)], p=0.3),
        #T.RandomApply([T.GaussianBlur(3)], p=0.1),
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0))],p=0.2),   # Shift horizontal

        # -------------------------

        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [0,1] -> [-1,1]
    ])

    dataset = CarDataset(CSV_PATHS, root_dir=DATA_ROOT, transform=transform)

    print("Total imágenes en el dataset:", len(dataset))
    print("Primer elemento del dataset:")
    img0, y0 = dataset[0]
    print("  - img0 shape:", img0.shape)
    print("  - y0:", y0)

    # Split train/val
    n_total = len(dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    model = PilotNetVW(input_shape=(3, 66, 200)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

     # Scheduler: baja LR si la val_loss se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # ---- Entrenamiento ----
        model.train()
        running_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", ncols=100)


        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (batch, 2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            progress.set_postfix({"loss": loss.item()})

        train_loss = running_loss / n_train

        # ---- Validación ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)

        val_loss /= n_val

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Scheduler de LR
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
              f"- train_loss: {train_loss:.4f} "
              f"- val_loss: {val_loss:.4f}")
        
        # ---- Early stopping + guardar mejor modelo ----
        if best_val_loss - val_loss > MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Nuevo mejor modelo guardado (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  -> Sin mejora (paciencia {patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print("Early stopping activado.")
            break

    # Guardar pesos
    #torch.save(model.state_dict(), MODEL_PATH)

    # Cargar mejor modelo antes de exportar a ONNX
    print("Cargando mejor modelo desde disco para exportar a ONNX...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    

    # ------ Exportar a ONNX -------
    dummy_input = torch.randn(1, 3, 66, 200).to(device)  # batch=1, RGB, 66x200

    onnx_path = "carlaFinal2.onnx"
    torch.onnx.export(
        model,                      # tu modelo
        dummy_input,                # ejemplo de entrada
        onnx_path,                  # nombre de salida
        input_names=['input'],
        output_names=['output'],
        opset_version=11,           # compatible con la mayoría de runtimes
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Modelo exportado correctamente a {onnx_path}")
    print(f"Modelo guardado en {MODEL_PATH}")
    writer.close()

if __name__ == "__main__":
    main()
