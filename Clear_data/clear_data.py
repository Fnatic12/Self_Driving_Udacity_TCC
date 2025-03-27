import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Caminhos
BASE_DIR = "/Users/victormilani/data_udacity"
CSV_ORIGINAL = os.path.join(BASE_DIR, "driving_log.csv")
IMG_FOLDER = os.path.join(BASE_DIR, "IMG")
CSV_LIMPO = os.path.join(BASE_DIR, "driving_log_cleaned.csv")

# Colunas
colunas = ['img_center', 'img_left', 'img_right', 'ang_left', 'ang_right', 'throttle', 'speed']
df = pd.read_csv(CSV_ORIGINAL, names=colunas)

# Corrigir e limpar caminhos
for col in ['img_center', 'img_left', 'img_right']:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].apply(lambda x: os.path.join(IMG_FOLDER, os.path.basename(x)))

# Calcular coluna média de direção
df['steering'] = (df['ang_left'].astype(float) + df['ang_right'].astype(float)) / 2
df['throttle'] = df['throttle'].astype(float)
df['speed'] = df['speed'].astype(float)

# Validar se imagem existe e é carregável
def imagem_valida(caminho):
    return os.path.exists(caminho) and cv2.imread(caminho) is not None

df = df[df['img_center'].apply(imagem_valida)]

# Remover velocidade ~0
df = df[df['speed'] > 0.1]

# Remover duplicatas
df = df.drop_duplicates()

# Balancear bins do steering (melhor do que dropar ângulos 0)
hist, bins = np.histogram(df['steering'], bins=100)
max_por_bin = 300
indices_para_manter = []

for i in range(len(bins)-1):
    bin_indices = df[(df['steering'] >= bins[i]) & (df['steering'] < bins[i+1])].index
    if len(bin_indices) > max_por_bin:
        bin_indices = np.random.choice(bin_indices, max_por_bin, replace=False)
    indices_para_manter.extend(bin_indices)

df = df.loc[indices_para_manter]
df = df.sample(frac=1).reset_index(drop=True)

# Salvar CSV limpo
df[['img_center', 'img_left', 'img_right', 'ang_left', 'ang_right', 'throttle', 'speed']].to_csv(CSV_LIMPO, index=False, header=False)
print(f"✅ Arquivo limpo salvo em: {CSV_LIMPO}")
print(f"🔢 Linhas restantes após limpeza: {len(df)}")

# Histograma comparativo
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
original_df = pd.read_csv(CSV_ORIGINAL, names=colunas)
plt.hist(original_df['steering'].astype(float), bins=50, color='orange', alpha=0.6)
plt.title("Antes da Limpeza")
plt.xlabel("Ângulo de Direção")
plt.ylabel("Frequência")

plt.subplot(1, 2, 2)
plt.hist(df['steering'], bins=50, color='green', alpha=0.6)
plt.title("Depois da Limpeza")
plt.xlabel("Ângulo de Direção")

plt.tight_layout()
plt.show()