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

# Colunas do CSV original
colunas = ['img_center', 'img_left', 'img_right', 'ang_left', 'ang_right', 'brake', 'speed']
df = pd.read_csv(CSV_ORIGINAL, names=colunas)

# Corrigir e limpar caminhos
for col in ['img_center', 'img_left', 'img_right']:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].apply(lambda x: os.path.join(IMG_FOLDER, os.path.basename(x)))

# Calcular coluna média de direção
df['steering'] = (df['ang_left'].astype(float) + df['ang_right'].astype(float)) / 2
df['brake'] = df['brake'].astype(float)
df['speed'] = df['speed'].astype(float)

# Validar se imagem existe e é carregável
def imagem_valida(caminho):
    return os.path.exists(caminho) and cv2.imread(caminho) is not None

df = df[df['img_center'].apply(imagem_valida)]

# Remover velocidade baixa
df = df[df['speed'] > 0.1]

# Remover duplicatas
df = df.drop_duplicates()

# Reduzir excessos de steering = 0 (se houver muitos)
zeros = df[df['steering'] == 0.0]
nao_zeros = df[df['steering'] != 0.0]

if len(zeros) > 15000:
    zeros = zeros.sample(n=10000, random_state=42)  # Mantém só 10k
    df = pd.concat([nao_zeros, zeros])
    print(f"⚠️ Steering 0.0 reduzido para 10.000 amostras (era {len(df)})")
else:
    df = pd.concat([nao_zeros, zeros])

# Embaralhar
df = df.sample(frac=1).reset_index(drop=True)

# Salvar CSV limpo (sem throttle)
df[['img_center', 'img_left', 'img_right', 'ang_left', 'ang_right', 'brake', 'speed']].to_csv(CSV_LIMPO, index=False, header=False)
print(f"✅ Arquivo limpo salvo em: {CSV_LIMPO}")
print(f"🔢 Linhas restantes após limpeza: {len(df)}")

# Gráfico antes e depois
plt.figure(figsize=(10, 4))

# Antes da limpeza
plt.subplot(1, 2, 1)
original_df = pd.read_csv(CSV_ORIGINAL, names=colunas)
original_df['steering'] = (original_df['ang_left'].astype(float) + original_df['ang_right'].astype(float)) / 2
plt.hist(original_df['steering'], bins=50, color='orange', alpha=0.6)
plt.title("Antes da Limpeza")
plt.xlabel("Ângulo de Direção")
plt.ylabel("Frequência")

# Depois da limpeza
plt.subplot(1, 2, 2)
plt.hist(df['steering'], bins=50, color='green', alpha=0.6)
plt.title("Depois da Limpeza")
plt.xlabel("Ângulo de Direção")

plt.tight_layout()
plt.show()