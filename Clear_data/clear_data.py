import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Caminhos corretos para seus arquivos
BASE_DIR = "users//victormilani//data_udacity"
CSV_ORIGINAL = os.path.join(BASE_DIR, "driving_log.csv")
IMG_FOLDER = os.path.join(BASE_DIR, "IMG")
CSV_LIMPO = os.path.join(BASE_DIR, "driving_log_cleaned.csv")

# Carregar o CSV original
colunas = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
df = pd.read_csv(CSV_ORIGINAL, names=colunas)

# Limpar espaços nos caminhos
for col in ['center', 'left', 'right']:
    df[col] = df[col].str.strip()

# Corrigir caminhos das imagens
for col in ['center', 'left', 'right']:
    df[col] = df[col].apply(lambda x: os.path.join(IMG_FOLDER, os.path.basename(x)))

# Verifica se a imagem existe e está OK
def imagem_valida(caminho):
    return os.path.exists(caminho) and cv2.imread(caminho) is not None

# Remover imagens inválidas
df = df[df['center'].apply(imagem_valida)]

# Remover linhas com velocidade quase zero
df = df[df['speed'].astype(float) > 0.1]

# Remover duplicatas
df = df.drop_duplicates()

# Remover excesso de ângulos zerados
angulo_zero = df[df['steering'].astype(float) == 0.0]
nao_zero = df[df['steering'].astype(float) != 0.0]
df = pd.concat([nao_zero, angulo_zero.sample(frac=0.15)])  # Mantém 15% das retas

# Embaralhar as linhas
df = df.sample(frac=1).reset_index(drop=True)

# Salvar novo CSV limpo
df.to_csv(CSV_LIMPO, index=False, header=False)
print(f"✅ Arquivo limpo salvo em: {CSV_LIMPO}")
print(f"🔢 Linhas restantes após limpeza: {len(df)}")

# Comparar distribuição dos dados antes e depois
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
original_df = pd.read_csv(CSV_ORIGINAL, names=colunas)
plt.hist(original_df['steering'].astype(float), bins=50, color='orange', alpha=0.6)
plt.title("Antes da Limpeza")
plt.xlabel("Ângulo de Direção")
plt.ylabel("Frequência")

plt.subplot(1, 2, 2)
plt.hist(df['steering'].astype(float), bins=50, color='green', alpha=0.6)
plt.title("Depois da Limpeza")
plt.xlabel("Ângulo de Direção")

plt.tight_layout()
plt.show()