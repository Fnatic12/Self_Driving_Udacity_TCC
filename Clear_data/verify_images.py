import os
import cv2

IMG_FOLDER = "/Users/victormilani/data_udacity/IMG"

imagens = [os.path.join(IMG_FOLDER, img) for img in os.listdir(IMG_FOLDER) if img.lower().endswith(('.jpg', '.png'))]

erros = []

for caminho in imagens:
    try:
        img = cv2.imread(caminho)
        if img is None:
            raise ValueError("Imagem vazia (None)")
    except Exception as e:
        print(f"Erro na imagem: {caminho} -> {e}")
        erros.append(caminho)

print(f"Total de imagens com erro: {len(erros)}")