import cv2

img_path = "/Users/victormilani/data_udacity/IMG/center_2025_03_16_23_20_11_553.jpg"
img = cv2.imread(img_path)

if img is None:
    print("Erro: OpenCV não conseguiu carregar a imagem.")
else:
    print("Imagem carregada com sucesso!")