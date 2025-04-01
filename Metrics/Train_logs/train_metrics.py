import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_log(csv_path='training_log_mobilenetv2.csv', save_dir='plots'):
    # Cria diretório para salvar os gráficos, se não existir
    os.makedirs(save_dir, exist_ok=True)

    # Carrega o CSV
    df = pd.read_csv(csv_path)

    # Verifica colunas disponíveis
    available_columns = set(df.columns)

    # Função auxiliar para salvar cada gráfico
    def save_plot(name):
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)

    # Gráfico 1: Loss total
    if {'epoch', 'loss', 'val_loss'}.issubset(available_columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df['loss'], label='Train Loss')
        plt.plot(df['val_loss'], label='Val Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss Total')
        plt.title('Loss Total por Época')
        plt.legend()
        plt.grid(True)
        save_plot('loss_total')
        plt.show()

    # Gráfico 4: Learning Rate
    if 'learning_rate' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df['learning_rate'], label='Learning Rate', color='purple')
        plt.xlabel('Época')
        plt.ylabel('Taxa de Aprendizado')
        plt.title('Taxa de Aprendizado por Época')
        plt.grid(True)
        save_plot('learning_rate')
        plt.show()

    print(f"✅ Gráficos salvos em: {os.path.abspath(save_dir)}")

# Executar
plot_training_log()