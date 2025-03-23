import pandas as pd
import matplotlib.pyplot as plt

def plot_training_log(csv_path='training_log.csv'):
    # Carrega o CSV de log de treino
    df = pd.read_csv(csv_path)

    # Verifica se as colunas necessárias existem
    required_columns = {'epoch', 'loss', 'val_loss', 'learning_rate'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Faltam colunas no CSV: {required_columns - set(df.columns)}")

    # Plota o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['loss'], label='Loss de Treinamento')
    plt.plot(df['epoch'], df['val_loss'], label='Loss de Validação')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução da Loss ao Longo do Treinamento')
    plt.legend()
    plt.grid(True)
    plt.show()

# Executar
plot_training_log()