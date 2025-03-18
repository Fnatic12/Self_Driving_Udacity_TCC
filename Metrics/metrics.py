import pandas as pd
import matplotlib.pyplot as plt

def plot_drive_metrics():
    """
    Plota as métricas do drive_metrics.csv para avaliar o desempenho do modelo.
    """
    try:
        # Carregar o arquivo de métricas
        log = pd.read_csv("drive_metrics.csv")

        # Verificar se todas as colunas essenciais estão presentes
        required_columns = {"timestamp", "steering_angle_pred", "steering_angle_real", 
                            "error", "speed", "response_time", "collision_flag"}
        available_columns = set(log.columns)

        missing_columns = required_columns - available_columns
        if missing_columns:
            raise ValueError(f"As colunas ausentes no CSV: {missing_columns}")

        # Converter timestamp para um formato numérico para melhor plotagem
        log["timestamp"] = pd.to_datetime(log["timestamp"])
        log["time"] = (log["timestamp"] - log["timestamp"].iloc[0]).dt.total_seconds()

        # Criar os gráficos
        plt.figure(figsize=(14, 8))

        # Erro entre o ângulo predito e o real
        plt.subplot(2, 2, 1)
        plt.plot(log["time"], log["error"], label="Erro de Direção", color="red")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Erro (°)")
        plt.title("Erro de Direção ao Longo do Tempo")
        plt.legend()
        plt.grid(True)

        # Velocidade do veículo
        plt.subplot(2, 2, 2)
        plt.plot(log["time"], log["speed"], label="Velocidade", color="blue")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Velocidade (MPH)")
        plt.title("Velocidade do Veículo")
        plt.legend()
        plt.grid(True)

        # Tempo de resposta do modelo
        plt.subplot(2, 2, 3)
        plt.plot(log["time"], log["response_time"], label="Tempo de Resposta", color="green")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Tempo de Resposta (s)")
        plt.title("Tempo de Resposta do Modelo")
        plt.legend()
        plt.grid(True)

        # Sinal de colisão
        plt.subplot(2, 2, 4)
        plt.plot(log["time"], log["collision_flag"], label="Colisões", linestyle="None", marker="o", color="black")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Colisão (1 = Sim, 0 = Não)")
        plt.title("Ocorrência de Colisões")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Erro: O arquivo 'drive_metrics.csv' não foi encontrado.")
    except ValueError as e:
        print(f"Erro nos dados: {e}")

plot_drive_metrics()