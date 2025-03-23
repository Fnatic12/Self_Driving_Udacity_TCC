import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_drive_metrics():
    """
    Plota as métricas do drive_metrics.csv, incluindo distância, velocidade,
    erro, tempo de resposta, colisões, MAE e MSE.
    """
    try:
        log = pd.read_csv("drive_metrics.csv")

        required_columns = {"timestamp", "steering_angle_pred", "steering_angle_real", 
                            "error", "speed", "response_time", "collision_flag"}
        missing_columns = required_columns - set(log.columns)
        if missing_columns:
            raise ValueError(f"As colunas ausentes no CSV: {missing_columns}")

        log["timestamp"] = pd.to_datetime(log["timestamp"])
        log["time"] = (log["timestamp"] - log["timestamp"].iloc[0]).dt.total_seconds()

        # Cálculo das métricas
        avg_speed = log["speed"].mean()
        log["delta_time"] = log["time"].diff().fillna(0)
        log["speed_mps"] = log["speed"] * 0.44704
        log["distance_delta"] = log["speed_mps"] * log["delta_time"]
        log["distance_accum"] = log["distance_delta"].cumsum()
        total_distance = log["distance_accum"].iloc[-1]

        # MAE e MSE
        mae = mean_absolute_error(log["steering_angle_real"], log["steering_angle_pred"])
        mse = mean_squared_error(log["steering_angle_real"], log["steering_angle_pred"])

        # Imprimir as métricas
        print(f"📏 Distância total percorrida: {total_distance:.2f} metros")
        print(f"🚀 Velocidade média: {avg_speed:.2f} MPH")
        print(f"📐 Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"📉 Erro Quadrático Médio (MSE): {mse:.4f}")

        # Gráficos
        plt.figure(figsize=(16, 12))

        # Erro de Direção
        plt.subplot(3, 2, 1)
        plt.plot(log["time"], log["error"], label="Erro de Direção", color="red", alpha=0.7)
        plt.axhline(y=mae, color="orange", linestyle="--", label=f"MAE: {mae:.2f}")
        plt.axhline(y=(mse**0.5), color="purple", linestyle="--", label=f"RMSE: {mse**0.5:.2f}")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Erro (°)")
        plt.title("Erro de Direção ao Longo do Tempo")
        plt.legend()
        plt.grid(True)

        # Velocidade
        plt.subplot(3, 2, 2)
        plt.plot(log["time"], log["speed"], label="Velocidade", color="blue")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Velocidade (MPH)")
        plt.title("Velocidade do Veículo")
        plt.legend()
        plt.grid(True)

        # Tempo de Resposta
        plt.subplot(3, 2, 3)
        plt.plot(log["time"], log["response_time"], label="Tempo de Resposta", color="green")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Tempo de Resposta (s)")
        plt.title("Tempo de Resposta do Modelo")
        plt.legend()
        plt.grid(True)

        # Colisões
        plt.subplot(3, 2, 4)
        plt.plot(log["time"], log["collision_flag"], label="Colisões", linestyle="None", marker="o", color="black")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Colisão (1 = Sim, 0 = Não)")
        plt.title("Ocorrência de Colisões")
        plt.legend()
        plt.grid(True)

        # Distância acumulada
        plt.subplot(3, 2, 5)
        plt.plot(log["time"], log["distance_accum"], label="Distância Acumulada (m)", color="purple")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Distância (m)")
        plt.title("Distância Percorrida ao Longo do Tempo")
        plt.legend()
        plt.grid(True)

        # Velocidade com linha média
        plt.subplot(3, 2, 6)
        plt.plot(log["time"], log["speed"], color="blue", alpha=0.6)
        plt.axhline(y=avg_speed, color="orange", linestyle="--", label=f"Média: {avg_speed:.2f} MPH")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Velocidade (MPH)")
        plt.title("Velocidade com Linha de Média")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Erro: O arquivo 'drive_metrics.csv' não foi encontrado.")
    except ValueError as e:
        print(f"Erro nos dados: {e}")

# Executar
plot_drive_metrics()