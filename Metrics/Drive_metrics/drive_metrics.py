import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def plot_drive_metrics(csv_file):
    try:
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"O arquivo '{csv_file}' não foi encontrado.")

        log = pd.read_csv(csv_file)

        required_columns = {"timestamp", "steering_angle_pred", "steering_angle_real", 
                            "error", "speed", "response_time", "collision_flag"}
        missing_columns = required_columns - set(log.columns)
        if missing_columns:
            raise ValueError(f"Colunas ausentes no CSV: {missing_columns}")

        log["timestamp"] = pd.to_datetime(log["timestamp"])
        log["time"] = (log["timestamp"] - log["timestamp"].iloc[0]).dt.total_seconds()

        avg_speed = log["speed"].mean()
        log["delta_time"] = log["time"].diff().fillna(0)
        log["speed_mps"] = log["speed"] * 0.44704
        log["distance_delta"] = log["speed_mps"] * log["delta_time"]
        log["distance_accum"] = log["distance_delta"].cumsum()
        total_distance = log["distance_accum"].iloc[-1]
        log["acceleration"] = log["speed"].diff().fillna(0)

        mae = mean_absolute_error(log["steering_angle_real"], log["steering_angle_pred"])
        mse = mean_squared_error(log["steering_angle_real"], log["steering_angle_pred"])
        rmse = mse ** 0.5

        print(f"📏 Distância total percorrida: {total_distance:.2f} metros")
        print(f"🚀 Velocidade média: {avg_speed:.2f} MPH")
        print(f"📐 Erro Médio Absoluto (MAE): {mae:.4f}")
        print(f"📉 Erro Quadrático Médio (MSE): {mse:.4f}")
        print(f"📈 Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f}")

        output_dir = "graficos_drive"
        os.makedirs(output_dir, exist_ok=True)

        # Painel principal
        plt.figure(figsize=(16, 12))

        plt.subplot(3, 2, 1)
        plt.plot(log["time"], log["error"], label="Erro de Direção", color="red", alpha=0.7)
        plt.axhline(y=mae, color="orange", linestyle="--", label=f"MAE: {mae:.2f}")
        plt.axhline(y=rmse, color="purple", linestyle="--", label=f"RMSE: {rmse:.2f}")
        plt.title("Erro de Direção ao Longo do Tempo")
        plt.xlabel("Tempo (s)"); plt.ylabel("Erro (°)"); plt.legend(); plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(log["time"], log["speed"], color="blue", label="Velocidade")
        plt.title("Velocidade do Veículo")
        plt.xlabel("Tempo (s)"); plt.ylabel("Velocidade (MPH)"); plt.legend(); plt.grid(True)

        plt.subplot(3, 2, 3)
        plt.plot(log["time"], log["response_time"], color="green", label="Tempo de Resposta")
        plt.title("Tempo de Resposta do Modelo")
        plt.xlabel("Tempo (s)"); plt.ylabel("Tempo (s)"); plt.legend(); plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(log["time"], log["collision_flag"], "ko", label="Colisões")
        plt.title("Ocorrência de Colisões")
        plt.xlabel("Tempo (s)"); plt.ylabel("Colisão"); plt.legend(); plt.grid(True)

        plt.subplot(3, 2, 5)
        plt.plot(log["time"], log["distance_accum"], color="purple", label="Distância Acumulada")
        plt.title("Distância ao Longo do Tempo")
        plt.xlabel("Tempo (s)"); plt.ylabel("Distância (m)"); plt.legend(); plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(log["time"], log["speed"], alpha=0.6, label="Velocidade")
        plt.axhline(y=avg_speed, color="orange", linestyle="--", label=f"Média: {avg_speed:.2f}")
        plt.title("Velocidade com Linha de Média")
        plt.xlabel("Tempo (s)"); plt.ylabel("Velocidade (MPH)"); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "painel_principal.png"))
        plt.close()

        # Aceleração
        plt.figure(figsize=(10, 4))
        plt.plot(log["time"], log["acceleration"], color="brown", label="Aceleração")
        plt.title("Aceleração Estimada do Veículo")
        plt.xlabel("Tempo (s)"); plt.ylabel("Delta Velocidade (MPH)"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aceleracao.png"))
        plt.close()

        # Comparação ângulos
        plt.figure(figsize=(10, 4))
        plt.plot(log["time"], log["steering_angle_real"], label="Real", color="black", alpha=0.5)
        plt.plot(log["time"], log["steering_angle_pred"], label="Predito", color="blue", alpha=0.7)
        plt.title("Comparação de Ângulo Real vs Predito")
        plt.xlabel("Tempo (s)"); plt.ylabel("Ângulo de Direção (°)"); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparacao_angulos.png"))
        plt.close()

        print(f"✅ Gráficos salvos em: {output_dir}/")

    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗Uso: python drive_metrics.py <nome_do_csv>")
    else:
        plot_drive_metrics(sys.argv[1])