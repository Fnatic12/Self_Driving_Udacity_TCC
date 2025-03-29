import pandas as pd
import matplotlib.pyplot as plt

# Substitua pelos caminhos reais dos seus arquivos CSV
steering2_log = pd.read_csv('training_log_steering2.csv')
train2_log = pd.read_csv('training_log_train2.csv')

# Plotando os valores de validação (val_loss)
plt.figure(figsize=(12, 6))
plt.plot(steering2_log['epoch'], steering2_log['val_loss'], label='Steering2 - val_loss')
plt.plot(train2_log['epoch'], train2_log['val_loss'], label='Train2 - val_loss')
plt.xlabel('Época')
plt.ylabel('Val Loss')
plt.title('Comparativo de Val Loss - Steering2 vs Train2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()