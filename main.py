import pandas as pd
import numpy as np
import random
import itertools
from collections import Counter
import statsmodels.api as sm

# Leitura do arquivo .xlsx em vez de .csv
data = pd.read_excel("Mega-Sena.xlsx")

def predict_arima(data, num_periods=60):
    number_counts = data.iloc[:, 2:8].apply(pd.Series.value_counts, axis=0).fillna(0).sum(axis=1).sort_index()
    number_counts.index = range(1, 61)
    model = sm.tsa.ARIMA(number_counts, order=(2, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=num_periods)

    return forecast

# Função atualizada para ler arquivos .xlsx
def load_draws_from_xlsx(file_path):
    draws = []
    data = pd.read_excel(file_path)
    for _index, row in data.iterrows():
        nums = [int(row[i]) for i in range(2, 8)]
        draws.append(nums)

    return draws

# Utilizando a nova função para ler arquivos .xlsx
original_draws = load_draws_from_xlsx('Mega-Sena.xlsx')

predicted_counts = predict_arima(data)
predicted_counts.index = range(1, len(predicted_counts) + 1)
probabilities = predicted_counts / predicted_counts.sum()

# Função atualizada para suportar diferentes tamanhos de jogo
def generate_arima_game(probabilities, original_draws, num_games=3, game_size=6):
    games = []
    while len(games) < num_games:
        game = sorted(np.random.choice(range(1, 61), size=game_size, replace=False, p=probabilities))
        if game[:6] not in original_draws:
            games.append(game)
    return games

# Gerar jogos com 6 números
arima_games = generate_arima_game(probabilities, original_draws)
print("3 jogos gerados com base nas previsões do ARIMA e que não estão nos sorteios originais:")
for game in arima_games:
    print(game)

# Gerar jogos com 7 números
arima_games_7 = generate_arima_game(probabilities, original_draws, game_size=7)
print("\n3 jogos de 7 números gerados com base nas previsões do ARIMA e que não estão nos sorteios originais:")
for game in arima_games_7:
    print(game)
