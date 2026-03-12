# BRForecast — Parâmetros de configuração

import os

# === Caminhos ===
DB_PATH = r"E:/Claude Code/Scrape Footystats/footystats.db"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# === ELO ===
HFA = 105  # Home Field Advantage calibrado via grid search (Brier Score, Série A 2021-2026)
ELO_SEASON_REGRESSION = 0.15  # Regressao a media entre temporadas (0=sem regressao)

# Rating inicial por divisão
RATING_INICIAL = {
    "A": 1500,
    "B": 1350,
    "C": 1200,
}

# Peso de K por divisão — Série A vale mais, C vale menos
K_BASE_DIVISION = {
    "A": 1.0,    # peso cheio
    "B": 0.7,    # 70% do K
    "C": 0.4,    # 40% do K
}

# K progressivo por posição na janela (temporadas mais recentes pesam mais)
# Janela: 2021-2026 (6 temporadas = 5 completas + 2026)
ELO_WINDOW_START = 2021
K_PROGRESSIVE = {
    2021: 12,
    2022: 14,
    2023: 18,
    2024: 22,
    2025: 26,
    2026: 30,
}

# === Temporadas ===
SERIE_A_IDS = {
    2013: 582, 2014: 581, 2015: 580, 2016: 101, 2017: 100,
    2018: 1198, 2019: 1936, 2020: 3817, 2021: 5713, 2022: 7097,
    2023: 9035, 2024: 11321, 2025: 14231, 2026: 16544,
}
SERIE_B_IDS = {
    2021: 5721, 2022: 7107, 2023: 9042, 2024: 11351, 2025: 14305, 2026: 16783,
}
SERIE_C_IDS = {
    2021: 5727, 2022: 7109, 2023: 9087, 2024: 11365, 2025: 14383,
}
LEAGUE_NAME = "Brazil Serie A"
TARGET_YEAR = 2026

# === Zonas do Brasileirão ===
ZONES = {
    "libertadores": list(range(1, 7)),       # Posições 1-6
    "sulamericana": list(range(7, 13)),      # Posições 7-12
    "rebaixamento": list(range(17, 21)),     # Posições 17-20
}

# === Modelo de empates ===
# Calibrado na Fase 2: taxa cai de ~32% (ELO próximo) para ~16% (ELO distante)
DRAW_MODEL = "variable"  # "variable" ou "constant"
DRAW_BASE_RATE = 0.274   # Taxa histórica Série A 2021-2026

# === Poisson ===
USE_XG = True              # Usar xG (True) ou gols reais (False) para calcular lambdas
POISSON_MAX_GOALS = 7      # Máximo de gols na matriz de probabilidades
ELO_LAMBDA_WEIGHT = 0.30   # Peso do ELO no ajuste dos lambdas
BLEND_ALPHA = 0.34         # Peso do modelo no blend com odds (0=odds puras, 1=modelo puro)
DIXON_COLES_RHO = -0.17    # Correcao Dixon-Coles (calibrado via grid search Serie A 2025; 0=Poisson puro)
MIN_MATCHES_SEASON = 60    # Mínimo de jogos na temporada alvo; se menos, inclui temporada anterior
EMA_ALPHA = 0.15           # Peso EMA para forcas ofensivas/defensivas (0=media simples)

# === Adjusted Goals ===
ADJUSTED_GOALS_WEIGHT = 0.30  # Blend: (1-w)*xG + w*adj_goals. 0=desligado

# === Match Importance ===
IMPORTANCE_N_SIMS = 500        # Mini-simulacoes para calcular importance (rapido)
IMPORTANCE_LAMBDA_BOOST = 0.05 # Boost maximo nos lambdas para jogos decisivos
IMPORTANCE_ELO_K_BOOST = 0.30  # Boost maximo no K do ELO para jogos decisivos (2026+)
IMPORTANCE_ELO_START_YEAR = 2026  # Ano a partir do qual importance afeta o K do ELO

# === Simulação ===
N_SIMULATIONS = 20000
N_SIMULATIONS_PROD = 20000
HOT_UPDATE = True  # Atualiza ELO + recalcula lambdas a cada jogo simulado
K_SIMULATION = 30  # K fixo para atualização de ELO durante simulação (Série A)
