# BRForecast — Plano de Implementação

## Visão geral

Sistema de previsão do Campeonato Brasileiro Série A 2026 que combina ratings ELO, modelo Poisson alimentado por xG e simulação Monte Carlo para calcular probabilidades de título, classificação para Libertadores/Sul-Americana e rebaixamento. Inclui análise individual de jogos (merecimento via xG), comparação com odds de casas de apostas e cenários interativos "What-if". Tudo apresentado em dashboard Streamlit com gráficos Plotly.

**Fonte de dados:** `footystats.db` (2GB, SQLite) — já coletado via scraper FootyStats, contendo 13k+ partidas brasileiras com xG, odds (68 colunas), estatísticas avançadas e classificações.

## Estrutura de pastas

```
BRForecast/
├── data/
│   └── (CSVs gerados pelo pipeline)
│       ├── elo_ratings_2026.csv
│       ├── tabela_atual_2026.csv
│       ├── jogos_restantes_2026.csv
│       ├── team_stats_2026.csv
│       └── simulation_results.csv
├── src/
│   ├── config.py              # Parâmetros: K, HFA, N_SIMS, DB_PATH, zonas
│   ├── elo.py                 # Cálculo de ELO
│   ├── poisson.py             # Lambdas + simulação de placares
│   ├── load_data.py           # Carrega dados do SQLite
│   ├── simulation.py          # Monte Carlo
│   ├── table.py               # Atualização de classificação
│   └── match_analysis.py      # Análise individual (xG, merecimento)
├── viz/
│   ├── exploration.ipynb      # Fase 0: exploração dos dados
│   └── odds_comparison.ipynb  # Fase 6: análise de odds
├── app.py                     # Dashboard Streamlit
├── data_processor.py          # Prepara dados para visualização
├── run.py                     # Ponto de entrada: roda pipeline completo
├── requirements.txt
└── PLANO_IMPLEMENTACAO.md
```

---

## Fase 0: Exploração e entendimento dos dados

### O que será feito

Criar notebook Jupyter que investiga sistematicamente o `footystats.db` para mapear todos os campos necessários ao projeto. Cada seção do notebook responde a perguntas específicas sobre os dados, documentando nomes exatos de colunas, cobertura temporal e estatísticas-base.

### Arquivos envolvidos

- `viz/exploration.ipynb` — notebook de exploração com todas as queries e respostas

### Implementação

**Seção 1: Inventário de temporadas**
- Query em `leagues` filtrando `country = 'Brazil'` e nome contendo "Série A" (ou variante)
- Listar: anos disponíveis, IDs de temporada, gaps
- Mapear IDs de temporada da Série A para uso nas fases seguintes

**Seção 2: Anatomia de uma partida (tabela `matches`)**
- Selecionar 5 partidas de uma temporada da Série A
- Documentar campos exatos de:
  - Gols: `homeGoalCount`, `awayGoalCount` (ou variantes)
  - Data: formato (Unix timestamp vs string), campo exato
  - IDs dos times: `homeID`, `awayID` — verificar join com `teams`
  - Status: campo que distingue jogos realizados vs futuros
  - Game week / rodada: campo exato

**Seção 3: Mapeamento de xG**
- Buscar colunas com "xg" ou "expected" em `matches` e `match_details`
- Determinar: xG por partida por time? Formato? Cobertura por temporada?
- Calcular % de partidas com xG preenchido, por temporada
- Verificar se `team_details` tem xG médio por temporada

**Seção 4: Mapa de odds**
- Identificar colunas de odds: `odds_ft_1`, `odds_ft_x`, `odds_ft_2` (ou variantes)
- Verificar formato (decimal?) e cobertura temporal
- Verificar odds de Over/Under 2.5

**Seção 5: Classificação (tabela `league_tables`)**
- Estrutura: campos disponíveis (pontos, vitórias, saldo, etc.)
- Cobertura: existe para 2026?
- Tipo: classificação final ou rodada a rodada?

**Seção 6: Jogos futuros da temporada 2026**
- Como identificar jogos não realizados (status, gols NULL, data futura)
- Quantos jogos realizados vs futuros na Série A 2026

**Seção 7: Estatísticas casa/fora (base para Fases 2 e 3)**
- Filtrar Série A das últimas 5+ temporadas
- Calcular: % mandante vence, % empate, % visitante vence
- Média de gols: mandante, visitante, total por jogo

### Validação automática (Claude Code checa)
- [ ] Notebook roda sem erro célula a célula
- [ ] Todas as 7 seções produzem output
- [ ] Campos exatos de gols, xG, odds e status estão documentados nas saídas

### Validação manual (você checa)
- [ ] Os campos mapeados fazem sentido (nomes batem com o que conhece do FootyStats)
- [ ] A cobertura de xG é suficiente para usar no modelo (>80% das partidas nas últimas 3+ temporadas)
- [ ] As estatísticas casa/fora estão dentro do esperado (~47% mandante, ~27% empate, ~26% visitante no Brasileirão)
- [ ] A temporada 2026 está no banco com jogos realizados + futuros (se o campeonato já começou)

### Pausa: aguardando confirmação para prosseguir para Fase 1

---

## Fase 1: Sistema ELO

### O que será feito

Implementar sistema ELO que processa todas as partidas históricas da Série A em ordem cronológica, calculando rating de força para cada time. Sem Home Field Advantage nesta fase (HFA vem na Fase 2). Exportar ratings e gráfico de evolução.

### Arquivos envolvidos

- `src/config.py` — parâmetros iniciais (K=20, RATING_INICIAL=1500, DB_PATH)
- `src/elo.py` — lógica de cálculo ELO completa
- `data/elo_ratings.csv` — ratings finais exportados

### Implementação

**`src/config.py`:**
- `DB_PATH` apontando para `E:/Claude Code/Scrape Footystats/footystats.db`
- `K_FACTOR = 20` (moderado, padrão clubelo.com)
- `RATING_INICIAL = 1500`
- `HFA = 0` (placeholder, calibrado na Fase 2)
- Mapeamento de zonas do Brasileirão: posições 1-6 Libertadores, 7-12 Sul-Americana, 17-20 rebaixamento

**`src/elo.py`:**

1. `load_historical_matches()` — query ao SQLite: todas as partidas da Série A, ordenadas cronologicamente por data. Usa os nomes de campos descobertos na Fase 0

2. `expected_score(rating_a, rating_b)` — fórmula: `1 / (1 + 10^((Rb - Ra) / 400))`

3. `get_match_result(home_goals, away_goals)` — retorna (1.0, 0.0), (0.5, 0.5) ou (0.0, 1.0)

4. `update_rating(rating, score, expected, k)` — `rating + k * (score - expected)`

5. `calculate_all_elos(matches_df)`:
   - Itera partida por partida em ordem cronológica
   - Times novos recebem RATING_INICIAL
   - Calcula expectativa SEM HFA (nesta fase)
   - Atualiza ambos os times
   - Salva histórico: `(data, time, elo_antes, elo_depois, adversário, resultado)`
   - Retorna: `ratings_dict`, `history_df`

6. `export_ratings(ratings, filepath)` — salva CSV: `time, elo_rating, ultima_partida`

7. `plot_elo_evolution(history, teams)` — gráfico Matplotlib/Plotly com evolução de pelo menos 4 times representativos (ex: Flamengo, Palmeiras, Cruzeiro, time que caiu)

### Validação automática (Claude Code checa)
- [ ] `src/elo.py` roda sem erro com o banco de dados
- [ ] `data/elo_ratings.csv` é gerado com todos os times da última temporada
- [ ] Top 5 ELO inclui times historicamente fortes (Flamengo, Palmeiras, Atlético-MG, etc.)
- [ ] Bottom 5 ELO inclui times recém-promovidos ou historicamente fracos
- [ ] Diferença entre 1º e 20º está entre 150-400 pontos
- [ ] Gráfico de evolução é gerado sem erro

### Validação manual (você checa)
- [ ] O ranking intuitivamente faz sentido (times fortes no topo, fracos embaixo)
- [ ] A evolução de ELO dos 4 times no gráfico reflete momentos reais (ex: queda de um time que foi rebaixado)
- [ ] K=20 produz ratings nem muito voláteis nem muito estáticos

### Pausa: aguardando confirmação para prosseguir para Fase 2

---

## Fase 2: Calibração de parâmetros

### O que será feito

Calibrar empiricamente o Home Field Advantage (HFA) e a taxa de empates usando dados históricos. O HFA ótimo é encontrado via grid search minimizando o Brier Score. A taxa de empates é analisada por faixa de diferença de ELO para decidir se usar constante ou variável.

### Arquivos envolvidos

- `src/elo.py` — atualizado para aceitar HFA como parâmetro
- `src/config.py` — atualizado com HFA ótimo e parâmetros de empate
- `viz/exploration.ipynb` — seção adicional com gráficos de calibração (ou nova seção no notebook)

### Implementação

**Calibração de HFA:**

1. Modificar `expected_score()` para aceitar HFA: `expected_score(rating_home + hfa, rating_away)`

2. Função `calibrate_hfa(matches_df, hfa_range=range(0, 130, 10))`:
   - Para cada valor de HFA no range:
     - Roda o cálculo de ELO com esse HFA
     - Para cada partida, calcula P(home), P(draw), P(away)
     - Calcula Brier Score: `mean((P_predicted - resultado_real)²)`
   - Retorna o HFA que minimiza o Brier Score
   - Gera gráfico: eixo X = HFA, eixo Y = Brier Score

3. Referência esperada: HFA entre 65-100 para o Brasileirão

**Análise de taxa de empates:**

1. Função `analyze_draw_rate(matches_df, history_df)`:
   - Para cada partida, calcula a diferença de ELO (|elo_home + hfa - elo_away|)
   - Agrupa em faixas: 0-50, 50-100, 100-150, 150-200, 200+
   - Calcula taxa de empates em cada faixa
   - Se taxa cai conforme diferença aumenta → modelo variável vale a pena

2. Se variável: `P_draw = taxa_base × (1 - abs(elo_diff) / 800)`
   Onde `taxa_base` = taxa histórica do Brasileirão (~27%)

3. Distribuição de probabilidades:
   - `P_home_win = E_home × (1 - P_draw)`
   - `P_away_win = (1 - E_home) × (1 - P_draw)`

**Atualização do config:**
- `HFA = <valor_otimo>`
- `DRAW_MODEL = "variable"` ou `"constant"`
- `DRAW_BASE_RATE = <taxa_historica>`

### Validação automática (Claude Code checa)
- [ ] Grid search roda sem erro e produz gráfico
- [ ] HFA ótimo está no range 40-120 (fora disso indica bug)
- [ ] Análise de empates por faixa de ELO gera tabela e gráfico
- [ ] `config.py` atualizado com os novos parâmetros

### Validação manual (você checa)
- [ ] O gráfico de calibração HFA tem formato de "U" (erro alto com HFA=0, cai, volta a subir)
- [ ] O HFA encontrado é plausível para o Brasileirão (65-100 esperado)
- [ ] A análise de empates mostra tendência clara: times próximos empatam mais?
- [ ] Decisão: taxa constante ou variável faz sentido com os dados?

### Pausa: aguardando confirmação para prosseguir para Fase 3

---

## Fase 3: Modelo Poisson + xG

### O que será feito

Implementar modelo Poisson que simula placares (não só resultados), alimentado por xG quando disponível. Calcular poder ofensivo e defensivo de cada time. Validar contra dados reais.

### Arquivos envolvidos

- `src/poisson.py` — cálculo de lambdas e simulação de placares
- `src/config.py` — parâmetros do Poisson (fonte de lambda: xG vs gols, janela temporal)

### Implementação

**Decisões baseadas na Fase 0:**
- Se cobertura de xG > 80% nas últimas 3 temporadas → usar xG
- Senão → usar gols reais
- Janela temporal: temporada inteira (estável) para começar

**`src/poisson.py`:**

1. `calculate_team_strengths(matches_df, use_xg=True)`:
   - Para cada time da temporada alvo:
     - `attack_strength` = média xG gerado / média xG da liga (ou gols marcados / média gols liga)
     - `defense_strength` = média xGA sofrido / média xGA da liga (ou gols sofridos / média gols liga)
   - Calcula médias da liga: gols mandante, gols visitante
   - Retorna DataFrame: `time, attack, defense, avg_xg, avg_xga`

2. `calculate_lambdas(home_team, away_team, team_strengths, league_avgs)`:
   ```
   lambda_home = attack_home × defense_away × avg_goals_home_league
   lambda_away = attack_away × defense_home × avg_goals_away_league
   ```
   - Retorna `(lambda_home, lambda_away)`

3. `simulate_score(lambda_home, lambda_away)`:
   - `gols_home = np.random.poisson(lambda_home)`
   - `gols_away = np.random.poisson(lambda_away)`
   - Retorna `(gols_home, gols_away)`

4. `score_probabilities(lambda_home, lambda_away, max_goals=7)`:
   - Matriz de probabilidades para cada placar possível (0x0 até 7x7)
   - Agrega: P(home_win), P(draw), P(away_win)
   - Retorna dicionário com probabilidades

5. `validate_poisson(matches_df, team_strengths)`:
   - Para jogos já realizados de uma temporada encerrada:
     - Calcula lambdas → probabilidades → Brier Score
   - Compara distribuição de gols simulada vs real (histograma)
   - Compara taxa de empates simulada vs real
   - Calcula Brier Score Poisson vs ELO simples

### Validação automática (Claude Code checa)
- [ ] `poisson.py` roda sem erro
- [ ] Poder ofensivo/defensivo calculado para todos os times da temporada alvo
- [ ] Simulação de 10.000 placares com lambda_home=1.5, lambda_away=1.0 produz:
  - Mandante vence ~47-52%
  - Empates ~23-28%
  - Média de gols ~2.3-2.7
- [ ] Brier Score do Poisson é igual ou melhor que o ELO simples

### Validação manual (você checa)
- [ ] Os times com maior attack_strength são os que esperaria (ofensivos na realidade)
- [ ] Os times com menor defense_strength são os que mais sofrem gols
- [ ] A distribuição de gols simulada é visualmente similar à real (sem excesso de 0x0 ou goleadas)
- [ ] O modelo com xG (se usado) é de fato melhor que com gols reais?

### Pausa: aguardando confirmação para prosseguir para Fase 4

---

## Fase 4: Dados da temporada 2026

### O que será feito

Carregar a foto atual do campeonato 2026: classificação, jogos realizados, jogos restantes. Fazer merge com ratings ELO e estatísticas Poisson para ter um DataFrame completo por time.

### Arquivos envolvidos

- `src/load_data.py` — funções de carga do SQLite
- `src/config.py` — ID da temporada 2026
- `data/tabela_atual_2026.csv` — classificação atual
- `data/jogos_restantes_2026.csv` — jogos futuros
- `data/team_stats_2026.csv` — ELO + poder ofensivo/defensivo por time

### Implementação

**`src/load_data.py`:**

1. `get_season_id(year=2026)`:
   - Query em `leagues` para encontrar o ID da Série A 2026
   - Retorna `season_id`

2. `load_current_table(season_id)`:
   - Se `league_tables` tem dados de 2026 → usar diretamente
   - Senão → construir a partir de `matches` (somar pontos, gols, etc.)
   - Retorna DataFrame: `time, jogos, vitorias, empates, derrotas, gols_pro, gols_contra, saldo, pontos`

3. `load_remaining_matches(season_id)`:
   - Query em `matches`: jogos da Série A 2026 que ainda não foram realizados
   - Critério: campo de status ou gols NULL ou data futura (definido na Fase 0)
   - Retorna DataFrame: `rodada, mandante, visitante, mandante_id, visitante_id`

4. `load_completed_matches(season_id)`:
   - Jogos já realizados da temporada 2026
   - Retorna com gols, xG, odds

5. `build_team_stats(season_id, elo_ratings, team_strengths)`:
   - Merge: tabela atual + ELO + poder ofensivo/defensivo Poisson
   - Retorna DataFrame: `time, jogos, pontos, elo, attack_xg, defense_xga, lambda_home_medio, lambda_away_medio`

6. Exportar os 3 CSVs para `data/`

**Tratamento de edge case:** Se a Série A 2026 ainda não começou no banco, o sistema deve ser funcional com a última temporada completa (2025 ou 2024) para backtest, e estar pronto para receber dados de 2026 assim que disponíveis.

### Validação automática (Claude Code checa)
- [ ] `load_data.py` conecta ao banco e executa queries sem erro
- [ ] Se há dados de 2026: 3 CSVs são gerados com dados reais
- [ ] Se não há dados de 2026: sistema usa última temporada completa como fallback
- [ ] O merge produz um DataFrame com todos os times e todas as colunas esperadas
- [ ] Nenhum time aparece com ELO ou stats faltando (NaN)

### Validação manual (você checa)
- [ ] A classificação atual bate com a classificação real (conferir em site de esportes)
- [ ] Os jogos restantes fazem sentido (rodadas futuras, times corretos)
- [ ] O merge ELO + stats está consistente (time com ELO alto tem bons stats e vice-versa)

### Pausa: aguardando confirmação para prosseguir para Fase 5

---

## Fase 5: Simulação Monte Carlo

### O que será feito

Simular o restante do campeonato 10.000 vezes usando o modelo Poisson para cada jogo. Agregar resultados em probabilidades de título, Libertadores, Sul-Americana e rebaixamento por time. Validar com backtest em temporada encerrada.

### Arquivos envolvidos

- `src/simulation.py` — lógica Monte Carlo
- `src/table.py` — atualização de classificação (pontos, gols, saldo, desempate)
- `run.py` — ponto de entrada do pipeline completo
- `data/simulation_results.csv` — resultados agregados

### Implementação

**`src/table.py`:**

1. `update_table(table_df, home_team, away_team, home_goals, away_goals)`:
   - Atualiza pontos (3/1/0), gols marcados/sofridos, saldo, vitórias/empates/derrotas
   - Retorna table_df atualizado

2. `apply_tiebreakers(table_df)`:
   - Ordena por: pontos → vitórias → saldo de gols → gols marcados
   - (Confronto direto omitido — complexo em simulação e raramente decisivo)

**`src/simulation.py`:**

1. `simulate_season(remaining_matches, current_table, team_strengths, league_avgs)`:
   - Copia tabela atual
   - Para cada jogo restante:
     - Calcula lambdas via Poisson
     - Sorteia placar
     - Atualiza tabela
   - Aplica desempate
   - Retorna classificação final (time → posição, pontos)

2. `run_monte_carlo(n_simulations, remaining_matches, current_table, team_strengths, league_avgs)`:
   - Loop de N simulações, cada uma chamando `simulate_season()`
   - Para cada simulação, registra posição e pontos de cada time
   - Retorna matriz: `n_sims × n_times × (posição, pontos)`

3. `aggregate_results(simulation_results, n_sims)`:
   - Para cada time:
     - `P(titulo)` = vezes em 1º / N
     - `P(libertadores)` = vezes em 1º-6º / N
     - `P(sulamericana)` = vezes em 7º-12º / N
     - `P(rebaixamento)` = vezes em 17º-20º / N
     - Pontos: média, min, max, percentil 10, percentil 90
     - Distribuição de posições: array de 20 probabilidades
   - Retorna DataFrame agregado

4. `backtest(season_id, start_round)`:
   - Carrega temporada encerrada (ex: 2024)
   - Simula a partir de rodada N como se os jogos restantes fossem desconhecidos
   - Compara probabilidades do modelo com resultado real
   - Roda para rodadas 10, 15, 20, 25, 30

**`run.py`:**
- Pipeline sequencial: config → ELO → Poisson → load_data → Monte Carlo → export
- Aceita argumentos: `--season`, `--n-sims`, `--backtest`

**Parâmetros iniciais:**
- `N_SIMULATIONS = 10000` (desenvolvimento)
- `N_SIMULATIONS_PROD = 20000` (produção)
- Sem hot update nos ratings durante simulação (simplicidade)

### Validação automática (Claude Code checa)
- [ ] `simulation.py` roda 1000 simulações sem erro em tempo razoável (<30s)
- [ ] Probabilidades de cada time somam ~100% para cada zona (com margem de arredondamento)
- [ ] P(titulo) do líder é a maior (ou entre as maiores)
- [ ] P(rebaixamento) dos últimos colocados é a maior
- [ ] Backtest: roda sem erro para pelo menos uma temporada encerrada
- [ ] `data/simulation_results.csv` é gerado corretamente

### Validação manual (você checa)
- [ ] As probabilidades de título fazem sentido intuitivo (time que lidera tem maior chance)
- [ ] O backtest mostra convergência: à medida que avança na temporada, o modelo acerta mais
- [ ] No backtest, o campeão real tinha probabilidade alta nas rodadas finais
- [ ] Os rebaixados reais tinham alta P(rebaixamento) no modelo
- [ ] Pontos esperados são plausíveis (campeão ~70-80 pts, rebaixado ~35-45 pts)

### Pausa: aguardando confirmação para prosseguir para Fase 6

---

## Fase 6: Análise de odds

### O que será feito

Converter odds do FootyStats para probabilidades implícitas e usá-las como benchmark do modelo. Comparar via Brier Score, Log-Loss e gráfico de calibração. Identificar divergências.

### Arquivos envolvidos

- `viz/odds_comparison.ipynb` — notebook com análise completa
- `src/poisson.py` — reutiliza funções de probabilidade

### Implementação

**No notebook `viz/odds_comparison.ipynb`:**

1. **Converter odds para probabilidades:**
   ```
   P_implied(x) = 1/odds(x)
   P_normalized(x) = P_implied(x) / sum(P_implied)  # remove overround
   ```

2. **Tabela comparativa por jogo (jogos já realizados):**
   - Colunas: partida, P(H) modelo, P(H) odds, P(D) modelo, P(D) odds, P(A) modelo, P(A) odds, resultado real

3. **Métricas globais:**
   - Brier Score: `mean((P_pred - resultado)²)` — calcular para modelo E para odds
   - Log-Loss: `mean(-log(P_resultado_correto))` — penaliza mais previsões confiantes erradas
   - Comparar modelo vs odds em ambas métricas

4. **Gráfico de calibração:**
   - Agrupar previsões em faixas (0-10%, 10-20%, ..., 90-100%)
   - Para cada faixa: % real de acertos vs % previsto
   - Diagonal = calibração perfeita. Desvios = modelo sobre/sub-confiante

5. **Mapa de divergências:**
   - Top 20 jogos onde |P_modelo - P_odds| é maior
   - Analisar: o modelo errou ou as odds erraram nesses casos?
   - Identificar padrões: o modelo diverge mais em jogos de times populares? Em casa?

### Validação automática (Claude Code checa)
- [ ] Notebook roda sem erro
- [ ] Brier Score calculado para modelo e odds (ambos entre 0.15-0.30, faixa típica de futebol)
- [ ] Gráfico de calibração gerado
- [ ] Tabela de divergências gerada com pelo menos 20 jogos

### Validação manual (você checa)
- [ ] O Brier Score do modelo está próximo ao das odds? (Se muito pior, pode indicar bug no modelo)
- [ ] O gráfico de calibração mostra curva razoável (sem desvios extremos da diagonal)
- [ ] As divergências identificadas fazem sentido (ex: modelo não capta lesão de jogador-chave)
- [ ] A análise traz insights sobre onde o modelo pode melhorar

### Pausa: aguardando confirmação para prosseguir para Fase 7

---

## Fase 7: Visão individual de jogos

### O que será feito

Para cada jogo já realizado da Série A 2026, criar análise de merecimento baseada em xG vs gols reais. Classificar cada resultado como "merecido", "parcialmente imerecido" ou "muito imerecido". Gerar ficha resumo por partida.

### Arquivos envolvidos

- `src/match_analysis.py` — funções de análise individual
- `src/load_data.py` — função adicional para carregar xG e stats por partida

### Implementação

**`src/match_analysis.py`:**

1. `classify_merit(goals_h, goals_a, xg_h, xg_a)`:
   - Determina resultado real (H/D/A) e resultado pelo xG (H/D/A, com margem de 0.3 para empate)
   - Se concordam → "Merecido"
   - Se discordam e |sorte_diff| > 1.5 → "Muito imerecido"
   - Senão → "Parcialmente imerecido"

2. `build_match_card(match_data, team_strengths)`:
   - Recebe dados de uma partida (gols, xG, odds, stats extras se disponíveis)
   - Calcula:
     - xG mandante vs gols mandante (sorte/azar)
     - xG visitante vs gols visitante
     - xG diff (quem mereceu ganhar)
     - P(placar real) via Poisson
     - P(vitória) via Poisson
   - Retorna dicionário com todas as métricas + veredicto de merecimento

3. `analyze_all_matches(completed_matches, team_strengths)`:
   - Aplica `build_match_card()` para cada jogo realizado
   - Retorna DataFrame com todas as fichas

4. `get_most_undeserved(match_cards, top_n=10)`:
   - Ordena por |gols_diff - xg_diff| descendente
   - Retorna top N resultados mais imerecidos da temporada

5. Se dados extras existirem (chutes, chutes a gol, posse, escanteios):
   - Incluir na ficha como métricas complementares
   - Se não existirem, ficha fica com xG vs gols apenas

### Validação automática (Claude Code checa)
- [ ] `match_analysis.py` roda sem erro
- [ ] Ficha gerada para cada jogo realizado da temporada alvo
- [ ] Classificação de merecimento atribuída a todos os jogos
- [ ] Lista dos 10 mais imerecidos gerada

### Validação manual (você checa)
- [ ] Os jogos classificados como "muito imerecido" realmente tiveram grande discrepância xG vs gols
- [ ] A ficha é legível e traz informação útil
- [ ] O top 10 de imerecidos inclui jogos que você lembra de ter sido polêmicos

### Pausa: aguardando confirmação para prosseguir para Fase 8

---

## Fase 8: Dashboard Streamlit

### O que será feito

Unir todas as fases anteriores em dashboard interativo Streamlit com 7 seções, usando Plotly para visualizações. Inclui cenário "What-if" onde o usuário fixa resultados de jogos futuros e roda nova simulação.

### Arquivos envolvidos

- `app.py` — dashboard principal Streamlit
- `data_processor.py` — prepara e formata dados para as visualizações Plotly

### Implementação

**`data_processor.py`:**
- Funções que transformam os DataFrames crus em formatos prontos para Plotly
- Cache de dados carregados (evita recalcular a cada interação)

**`app.py` — 7 seções:**

**Seção 1: Probabilidades gerais**
- Tabela interativa (st.dataframe com formatação condicional)
- Colunas: Time, Pontos, Título%, Libertadores%, Sul-Americana%, Rebaixamento%, Pontos esperados
- Barras horizontais coloridas: verde=Libertadores, azul=Sul-Americana, vermelho=rebaixamento
- Plotly horizontal bar chart

**Seção 2: Power Rankings**
- Times ordenados por ELO atual
- Variação de ELO nas últimas 5 rodadas (seta verde/vermelha)
- Gráfico Plotly: evolução de ELO na temporada (linhas por time, hover com detalhes)

**Seção 3: Heatmap de posições**
- Plotly heatmap 20×20: P(time i termina na posição j)
- Times no eixo Y (ordenados por ELO ou pontos), posições no eixo X
- Cores mais intensas = maior probabilidade
- Hover mostra P exata

**Seção 4: Deep dive por time**
- `st.selectbox()` para escolher time
- Histograma de pontos finais (distribuição de N simulações)
- Barras de distribuição de posições
- Evolução de ELO na temporada (linha temporal)
- Próximos jogos com P(H)/P(D)/P(A) do modelo

**Seção 5: Visão individual de jogos**
- Lista/dropdown de jogos já realizados
- Ao selecionar: ficha de merecimento completa
- xG vs gols, veredicto, P(placar real)
- Comparação modelo vs odds para aquele jogo específico

**Seção 6: Cenário What-if**
- Interface para fixar resultados de jogos futuros:
  - `st.selectbox()` para escolher jogo
  - `st.radio()`: vitória mandante / empate / vitória visitante
  - Botão "Adicionar" → acumula jogos fixados
  - Lista de jogos fixados com botão "Remover"
- Botão "Simular" → roda Monte Carlo com jogos fixados (resultado pré-definido) + demais jogos simulados normalmente
- Exibe tabela de probabilidades atualizada (lado a lado com cenário base)
- Highlight das diferenças: quais times mais afetados

**Seção 7: Modelo vs Odds**
- Gráfico de calibração (scatter com linha diagonal)
- Brier Score comparativo (barras: modelo vs odds)
- Tabela de maiores divergências (clicável para ver ficha do jogo)

**Stack:** Streamlit + Plotly + Pandas. Sem banco de dados intermediário — lê diretamente os CSVs gerados pelo pipeline.

**Ordem de implementação dentro desta fase:**
1. Seção 1 + 2 (tabela + rankings) — mais simples, valida o pipeline
2. Seção 3 (heatmap)
3. Seção 4 (deep dive)
4. Seção 5 (visão individual)
5. Seção 6 (what-if) — mais complexo, precisa de re-simulação
6. Seção 7 (modelo vs odds)

### Validação automática (Claude Code checa)
- [ ] `streamlit run app.py` inicia sem erro
- [ ] Todas as 7 seções renderizam sem crash
- [ ] Gráficos Plotly são interativos (hover, zoom)
- [ ] What-if: fixar um resultado e simular não dá erro
- [ ] Deep dive: selecionar qualquer time mostra dados corretos

### Validação manual (você checa)
- [ ] O dashboard é visualmente claro e fácil de navegar
- [ ] As probabilidades exibidas batem com os CSVs (sem bug de visualização)
- [ ] O heatmap é legível (cores distinguíveis, hover informativo)
- [ ] O what-if produz mudanças plausíveis nas probabilidades
- [ ] A visão individual de jogos mostra fichas informativas
- [ ] Performance aceitável (não trava ao interagir)

---

## Fase 9: Refinamentos (futuro, pós-MVP)

Lista priorizada de melhorias para implementar após o MVP estar funcionando:

| # | Refinamento | Complexidade | Impacto | Descrição |
|---|-------------|-------------|---------|-----------|
| 1 | Caching de simulações | Baixa | Alto (UX) | Salvar resultados em JSON com metadata (timestamp, parâmetros). Só re-simular se dados mudaram |
| 2 | EMA para lambdas | Média | Alto | Peso maior para jogos recentes no cálculo de poder ofensivo/defensivo. Captura forma atual |
| 3 | Regressão à média entre temporadas | Baixa | Médio | `Rating × 0.67 + 1500 × 0.33` ao início de cada temporada. Simula mudanças de elenco |
| 4 | Hot ELO na simulação | Média | Médio | Atualizar ELO após cada jogo simulado dentro do Monte Carlo. Captura momentum |
| 5 | Séries B/C alimentando ELO | Média | Médio | Times promovidos entram com ELO da divisão inferior, não com 1500 |
| 6 | K variável por margem | Baixa | Baixo | `K × ln(saldo + 1)` — goleadas valem mais, com retorno decrescente |
| 7 | Auto-refresh com novos jogos | Alta | Alto | Detectar novos dados no banco e re-rodar pipeline automaticamente |
| 8 | Modelo Dixon-Coles | Alta | Médio | Correção do Poisson para placares baixos (0x0, 1x0, 0x1 são mais comuns que o Poisson puro prevê) |

---

## Saída final

### Arquivos gerados
- `data/elo_ratings.csv` — ratings ELO de todos os times
- `data/tabela_atual_2026.csv` — classificação atual
- `data/jogos_restantes_2026.csv` — jogos futuros
- `data/team_stats_2026.csv` — ELO + poder ofensivo/defensivo
- `data/simulation_results.csv` — probabilidades por time
- `viz/exploration.ipynb` — exploração documentada dos dados
- `viz/odds_comparison.ipynb` — análise de odds

### Como usar
1. **Pipeline completo:** `python run.py --season 2026 --n-sims 10000`
2. **Dashboard:** `streamlit run app.py`
3. **Backtest:** `python run.py --backtest --season 2024`
4. **Atualizar após novos jogos:** re-rodar `python run.py` e recarregar o dashboard

### Dependências (requirements.txt)
- `pandas`, `numpy` — manipulação de dados
- `scipy` — distribuição Poisson (stats.poisson)
- `matplotlib` — gráficos estáticos (notebooks)
- `plotly` — gráficos interativos (dashboard)
- `streamlit` — framework do dashboard
- `tqdm` — barras de progresso nas simulações
