# BRForecast — Roteiro de Aprendizado v2

**Previsão do Campeonato Brasileiro Série A 2026 usando ELO + Poisson + Monte Carlo**

Inspirado no [PLForecast](https://github.com/kevzho/PLForecast) de Kevin Zhou, adaptado para o Brasileirão com dados do FootyStats. Expandido com xG, análise de odds, visão individual de jogos e cenários interativos.

---

## O que você vai construir

Um sistema completo que:

1. Calcula ratings ELO calibrados para todos os times da Série A
2. Usa modelo Poisson alimentado por xG para simular placares realistas
3. Roda simulação Monte Carlo do restante do campeonato
4. Compara as previsões do modelo com odds de casas de apostas
5. Permite análise pós-jogo individual (o resultado foi merecido?)
6. Oferece cenários "What-if" interativos (e se o Palmeiras perder 3 jogos?)
7. Exibe tudo em um dashboard Streamlit

**Pipeline completo:**

```
footystats.db (histórico completo)
    │
    ├─→ ELO ratings (força de cada time)
    │       │
    │       ├─→ Home Advantage calibrado (dos dados)
    │       └─→ Taxa de empates calibrada (dos dados)
    │
    ├─→ xG / xGA médio por time (poder ofensivo e defensivo)
    │
    ├─→ Modelo Poisson (simula placares, não só resultados)
    │       │
    │       └─→ Monte Carlo (N mil simulações do restante)
    │               │
    │               ├─→ Probabilidades: título, Libertadores, Sul-Americana, rebaixamento
    │               ├─→ Pontos esperados por time
    │               └─→ Cenários What-if
    │
    ├─→ Análise de odds (benchmark contra casas de apostas)
    │
    └─→ Visão individual de jogos (xG vs gols reais → merecimento)
```

---

## Mapa geral das fases

| Fase | O que faz | Features que habilita |
|------|-----------|----------------------|
| 0 | Exploração dos dados | Base para todas as decisões |
| 1 | Sistema ELO | Power rankings, força dos times |
| 2 | Calibração de parâmetros | Home Advantage, taxa de empates |
| 3 | Modelo Poisson + xG | Simulação de placares realistas |
| 4 | Dados da temporada 2026 | Tabela atual + jogos restantes |
| 5 | Simulação Monte Carlo | Probabilidades de posição |
| 6 | Análise de odds | Benchmark do modelo |
| 7 | Visão individual de jogos | Análise pós-jogo com xG |
| 8 | Dashboard Streamlit | Tudo junto + What-if interativo |
| 9 | Refinamentos | Melhorias incrementais |

---

## FASE 0: Exploração e entendimento dos dados

**Objetivo:** Conhecer intimamente os dados que você tem antes de escrever qualquer lógica.

### 0.1 — Inventário de temporadas

Rode a query que lista as leagues brasileiras e anote:

- Quantas temporadas da Série A existem? De qual ano até qual ano?
- Os IDs são sequenciais ou têm gaps?
- O campo `year` reflete o ano real?

### 0.2 — Anatomia de uma partida (tabela `matches`)

Pegue 5 partidas e examine campo a campo:

- Campos de gols: como se chamam? (`homeGoals`? `home_goals`? `homeGoalCount`?)
- Data: Unix timestamp? String? Qual o formato?
- IDs dos times: numéricos? Batem com a tabela `teams`?
- Status: existe campo indicando se a partida já foi jogada ou é futura?
- Temporada 2026: já tem partidas da Série A 2026 no banco?

### 0.3 — Mapeamento de xG (NOVO)

Isso é crítico para as fases de Poisson e análise individual de jogos.

Investigue nas tabelas `matches`, `match_details` e `teams`:

- Em qual tabela estão os dados de xG? (`matches`? `match_details`? ambas?)
- Como se chamam os campos? (possibilidades: `xg`, `expected_goals`, `home_xg`, `away_xg`, `team_a_xg`)
- É xG por partida (ex: Flamengo teve xG 1.8 neste jogo) ou xG acumulado na temporada?
- Qual a cobertura? Todas as temporadas têm xG ou só as mais recentes?
- Existem dados de xG em `team_details` como média por temporada?

```python
# Pseudo-código para investigar xG
xg_cols = [c for c in matches.columns if 'xg' in c.lower() or 'expected' in c.lower()]
print(f"Colunas de xG em matches: {xg_cols}")

xg_cols_details = [c for c in match_details.columns if 'xg' in c.lower() or 'expected' in c.lower()]
print(f"Colunas de xG em match_details: {xg_cols_details}")

xg_cols_teams = [c for c in teams.columns if 'xg' in c.lower() or 'expected' in c.lower()]
print(f"Colunas de xG em teams: {xg_cols_teams}")

# Cobertura temporal
# Para cada temporada, qual % das partidas tem xG preenchido?
```

### 0.4 — Mapa de odds

Você já tem o código no notebook exploratório. Agora vá além:

- As odds estão no formato decimal (ex: 2.10)?
- As 3 odds principais (`odds_ft_1`, `odds_ft_x`, `odds_ft_2`) cobrem quais temporadas?
- Existem odds de Over/Under 2.5 gols? (útil para validar o Poisson)

### 0.5 — Classificação e jogos restantes

Examine a tabela `league_tables`:

- Contém classificação final ou rodada a rodada?
- Quais campos? (pontos, vitórias, gols marcados, saldo, etc.)
- Existe para a Série A 2026?

Examine `matches` para jogos futuros:

- Como identificar jogos ainda não realizados? (status? gols NULL? data futura?)

### 0.6 — Estatísticas de casa/fora

Calcule e anote — você vai precisar disso nas Fases 2 e 3:

```python
# Pseudo-código
# Filtrar só Série A, últimas 5+ temporadas
mandante_vence = (matches['homeGoals'] > matches['awayGoals']).sum()
empates = (matches['homeGoals'] == matches['awayGoals']).sum()
visitante_vence = (matches['homeGoals'] < matches['awayGoals']).sum()
total = mandante_vence + empates + visitante_vence

print(f"Mandante vence: {mandante_vence/total:.1%}")
print(f"Empate: {empates/total:.1%}")
print(f"Visitante vence: {visitante_vence/total:.1%}")

# Média de gols por jogo
print(f"Média gols mandante: {matches['homeGoals'].mean():.2f}")
print(f"Média gols visitante: {matches['awayGoals'].mean():.2f}")
print(f"Média gols total: {(matches['homeGoals'] + matches['awayGoals']).mean():.2f}")
```

### Entrega da Fase 0

Notebook com suas anotações respondendo todas as perguntas acima. Especialmente:
- Nome exato dos campos de gols, xG, odds, status
- Cobertura temporal de xG e odds
- Estatísticas de casa/fora/empate do Brasileirão

---

## FASE 1: Sistema ELO

**Objetivo:** Calcular um rating numérico para cada time que reflita sua força atual.

### 1.1 — Entenda a teoria (antes de codar)

O sistema ELO tem 3 componentes:

**Componente 1: Resultado esperado**

```
E = 1 / (1 + 10^((Rating_adversário - Rating_time) / 400))
```

O 400 é o fator de escala. Diferença de 400 pontos = ~91% de chance para o mais forte. Diferença de 0 = 50%.

**Exercício manual:** Calcule na mão:
- Flamengo ELO 1600 vs Cuiabá ELO 1350 → qual a probabilidade esperada?
- Palmeiras 1580 vs Botafogo 1560 → e agora?

**Componente 2: Atualização do rating**

```
Novo_Rating = Rating_atual + K × (Resultado_real - Resultado_esperado)
```

Resultado_real: 1 (vitória), 0.5 (empate), 0 (derrota).

**Componente 3: O fator K** — controla a volatilidade.

### 1.2 — Decisões da Fase 1

#### Decisão 1: Rating inicial

| Opção | Valor | Prós | Contras |
|-------|-------|------|---------|
| A — Fixo | 1500 | Simples | Promovidos = tradicionais no início |
| B — Por divisão | Série A: 1500, B: 1350, C: 1200 | Mais realista | Precisa mapear origem |
| C — Regressão entre temporadas | Rating × 0.67 + 1500 × 0.33 a cada ano | Simula mudanças de elenco | Mais complexo |

**Sugestão:** Comece com A. Implemente C depois como refinamento.

#### Decisão 2: Fator K

| Opção | Valor K | Efeito |
|-------|---------|--------|
| Conservador | 15-20 | Estável, reage devagar |
| Moderado (clubelo.com) | 20 | Equilíbrio padrão |
| Agressivo | 30-40 | Reage rápido a fases de forma |

**Sugestão:** K=20. Otimize depois na fase de validação.

#### Decisão 3: Margem de gols no K

| Opção | Como funciona |
|-------|---------------|
| A — Sem ajuste | K fixo sempre |
| B — Multiplicador logarítmico | K × ln(saldo + 1) — goleadas valem mais, mas com retorno decrescente |

**Sugestão:** Comece sem (A). Adicione B depois.

### 1.3 — Implementação

**Passo 1:** Extraia e ordene cronologicamente TODAS as partidas da Série A (todas as temporadas disponíveis).

**Passo 2:** Crie `ratings = {}` — dicionário time → ELO.

**Passo 3:** Itere partida por partida, em ordem cronológica:
1. Verifique se os times já têm rating; se não, atribua 1500
2. Calcule resultado esperado (sem HFA por enquanto — HFA vem na Fase 2)
3. Determine resultado real (1, 0.5, 0)
4. Atualize ratings de ambos
5. Salve o histórico: `(data, time, elo_antes, elo_depois)`

**Passo 4:** Exporte CSV: `time, elo_rating, ultima_partida`.

### 1.4 — Validação do ELO

Antes de seguir:

- Top 5 ELO = times historicamente fortes?
- Bottom 5 ELO = times que caíram ou são fracos?
- Diferença entre 1º e 20º está entre 200-400 pontos?
- Plote evolução do ELO de 4 times ao longo dos anos. Faz sentido?

### Entrega da Fase 1

- `src/elo.py` — script que processa todas as partidas
- `data/elo_ratings.csv` — ratings atuais
- Gráfico de evolução de ELO (pelo menos 4 times)
- Anotações: K=___, rating inicial=___, por quê

---

## FASE 2: Calibração de parâmetros

**Objetivo:** Usar seus dados históricos para calibrar Home Advantage e taxa de empates de forma empírica — não com chutes.

### 2.1 — Home Field Advantage (HFA)

O HFA é um bônus em pontos ELO adicionado ao mandante na hora de calcular o resultado esperado.

**Como calibrar com seus dados:**

Você já calculou o % de vitórias do mandante na Fase 0. Agora o exercício é:

1. Rode o cálculo de ELO da Fase 1 variando o HFA de 0 a 120 em incrementos de 10
2. Para cada valor de HFA, calcule o Brier Score ou a Log-Loss das previsões vs resultados reais
3. O HFA que minimiza o erro é o seu HFA ótimo

```python
# Pseudo-código
resultados_calibracao = []

for hfa in range(0, 130, 10):
    # Roda o ELO com esse HFA
    # Para cada partida, calcula a probabilidade esperada
    # Compara com o resultado real
    # Calcula o Brier Score
    brier = calcular_brier(partidas, hfa)
    resultados_calibracao.append({'hfa': hfa, 'brier': brier})

# Plota: eixo X = HFA, eixo Y = Brier Score
# O ponto mais baixo é o seu HFA ótimo
```

**Referência:** Na Premier League, o HFA ótimo fica entre 50-70 pontos. No Brasileirão, espera-se um valor mais alto (65-100) porque o mando de campo historicamente pesa mais no Brasil.

### 2.2 — Taxa de empates

O ELO clássico dá P(time A vence) mas não P(empate). Você precisa modelar isso.

**Opção A — Constante:** Use a taxa histórica que calculou na Fase 0 (provavelmente ~25-30% no Brasileirão).

**Opção B — Variável com a diferença de ELO:** Times próximos em ELO empatam mais. Uma fórmula simples:

```
P_draw = taxa_base × (1 - abs(elo_home_ajustado - elo_away) / 800)
```

Onde `taxa_base` é a taxa histórica e `elo_home_ajustado = elo_home + HFA`.

Depois: `P_home_win = E_home × (1 - P_draw)` e `P_away_win = (1 - E_home) × (1 - P_draw)`.

**Exercício:** Calcule com seus dados: a taxa de empates é maior quando a diferença de ELO é pequena? Isso valida se a opção B faz sentido.

```python
# Pseudo-código
# Para cada partida, calcule a diferença de ELO
# Agrupe em faixas (0-50, 50-100, 100-150, etc.)
# Calcule a taxa de empates em cada faixa
# Se a taxa cai conforme a diferença aumenta → opção B vale a pena
```

### Entrega da Fase 2

- Gráfico de calibração HFA (Brier Score vs HFA)
- Valor ótimo de HFA encontrado
- Análise da taxa de empates por faixa de diferença de ELO
- Decisão: taxa constante ou variável?
- Atualizar `src/config.py` com os parâmetros calibrados

---

## FASE 3: Modelo Poisson + xG

**Objetivo:** Em vez de simular só "vitória/empate/derrota", simular placares realistas usando distribuição de Poisson alimentada por xG.

### 3.1 — Por que Poisson?

O modelo Poisson assume que gols em futebol seguem uma distribuição de Poisson — cada time tem uma taxa esperada de gols (lambda), e o número real de gols é sorteado dessa distribuição.

Isso permite:
- Simular placares (2x1, 0x0, 3x2) em vez de só resultados
- Calcular saldo de gols nas simulações (importante para desempate)
- Ter uma base mais sólida para a análise individual de jogos

### 3.2 — Como calcular o lambda de cada time

O lambda (taxa esperada de gols) de cada time em cada jogo depende de:
- Poder ofensivo do time (quanto ele ataca acima da média)
- Poder defensivo do adversário (quanto ele defende acima/abaixo da média)
- Média geral de gols da liga
- Se é mandante ou visitante

**Com gols reais (versão básica):**

```
Ataque_time = média de gols marcados pelo time / média de gols da liga
Defesa_time = média de gols sofridos pelo time / média de gols da liga

Lambda_mandante = Ataque_mandante × Defesa_visitante × média_gols_mandante_liga
Lambda_visitante = Ataque_visitante × Defesa_mandante × média_gols_visitante_liga
```

**Com xG (versão melhorada):**

Substitua "gols marcados" por "xG gerado" e "gols sofridos" por "xGA sofrido". Isso remove o ruído da sorte (pênaltis defendidos, gols bizarros, etc.) e captura melhor a qualidade do jogo.

```
Ataque_xG_time = média de xG gerado pelo time / média de xG da liga
Defesa_xGA_time = média de xGA sofrido pelo time / média de xGA da liga

Lambda_mandante = Ataque_xG_mandante × Defesa_xGA_visitante × média_gols_mandante_liga
Lambda_visitante = Ataque_xG_visitante × Defesa_xGA_mandante × média_gols_visitante_liga
```

### 3.3 — Decisões da Fase 3

#### Decisão 4: Usar gols reais ou xG para calcular os lambdas?

| Opção | Prós | Contras |
|-------|------|---------|
| Gols reais | Simples, sem dependência de dados extras | Ruidoso — times com "sorte" parecem melhores |
| xG | Mais preditivo, captura qualidade real | Depende da cobertura de xG nos dados |
| Média ponderada (70% xG + 30% gols) | Equilíbrio entre qualidade e resultados reais | Mais um parâmetro para calibrar |

**Sugestão:** Use xG se a cobertura for boa (>80% das partidas das últimas 3+ temporadas). Se não, comece com gols reais.

#### Decisão 5: Janela temporal para cálculo dos lambdas

Usar todas as partidas da temporada ou só as últimas N rodadas?

| Opção | Efeito |
|-------|--------|
| Temporada inteira | Mais estável, menos sensível a fases |
| Últimas 10 rodadas | Captura forma atual, mais volátil |
| Média ponderada exponencial (EMA) | Peso maior para jogos recentes, considera toda a temporada |

**Sugestão:** Temporada inteira para começar. EMA como refinamento.

#### Decisão 6: Combinar ELO com Poisson ou usar separados?

| Opção | Como funciona |
|-------|---------------|
| A — Poisson puro | Lambda calculado só com xG/gols, sem ELO. ELO fica só como ranking |
| B — ELO para probabilidade, Poisson para placar | ELO define quem vence, Poisson define o placar dentro desse resultado |
| C — ELO ajusta o lambda do Poisson | Lambda base do Poisson é ajustado pela diferença de ELO |

**Sugestão:** Opção A para começar (Poisson puro para simulação, ELO como ranking separado). A opção C é mais sofisticada e pode ser refinamento futuro.

### 3.4 — Implementação

**Passo 1:** Calcule poder ofensivo e defensivo de cada time da Série A 2026 (usando xG se disponível).

**Passo 2:** Escreva a função `simular_placar(lambda_home, lambda_away)`:

```python
import numpy as np

def simular_placar(lambda_home, lambda_away):
    gols_home = np.random.poisson(lambda_home)
    gols_away = np.random.poisson(lambda_away)
    return gols_home, gols_away
```

**Passo 3:** Teste a distribuição. Simule 10.000 vezes um jogo com lambda_home=1.5 e lambda_away=1.0. Verifique:
- O mandante vence em ~50% das vezes?
- O % de empates é ~25%?
- A média de gols é ~2.5?

### 3.5 — Validação do Poisson

Compare as previsões do Poisson com resultados reais da temporada passada:
- O modelo prevê ~25-30% de empates? (Compatível com o Brasileirão?)
- A distribuição de gols simulada é parecida com a real?
- Existe excesso de 0x0 ou de goleadas na simulação vs realidade?

**Teste de calibração:** Para jogos já realizados da Série A 2024 ou 2025, calcule os lambdas e compare:
- A probabilidade que o modelo dava para o resultado real
- O Brier Score do Poisson vs do ELO simples

### Entrega da Fase 3

- `src/poisson.py` — cálculo de lambdas e simulação de placares
- Poder ofensivo/defensivo de cada time (xG-based se possível)
- Validação: distribuição de gols simulada vs real
- Brier Score comparando ELO simples vs Poisson

---

## FASE 4: Dados da temporada 2026

**Objetivo:** Montar a foto atual do campeonato — classificação + jogos restantes.

### 4.1 — Tabela atual

DataFrame necessário:

| time | jogos | vitorias | empates | derrotas | gols_pro | gols_contra | saldo | pontos |
|------|-------|----------|---------|----------|----------|-------------|-------|--------|

Origem: `league_tables` do banco (se existir para 2026) ou construir a partir de `matches`.

### 4.2 — Jogos restantes

DataFrame necessário:

| rodada | mandante | visitante |
|--------|----------|-----------|

Identificar jogos futuros: campo de status, gols NULL, ou data futura.

### 4.3 — Merge com ELO + estatísticas Poisson

Juntar tudo em um DataFrame completo:

| time | jogos | pontos | elo | ataque_xg | defesa_xga | lambda_home_medio | lambda_away_medio |
|------|-------|--------|-----|-----------|------------|-------------------|-------------------|

### Entrega da Fase 4

- `src/load_data.py` — carrega tudo do SQLite
- `data/tabela_atual_2026.csv`
- `data/jogos_restantes_2026.csv`
- `data/team_stats_2026.csv` (com ELO + poder ofensivo/defensivo)

---

## FASE 5: Simulação Monte Carlo

**Objetivo:** Simular o restante do campeonato milhares de vezes usando o modelo Poisson.

### 5.1 — Lógica

```
Para cada simulação (de 1 até N_SIMULAÇÕES):
    1. Copiar a tabela atual
    2. Copiar os ratings ELO e estatísticas ofensivas/defensivas
    3. Para cada jogo restante:
        a. Calcular lambda_home e lambda_away (com Poisson + xG)
        b. Sortear um placar usando Poisson
        c. Atualizar a tabela (pontos, gols, saldo)
        d. (Opcional) Atualizar ELOs e lambdas (hot update)
    4. Aplicar critérios de desempate
    5. Registrar a posição final e os pontos de cada time
```

### 5.2 — Decisões da Fase 5

#### Decisão 7: Hot update nas simulações — **DECIDIDO: Hot ELO + Hot Lambda**

| Opção | Efeito |
|-------|--------|
| Sem update (fixo) | Mais rápido, mais simples |
| Hot ELO | Atualiza ELO após cada jogo simulado. Captura momentum |
| **Hot ELO + Hot Lambda** | **Atualiza ELO e recalcula lambdas. Máximo realismo, mais lento** |

**Escolha:** Hot ELO + Hot Lambda. Após cada jogo simulado, o ELO de ambos os times é atualizado com K=30 (fixo). Os lambdas dos jogos seguintes na mesma simulação são recalculados usando o ELO atualizado, capturando momentum e mudanças de forma dentro da simulação. As forças ofensivas/defensivas (baseadas em xG real) permanecem fixas. Implementado em `simulation.py` com flag `HOT_UPDATE=True` no config.

#### Decisão 8: Número de simulações — **DECIDIDO: 20.000**

| Quantidade | Tempo aprox. | Precisão |
|------------|-------------|----------|
| 1.000 | Segundos | Instável |
| 10.000 | ~30 seg (estático) | Bom equilíbrio |
| **20.000** | **~3-5 min (hot)** | **Estável** |

**Escolha:** 20.000 simulações sempre (dev e prod). Com hot update, o tempo sobe para ~3-5 minutos, mas a estabilidade dos resultados compensa.

#### Decisão 9: Critérios de desempate — **DECIDIDO: Até saldo de gols**

O Brasileirão usa, em ordem: número de vitórias → saldo de gols → gols marcados → confronto direto.

**Escolha:** Implementado até gols marcados (pontos → vitórias → saldo → gols pró). Confronto direto omitido — complexo em simulação e raramente decide.

### 5.3 — Agregação dos resultados

Após N simulações, calcule para cada time:

```python
probabilidades = {
    'titulo': vezes_posicao_1 / N,
    'libertadores': vezes_posicao_1a6 / N,
    'sulamericana': vezes_posicao_7a12 / N,
    'rebaixamento': vezes_posicao_17a20 / N,
    'pontos_esperados': media_pontos,
    'pontos_min': min_pontos,
    'pontos_max': max_pontos,
    'pontos_p10': percentil_10,
    'pontos_p90': percentil_90,
    'distribuicao_posicoes': [prob_pos_1, prob_pos_2, ..., prob_pos_20]
}
```

### 5.4 — Backtest obrigatório

Rode a simulação para a Série A 2024 (temporada encerrada). Simule a partir de diferentes rodadas (10, 15, 20, 25, 30) e compare:
- O campeão real tinha a maior probabilidade?
- Os rebaixados tinham alta % de queda?
- À medida que avança na temporada, o modelo converge para a realidade?

### Entrega da Fase 5

- `src/simulation.py` — lógica de Monte Carlo
- `run.py` — ponto de entrada
- `data/simulation_results_2026.csv`
- Backtest documentado com Série A 2024

---

## FASE 6: Análise de odds

**Objetivo:** Usar as odds do FootyStats como benchmark para avaliar a qualidade do seu modelo.

### 6.1 — Converter odds para probabilidades

```python
def odds_para_probabilidade(odds_home, odds_draw, odds_away):
    # Probabilidades implícitas (com overround)
    p_home = 1 / odds_home
    p_draw = 1 / odds_draw
    p_away = 1 / odds_away

    # Normalizar (remover overround)
    total = p_home + p_draw + p_away
    return p_home/total, p_draw/total, p_away/total
```

### 6.2 — Comparações a fazer

**Por jogo (para jogos já realizados):**

| Partida | P(H) Modelo | P(H) Odds | P(D) Modelo | P(D) Odds | P(A) Modelo | P(A) Odds | Resultado |
|---------|-------------|-----------|-------------|-----------|-------------|-----------|-----------|

**Métricas globais:**

- **Brier Score:** `média de (previsão - resultado)²`. Calcule para seu modelo E para as odds. Compare.
- **Log-Loss:** Penaliza mais fortemente previsões confiantes que erram.
- **Calibração:** Agrupe previsões em faixas (0-10%, 10-20%, ..., 90-100%). Em cada faixa, o % real de acertos deveria ser próximo da probabilidade prevista.

### 6.3 — Onde o modelo diverge das odds

Identifique os jogos onde a maior diferença entre seu modelo e as odds. Isso revela:
- Onde seu modelo pode estar errado (ELO desatualizado, time teve mudança de elenco)
- Onde as odds podem estar enviesadas (times populares, efeito público)

### Entrega da Fase 6

- `viz/odds_comparison.ipynb` — análise completa
- Brier Score do modelo vs odds
- Gráfico de calibração
- Mapa de divergências modelo vs odds

---

## FASE 7: Visão individual de jogos (NOVO)

**Objetivo:** Para cada jogo já realizado, responder: o resultado foi merecido?

### 7.1 — O conceito

Um jogo onde o time A ganhou de 1x0 mas teve xG de 0.4 vs 1.8 do adversário é um resultado "imerecido" — o vencedor teve sorte. Inversamente, um 2x0 com xG 2.5 vs 0.3 é totalmente merecido.

### 7.2 — Métricas por jogo

Para cada partida realizada, calcule:

| Métrica | O que mostra |
|---------|-------------|
| xG mandante vs Gols mandante | Mandante teve sorte (gols > xG) ou azar (gols < xG)? |
| xG visitante vs Gols visitante | Mesma lógica para visitante |
| xG diff (xG_home - xG_away) | Quem "mereceu" ganhar pelo xG |
| Resultado real vs resultado esperado pelo xG | O resultado refletiu a qualidade do jogo? |
| P(resultado real) pelo Poisson | Quão provável era esse resultado acontecer? |

### 7.3 — Classificação de merecimento

Crie uma classificação simples:

```python
def classificar_merecimento(gols_home, gols_away, xg_home, xg_away):
    resultado_real = 'H' if gols_home > gols_away else ('D' if gols_home == gols_away else 'A')
    resultado_xg = 'H' if xg_home > xg_away else ('D' if abs(xg_home - xg_away) < 0.3 else 'A')

    if resultado_real == resultado_xg:
        return "Merecido"
    else:
        sorte_diff = (gols_home - gols_away) - (xg_home - xg_away)
        if abs(sorte_diff) > 1.5:
            return "Muito imerecido"
        else:
            return "Parcialmente imerecido"
```

### 7.4 — Visualização por jogo

Para cada partida, uma "ficha" com:

```
┌─────────────────────────────────────────────┐
│  Flamengo 2 × 1 Palmeiras                  │
│  Rodada 15 — 10/07/2026                     │
├─────────────────────────────────────────────┤
│  xG: 1.3 vs 2.1                            │
│  Chances claras: 3 vs 5                     │
│  Chutes a gol: 4 vs 7                      │
│  Veredicto: IMERECIDO — Palmeiras merecia   │
│  P(2x1 para Flamengo): 8.2%                │
│  P(vitória Flamengo): 32.1%                │
│  Odds pré-jogo: 2.50 | 3.20 | 2.80         │
├─────────────────────────────────────────────┤
│  [Barra visual: xG timeline]               │
│  [Gráfico: gols vs xG ambos os times]      │
└─────────────────────────────────────────────┘
```

### 7.5 — Dados necessários

Verifique na Fase 0 se `match_details` contém:
- xG por time por partida (obrigatório)
- Chutes / chutes a gol (desejável)
- Chances claras / big chances (desejável)
- Posse de bola (desejável)
- Escanteios, faltas (opcional)

Se os dados extras não existirem, a ficha fica mais simples (só xG vs gols).

### Entrega da Fase 7

- `src/match_analysis.py` — funções de análise individual
- Ficha de merecimento para cada jogo realizado da Série A 2026
- Lista dos 10 resultados mais "imerecidos" da temporada

---

## FASE 8: Dashboard Streamlit

**Objetivo:** Unir tudo em um dashboard interativo.

### 8.1 — Abas/Seções do dashboard

**Seção 1: Probabilidades gerais**
- Tabela com todos os times: Título %, Libertadores %, Sul-Americana %, Rebaixamento %, Pontos esperados
- Barras horizontais coloridas por zona (verde=Libertadores, azul=Sul-Americana, vermelho=rebaixamento)

**Seção 2: Power Rankings**
- Times ordenados por ELO
- Variação de ELO nas últimas 5 rodadas

**Seção 3: Heatmap de posições**
- Matriz 20×20: probabilidade de cada time terminar em cada posição
- Cores mais intensas = maior probabilidade

**Seção 4: Deep dive por time**
- Dropdown para selecionar um time
- Distribuição de pontos finais (histograma)
- Distribuição de posições (barras)
- Evolução de ELO na temporada
- Próximos jogos com probabilidades

**Seção 5: Visão individual de jogos (NOVO)**
- Dropdown ou tabela para selecionar um jogo já realizado
- Ficha de merecimento (xG vs gols, veredicto)
- Comparação modelo vs odds para aquele jogo

**Seção 6: Cenário What-if (NOVO)**
- Interface onde o usuário seleciona resultados de jogos futuros
- Dropdown: selecionar um jogo futuro → escolher: vitória mandante / empate / vitória visitante
- Pode fixar múltiplos jogos
- Botão "Simular" → roda Monte Carlo com os resultados fixados
- Mostra como as probabilidades mudam

```python
# Pseudo-código do What-if
jogos_fixados = {
    ('Palmeiras', 'Flamengo'): 'away_win',  # Flamengo vence
    ('Corinthians', 'São Paulo'): 'draw',     # Empate
}

# Na simulação, para jogos fixados, usar o resultado definido
# Para os demais, simular normalmente
```

**Seção 7: Modelo vs Odds**
- Gráfico de calibração
- Brier Score comparativo
- Divergências por jogo

### 8.2 — Stack

- **Streamlit** — framework
- **Plotly** — gráficos interativos (heatmap, barras, distribuições, scatter)
- **Pandas** — manipulação dos dados

### 8.3 — Implementação

Comece pela seção mais simples (Power Rankings) e vá adicionando complexidade:

1. Primeiro: Seção 1 + 2 (tabela de probabilidades + rankings)
2. Depois: Seção 3 (heatmap)
3. Depois: Seção 4 (deep dive)
4. Depois: Seção 5 (visão individual)
5. Depois: Seção 6 (What-if)
6. Por último: Seção 7 (modelo vs odds)

### Entrega da Fase 8

- `app.py` — dashboard principal
- `data_processor.py` — prepara dados para Plotly
- Dashboard rodando com `streamlit run app.py`

---

## FASE 9: Refinamentos (pós-MVP)

Melhorias para implementar uma a uma, quando o core estiver funcionando:

| Refinamento | Complexidade | Impacto |
|-------------|-------------|---------|
| Regressão à média entre temporadas | Baixa | Médio |
| Hot ELO na simulação | Média | Médio |
| K variável por margem de vitória | Baixa | Baixo |
| EMA para lambdas (forma recente) | Média | Alto |
| Séries B/C alimentando o ELO de promovidos | Média | Médio |
| Caching de simulações (JSON com metadata) | Baixa | Alto (UX) |
| Auto-refresh quando novos jogos entram no banco | Alta | Alto (automação) |
| Modelo Dixon-Coles (Poisson corrigido para empates baixos) | Alta | Médio |

---

## Estrutura final do projeto

```
BRForecast/
├── data/
│   ├── footystats.db                # banco SQLite (já existe)
│   ├── elo_ratings_2026.csv         # Fase 1
│   ├── tabela_atual_2026.csv        # Fase 4
│   ├── jogos_restantes_2026.csv     # Fase 4
│   ├── team_stats_2026.csv          # Fase 4 (ELO + xG stats)
│   └── simulation_results.csv       # Fase 5
├── src/
│   ├── config.py                    # K, HFA, N_SIMS, zonas do brasileirão
│   ├── elo.py                       # cálculo de ELO
│   ├── poisson.py                   # cálculo de lambdas + simulação de placares
│   ├── load_data.py                 # carrega dados do SQLite
│   ├── simulation.py                # Monte Carlo
│   ├── table.py                     # atualização de classificação
│   └── match_analysis.py            # análise individual (xG, merecimento)
├── viz/
│   ├── elo_exploration.ipynb        # Fases 0-2
│   └── odds_comparison.ipynb        # Fase 6
├── app.py                           # Streamlit dashboard
├── data_processor.py                # prepara dados para visualização
├── run.py                           # ponto de entrada simulação
├── cache_manager.py                 # cache de resultados
└── requirements.txt
```

---

## Checklist de progresso

### Fase 0: Exploração
- [ ] Campos de gols, datas, status identificados
- [ ] Campos de xG localizados e cobertura mapeada
- [ ] Campos de odds mapeados
- [ ] % mandante/empate/visitante calculado
- [ ] Média de gols por jogo calculada
- [ ] Temporada 2026 verificada no banco

### Fase 1: ELO
- [ ] Decisões: K=___, rating inicial=___
- [ ] Script `elo.py` funcionando
- [ ] CSV com ratings exportado
- [ ] Gráfico de evolução de ELO
- [ ] Validação: top/bottom 5 fazem sentido

### Fase 2: Calibração
- [ ] HFA ótimo encontrado: ___
- [ ] Taxa de empates analisada (constante ou variável)
- [ ] `config.py` atualizado

### Fase 3: Poisson + xG
- [ ] Decisão: xG ou gols reais para lambdas
- [ ] Poder ofensivo/defensivo de cada time calculado
- [ ] `simular_placar()` testada
- [ ] Distribuição de gols simulada vs real comparada
- [ ] Brier Score: Poisson vs ELO simples

### Fase 4: Dados 2026
- [ ] Tabela atual carregada
- [ ] Jogos restantes identificados
- [ ] Merge com ELO + stats feito

### Fase 5: Monte Carlo
- [ ] Decisões: N_sims=___, hot update sim/não
- [ ] Simulação completa rodada
- [ ] Probabilidades por time e zona calculadas
- [ ] Backtest com Série A 2024

### Fase 6: Odds
- [ ] Odds convertidas para probabilidades
- [ ] Brier Score modelo vs odds
- [ ] Gráfico de calibração
- [ ] Divergências mapeadas

### Fase 7: Visão individual
- [ ] Dados de xG por partida disponíveis
- [ ] Classificação de merecimento implementada
- [ ] Ficha de jogo funcionando
- [ ] Top 10 resultados mais imerecidos

### Fase 8: Dashboard
- [ ] Seção 1: Probabilidades gerais
- [ ] Seção 2: Power Rankings
- [ ] Seção 3: Heatmap de posições
- [ ] Seção 4: Deep dive por time
- [ ] Seção 5: Visão individual de jogos
- [ ] Seção 6: What-if interativo
- [ ] Seção 7: Modelo vs Odds

---

## Referências para estudo

### ELO em futebol
- clubelo.com — Metodologia de ELO para clubes
- eloratings.net — ELO de seleções
- "How to Build an ELO Rating System for the Premier League in Python" — StatsUltra (2025)
- opisthokonta.net — "Tuning the Elo ratings: The K-factor and home field advantage"

### Poisson em futebol
- "Dixon-Coles model" — Busque por "Dixon Coles football prediction Python" (modelo Poisson melhorado)
- "Predicting Football Results With Statistical Modelling" — dashee87.github.io (tutorial clássico)

### Monte Carlo
- PLForecast (github.com/kevzho/PLForecast) — Projeto de referência
- "Predicting the Outcome of the EPL By Using Monte Carlo Method" — Medium/The Startup

### xG
- Documentação do FootyStats sobre Expected Goals
- "What is xG?" — StatsBomb (explicação do conceito)

---

*Roteiro v2 — março/2026. Inclui modelo Poisson, análise de xG, visão individual de jogos, cenários What-if e comparação com odds. Adapte conforme seu ritmo.*
