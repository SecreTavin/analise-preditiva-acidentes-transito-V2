# 🚗 Análise Preditiva de Acidentes de Trânsito

![Status](https://img.shields.io/badge/Status-Completo-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Data](https://img.shields.io/badge/Dataset-817K%20registros-red)

Projeto de Machine Learning e Análise de Séries Temporais para prever a gravidade de acidentes de trânsito usando dados reais da Polícia Rodoviária Federal (PRF). Desenvolvido como parte do trabalho A3 do curso de Engenharia de Software.

## 📊 Sobre o Projeto

Este projeto implementa um **pipeline completo de análise de dados** que combina técnicas de **Machine Learning** e **Análise de Séries Temporais** seguindo as **15 etapas** do Guia Prático de Análise de Séries Temporais. 

O objetivo é entender padrões em acidentes de trânsito e prever quais têm maior probabilidade de serem fatais, fornecendo insights acionáveis para políticas públicas de segurança viária.

### 🎯 Objetivos

- **Classificação:** Prever se um acidente será FATAL ou NÃO-FATAL
- **Análise Temporal:** Identificar padrões sazonais e tendências na taxa de fatalidade
- **Insights:** Descobrir fatores de risco mais significativos
- **Conformidade:** Seguir rigorosamente as 15 etapas do guia científico

---

## 📈 Dados do Projeto

### Dataset Atual
- **817.958 registros** de acidentes
- **Período:** Janeiro a Agosto de 2025
- **Fonte:** Polícia Rodoviária Federal (PRF)
- **Desbalanceamento:** 0.41% fatais vs 99.59% não-fatais
- **Série Temporal:** 243 dias de dados contínuos

### Variáveis Principais
- **Temporais:** data_hora, hora, dia_semana, mes, ano
- **Geográficas:** endereco_id, numero_vitimas, numero_veiculos
- **Descritivas:** tipo_acidente, descricao, gravidade

---

## 🔧 Pipeline Completo: 15 Etapas

### Etapa 1: Definir o Objetivo ✅
- Classificação binária: Fatal vs Não-Fatal
- Análise de série temporal da taxa de fatalidade

### Etapa 2: Carregar e Organizar os Dados ✅
- Conexão MySQL com SQLAlchemy
- 817.958 registros carregados
- Período: Jan-Ago 2025

### Etapa 3: Verificar Dados Faltantes ✅
- Tratamento de valores ausentes
- Numéricos: preenchidos com 0
- Categóricos: preenchidos com "Desconhecido"

### Etapa 4: Identificar Outliers ✅ **NOVO**
- **Método 1:** Z-Score (σ > 3)
- **Método 2:** IQR (Intervalo Interquartil)
- **Resultado:** 45.590 outliers em `endereco_id` e `numero_vitimas`
- **Decisão:** Mantidos (representam eventos reais)

### Etapa 5: Estatísticas Descritivas ✅
- Taxa média de fatalidade: 0.41%
- 3.357 acidentes fatais
- 814.601 acidentes não-fatais

### Etapa 6: Visualizar a Série ✅
- 6 gráficos exploratórios gerados
- Heatmap de gravidade por hora
- Distribuição temporal (hora/dia/mês)

### Etapa 7: Decomposição da Série Temporal ✅ **NOVO**
- **Modelo:** Aditivo
- **Período:** 7 dias (semanal)
- **Componentes extraídos:**
  - Tendência
  - Sazonalidade semanal
  - Resíduos

### Etapa 8: Teste de Estacionariedade ✅ **NOVO**
- **Teste ADF:** p=0.00001 → Estacionária ✅
- **Teste KPSS:** p=0.01545 → Não-estacionária ⚠️
- **Interpretação:** Série na "fronteira" de estacionariedade
- **Análise adicional:** ACF/PACF gerados

### Etapa 9: Identificar Padrões ✅
- **Natureza:** Não-linear (Random Forest ideal)
- **Sazonalidade:** SIM (ciclo semanal detectado)
- **Tendência:** Presente

### Etapa 10-11: Seleção e Divisão ✅
- **Modelos:** Random Forest, Regressão Logística, SVM Linear
- **Divisão:** 70% treino (572.570) / 30% teste (245.388)
- **Normalização:** StandardScaler aplicado

### Etapa 12: Avaliação de Modelos ✅
- **Random Forest:** AUC-ROC = 0.9861 ⭐ MELHOR
- **Regressão Logística:** AUC-ROC = 0.9837
- **SVM Linear:** AUC-ROC = 0.1809

### Etapa 13: Análise de Resíduos ✅ **EXPANDIDO**
- **Teste Shapiro-Wilk:** p < 0.05 (resíduos não-normais)
- **Q-Q Plot:** Gerado para análise visual
- **Scatter:** Resíduos vs Probabilidades

### Etapa 14: Detectar Anomalias ✅ **NOVO**
- **Método:** Isolation Forest
- **Contamination:** 5%
- **Resultado:** 12.270 anomalias detectadas (5.00%)

### Etapa 15: Testes Estatísticos Finais ✅ **NOVO**
- **Chi-Quadrado:** p=0.0 (predições correlacionadas com reais)
- **Cohen Kappa:** 0.2241 (RF) - Concordância "Razoável"
- **Balanced Accuracy:** Calculado para todos os modelos

---

## 🏆 Resultados

### Performance dos Modelos

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|--------|----------|----------|--------|----------|---------|
| **Random Forest** ⭐ | 99.54% | 37.78% | 18.37% | 0.2472 | **0.9861** |
| Regressão Logística | 94.83% | 7.35% | 100% | 0.1369 | 0.9837 |
| SVM Linear | 97.02% | 8.61% | 65.24% | 0.1522 | 0.1809 |

### Matriz de Confusão - Random Forest (Melhor Modelo)

|   | Previsto Não-Fatal | Previsto Fatal |
|---|-------------------|----------------|
| **Real Não-Fatal** | 244.076 (TN) | 305 (FP) |
| **Real Fatal** | 822 (FN) | 185 (TP) |

**Interpretação:**
- ✅ **Taxa FP:** 0.12% - Pouquíssimos alarmes falsos
- ⚠️ **Taxa FN:** 81.63% - Detecta 18.37% dos acidentes fatais
- 💡 **Trade-off aceitável:** Alta precisão (37.78%) com recall moderado

### Feature Importance (Top 10)

| Rank | Feature | Importância |
|------|---------|-------------|
| 1 | endereco_id | 21.24% |
| 2 | periodo_dia | 16.73% |
| 3 | dia | 15.57% |
| 4 | mes | 15.38% |
| 5 | ano | 8.57% |
| 6 | numero_vitimas | 7.40% |
| 7 | dia_semana | 7.01% |
| 8 | hora | 3.85% |
| 9 | descricao | 1.89% |
| 10 | tipo_acidente | 1.46% |

**Insight:** A localização (`endereco_id`) é o preditor mais importante (21.24%)!

### Testes de Estacionariedade

| Teste | P-valor | Resultado |
|-------|---------|-----------|
| ADF | 0.00001 | ✅ Estacionária |
| KPSS | 0.01545 | ⚠️ Não-estacionária |

**Interpretação:** Resultados contraditórios indicam série na "fronteira". Requer análise ACF/PACF.

---

## 💻 Tecnologias Utilizadas

### Core
- **Python 3.12**
- **pandas** & **numpy** - Manipulação de dados
- **scikit-learn** - Machine Learning
- **MySQL** + **SQLAlchemy** - Banco de dados

### Visualização
- **matplotlib** & **seaborn** - Gráficos

### Análise de Séries Temporais
- **statsmodels** - Decomposição, ADF, KPSS, ACF/PACF
- **scipy.stats** - Testes estatísticos (Chi², Shapiro-Wilk)

### Machine Learning Avançado
- **RandomizedSearchCV** - Otimização de hiperparâmetros
- **Isolation Forest** - Detecção de anomalias
- **Cohen Kappa** - Concordância

---

## 📁 Estrutura de Arquivos

```
projeto/
│
├── halving_search_15_ETAPAS_FINAL.py   # Código principal (900+ linhas)
│
├── outputs/                             # Arquivos gerados (15+)
│   ├── 04_analise_outliers.csv
│   ├── 06_visualizacao_serie.png
│   ├── 07_decomposicao_series.png
│   ├── 07_componentes_decomposicao.csv
│   ├── 08_acf_pacf.png
│   ├── 08_testes_estacionariedade.csv
│   ├── 12_comparacao_modelos.csv
│   ├── 12_resultados_modelos.png
│   ├── 13_analise_residuos.csv
│   ├── 13_analise_residuos_visual.png
│   ├── 14_resumo_anomalias.csv
│   ├── 15_testes_finais.csv
│   ├── previsoes_proximos_3_dias.csv
│   ├── resumo_matrizes_confusao.csv
│   └── random_forest_modelo.pkl        # Modelo treinado
│
├── docs/
│   ├── RESUMO_ALTERACOES_CODIGO.md     # Changelog técnico
│   ├── DOCUMENTO_MUDANCAS_IMPLEMENTADAS.md
│   └── CORRECOES_NUMERICAS_RELATORIO.md
│
└── README.md                            # Este arquivo
```

---

## 🚀 Como Executar

### Pré-requisitos
```bash
# Python 3.12+
# MySQL Server rodando

# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn sqlalchemy mysql-connector-python statsmodels scipy
```

### Execução
```bash
# Rodar pipeline completo
python halving_search_15_ETAPAS_FINAL.py

# Tempo esperado: 15-20 minutos
# Saída: 15+ arquivos gerados
```

### Configuração MySQL
```python
# Ajustar credenciais no código (linha 50)
engine = create_engine('mysql+mysqlconnector://USER:PASS@localhost/analise_transito')
```

---

## 📊 Outputs Gerados

### Visualizações (6 gráficos)
1. **06_visualizacao_serie.png** - EDA com 6 subplots
2. **07_decomposicao_series.png** - Tendência + Sazonalidade + Resíduos
3. **08_acf_pacf.png** - Autocorrelação e série temporal
4. **12_resultados_modelos.png** - Performance comparativa
5. **13_analise_residuos_visual.png** - Normalidade dos resíduos

### Dados (10 CSVs)
- Outliers, Estacionariedade, Resíduos, Anomalias
- Matrizes de Confusão, Feature Importance
- Testes Estatísticos, Comparação de Modelos
- Previsões próximos 3 dias

### Modelo Treinado
- **random_forest_modelo.pkl** - Modelo serializado pronto para deploy

---

## 🎓 Aprendizados e Insights

### Técnicos
- ✅ **Tratamento de desbalanceamento:** `class_weight='balanced'` funcionou bem
- ✅ **Random Forest superior:** Ensemble learning > modelos lineares para este problema
- ✅ **SVM Linear limitado:** Kernel linear muito simples (AUC 0.18 vs 0.98 do RF)
- ✅ **Localização crítica:** `endereco_id` é o preditor #1 (21.24%)

### Metodológicos
- ✅ **15 etapas implementadas:** Conformidade 100% com guia para realização do trabalho
- ✅ **Série temporal adaptada:** Decomposição aplicada à taxa de fatalidade
- ✅ **Testes robustos:** ADF + KPSS + ACF/PACF + Chi² + Kappa
- ✅ **Anomalias detectadas:** 5% dos dados identificados como outliers multivariados

### Negócio
- 📍 **Locais de risco:** Alguns endereços têm taxa de fatalidade muito maior
- 🕐 **Hora crítica:** 18h apresenta maior volume de acidentes
- 📅 **Dia crítico:** Sábados têm 8.64% de taxa de fatalidade (vs 6.81% qui, 7.02% sex)
- 🔄 **Sazonalidade semanal:** Padrão de 7 dias confirmado

---

## 📝 Changelog: Versão 2.0 (Nov 2025)

### ✨ Novas Funcionalidades
- ✅ **Etapa 4:** Análise de Outliers (Z-Score + IQR)
- ✅ **Etapa 7:** Decomposição de Série Temporal (Tendência/Sazonalidade/Resíduos)
- ✅ **Etapa 8:** Testes de Estacionariedade (ADF + KPSS + ACF/PACF)
- ✅ **Etapa 13:** Análise de Resíduos Expandida (Shapiro-Wilk + Q-Q Plot)
- ✅ **Etapa 14:** Detecção de Anomalias (Isolation Forest)
- ✅ **Etapa 15:** Testes Estatísticos Finais (Chi² + Kappa)

### 🔄 Melhorias
- ⬆️ **Dataset:** 45k → 817k registros (+1700%)
- ⬆️ **Código:** 500 → 900+ linhas (+80%)
- ⬆️ **Arquivos gerados:** 6 → 15+ (+150%)
- ⬆️ **Importações:** 12 → 22 bibliotecas (+83%)
- ⬆️ **Etapas:** 12 → 15 completas (+25%)

### 🐛 Correções
- ✅ **Importações corrigidas:** `chi2_contingency` de `scipy.stats` (não `sklearn.metrics`)
- ✅ **ADF/KPSS corrigidos:** De `statsmodels.tsa.stattools` (não `scipy.stats`)
- ✅ **Figsize corrigido:** Etapa 7 agora usa `plt.subplots()` ao invés de `decomposicao.plot()`

### 📊 Resultados Atualizados
- 🔄 **AUC-ROC RF:** 0.9862 → 0.9861
- 🔄 **Recall RF:** 16.68% → 18.37% (melhoria!)
- 🔄 **Feature #1:** dia_semana → endereco_id
- 🔄 **SVM:** 0.0342 → 0.1809 (grande correção)

---

## 🔮 Próximos Passos

### Em Planejamento
- [ ] Adicionar mais 2 datasets (meta: 5M+ registros)
- [ ] Implementar Redes Neurais (MLP/LSTM)
- [ ] Dashboard interativo (Streamlit/Dash)

### Melhorias Técnicas
- [ ] Feature engineering avançado (clustering de endereços)
- [ ] Ensemble stacking (RF + LR + XGBoost)
- [ ] Calibração de probabilidades
- [ ] Cross-validation temporal

### Deploy
- [ ] API REST para servir modelo
- [ ] Containerização (Docker)
- [ ] CI/CD pipeline
- [ ] Monitoramento de drift

---

## 👨‍💻 Autor

**Octavio Augusto Arruda dos Prazeres**  
Engenharia de Software | Universidade UNA
📧 Email: arrudaoctavio178@gmail.com 
🔗 LinkedIn: www.linkedin.com/in/octavio-prazeres
📂 GitHub: https://github.com/SecreTavin

---

## 📚 Referências

1. **He, H., & Garcia, E. A.** (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

2. **Cleveland, R. B., et al.** (1990). STL: A seasonal-trend decomposition. *Journal of Official Statistics*, 6(1), 3-73.

3. **Dickey, D. A., & Fuller, W. A.** (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427-431.

4. **Kwiatkowski, D., et al.** (1992). Testing the null hypothesis of stationarity. *Journal of Econometrics*, 54(1-3), 159-178.

5. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). Isolation Forest. *IEEE International Conference on Data Mining*, 413-422.

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos como parte do trabalho A3 da disciplina de Inteligência Artificial.

---

**📅 Última Atualização:** 15 de Novembro de 2025  
**🏷️ Versão:** 2.0 - Completa com 15 Etapas  
**📊 Status:** ✅ Pronto para Apresentação A3
