#  Análise Preditiva de Acidentes de Trânsito

Projeto de Machine Learning para prever a gravidade de acidentes de trânsito usando dados reais. Desenvolvido como parte do curso de Engenharia de Software.
(será adicionado a esse repositório novas informações sobre alterações ou dados que ainda serão implementados).
(Trabalho ainda em desenvolvimento).
##  Sobre o Projeto

Neste projeto, eu desenvolvi um pipeline completo de análise de dados e machine learning para entender padrões em acidentes de trânsito e prever quais têm maior probabilidade de serem fatais.

Trabalhar com esses dados foi bem interessante porque vai além de números — cada registro representa uma situação real de trânsito. A ideia é usar IA para ajudar a identificar fatores de risco e, quem sabe, contribuir para políticas de segurança viária.

## O Que Eu Fiz

### 1- Coleta e Exploração dos Dados
- Base de dados com **45.590 acidentes** registrados entre janeiro e agosto de 2025. (será adicionado mais 2 datasets, somando assim mais de 5kk de dados).
- Conectei direto com MySQL usando SQLAlchemy
- Criei visualizações para entender padrões:
  - Quando mais acontecem acidentes (hora, dia, mês)
  - Distribuição de gravidade
  - Tipos de acidentes mais comuns
  - Heatmap mostrando gravidade por hora do dia

### 2- Engenharia de Features
Criei novas variáveis a partir dos dados originais:
- **Período do dia**: Madrugada, Manhã, Tarde ou Noite
- **Final de semana**: Se o acidente foi em sábado/domingo
- **Horário de pico**: Se foi entre 7h-9h ou 17h-19h

### 3- Preparação dos Dados
- Tratamento de valores faltantes
- Codificação de variáveis categóricas (Label Encoding)
- Normalização dos dados numéricos
- Separação em treino (70%) e teste (30%)

### 4- Modelagem e Otimização
Testei 3 algoritmos diferentes de ML:
- **Random Forest** 🌲
- **Regressão Logística** 📈
- **SVM** 🎯

Usei `RandomizedSearchCV` para encontrar os melhores hiperparâmetros de cada modelo. Isso economiza tempo comparado ao GridSearch, mas ainda garante bons resultados.

### 5- Avaliação
Analisei os modelos usando:
- **Acurácia**: Taxa de acertos geral
- **AUC-ROC**: Capacidade de distinguir entre acidentes fatais e não fatais
- **Matriz de Confusão**: Para ver onde o modelo erra
- **Feature Importance**: Quais variáveis mais influenciam nas previsões

## Resultados

O **Random Forest** foi o modelo de melhor desempenho. Ele conseguiu a melhor performance na identificação de acidentes fatais.

Alguns insights interessantes:
- 7.4% dos acidentes são classificados como fatais
- Horários específicos têm maior incidência de acidentes graves
- Certas características (tipo de acidente, localização, condições) são fortes preditores

## Tecnologias Utilizadas
- Python 3.12
- pandas & numpy (manipulação de dados)
- scikit-learn (machine learning)
- matplotlib & seaborn (visualizações)
- SQLAlchemy (conexão com banco de dados)
- MySQL (armazenamento de dados)


## Outputs

O código gera automaticamente:
- 3 arquivos PNG com visualizações
- 2 arquivos CSV com resultados detalhados
- Relatório no terminal com métricas e insights

## Aprendizados

Este projeto me ensinou muito sobre:
- Como lidar com dados desbalanceados (poucos casos fatais vs. muitos não fatais)
- A importância da engenharia de features
- Comparação justa entre diferentes algoritmos
- Como otimizar hiperparâmetros de forma eficiente
- Visualização de dados para storytelling

## Próximos Passos

Algumas melhorias que planejo implementar:
- [ ] Testar técnicas de balanceamento (SMOTE)
- [ ] Adicionar mais métricas (Precision, Recall, F1-Score)
- [ ] Implementar validação temporal
- [ ] Explorar modelos ensemble mais avançados
- [ ] Criar uma API para servir o modelo
- [ ] Adicionar mais datasets ao banco de dados

**Desenvolvido por:** Octavio Augusto Arruda dos Prazeres 

**Curso:** Engenharia de Software  

**Disciplina:** Inteligência Artificial  

**Ano:** 2025


