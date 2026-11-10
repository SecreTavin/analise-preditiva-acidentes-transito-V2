# O que estava errado na versão 2

Diagnóstico do repositório
[analise-preditiva-acidentes-transito-V2](https://github.com/SecreTavin/analise-preditiva-acidentes-transito-V2).
O banco MySQL original não existe mais; tudo aqui foi deduzido dos scripts, dos
CSVs de resultado e dos gráficos da V2. Nenhum número da V2 foi reexecutado.

Os arquivos citados abaixo saíram da branch principal nesta versão. Continuam
acessíveis na tag
[`v2-original`](https://github.com/SecreTavin/analise-preditiva-acidentes-transito-V2/tree/v2-original/A3_finalizado).

## 1. O modelo separava fontes de dados, não acidentes fatais

A V2 juntou três fontes numa tabela `acidente` de 817.958 linhas. Só as cerca de
45.590 linhas da PRF tinham data, hora e gravidade reais. As outras 772 mil
tinham `gravidade = 'GRAVE'` fixo e nenhuma `data_hora`.

| Evidência | Onde aparece na V2 |
| --- | --- |
| 94,4% da base com a mesma gravidade, `GRAVE` em maiúsculas | Gráfico de pizza em `06_visualizacao_serie.png` |
| Os gráficos por hora e por dia somam cerca de 45 mil acidentes, não 818 mil | Mesmo arquivo |
| `endereco_id` e `numero_vitimas` têm exatamente 45.590 "outliers" cada | `04_analise_outliers.csv` |
| A regressão logística marca 13.701 de 245.388 como fatais (5,58%); a PRF é 5,57% da base | `matriz_confusao_regressão_logística.csv` |
| Recall de 100% com zero falso negativo | Mesmo arquivo |
| `ano` com 8,6% de importância numa base em que todo acidente datado é de 2025 | `importancia_features.csv` |

Todos os 3.357 acidentes fatais estavam no bloco da PRF. Qualquer variável que
fosse nula fora dele (`ano`, `mes`, `dia`, `periodo_dia`, `endereco_id`)
bastava para "acertar". O AUC-ROC de 0,98 mede isso.

Dentro do bloco da PRF, onde o problema é de verdade, a taxa de fatalidade era
7,4% (e não 0,41%) e o Random Forest achava 185 de 1.007 fatais.

## 2. A previsão de três dias era um total histórico

`previsoes_proximos_3_dias.csv` dá 6.068 acidentes para a quinta-feira
13/11/2025. É a soma de todas as quintas de janeiro a agosto, o mesmo valor da
barra "Qui" no gráfico por dia da semana.

## 3. Etapas que não mediam nada

| Etapa da V2 | Problema |
| --- | --- |
| Shapiro-Wilk nos "resíduos" | Resíduo de classificação binária só vale −1, 0 ou 1; nunca será normal |
| Qui-quadrado entre previsto e real | Dá p ≈ 0 para qualquer modelo melhor que o acaso |
| Isolation Forest com `contamination=0.05` | Devolve 5% de anomalias por construção |
| SVM com `max_iter=1000` | Não convergiu; AUC de 0,18 é um modelo invertido, não um modelo fraco |
| `LabelEncoder` em `descricao` | Transforma texto livre em número ordinal sem sentido |
| Divisão aleatória 70/30 | Mistura passado e futuro; com `dia` e `mes` como variáveis, infla o resultado |

## 4. Repositório

* Senha do MySQL em texto puro em vários scripts de um repositório público.
* `random_forest_modelo.pkl` de 46 MB versionado.
* README cita arquivos e pastas que não existem.

## Como a V3 responde

| Problema | Resposta | Onde |
| --- | --- | --- |
| Fontes misturadas | Uma fonte só, com esquema estável (PRF, 2017 em diante) | `config.py`, `ingestao.py` |
| Variáveis que entregam o desfecho | Lista explícita de colunas proibidas, conferida por teste | `features.py`, `tests/` |
| Divisão aleatória | Treino, validação e teste em ordem cronológica | `features.dividir` |
| Acurácia e AUC-ROC em classe rara | PR-AUC, calibração e captura no decil de maior risco | `classificacao.avaliar` |
| Previsão sem validação | Backtest de origem móvel contra referência ingênua | `serie.backtest` |
| Números digitados no README | Relatório gerado a partir das métricas gravadas | `relatorio.py` |
| Senha e modelo no git | Sem banco com credencial; `data/` e `models/` ignorados | `.gitignore` |
