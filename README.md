# Acidentes em rodovias federais: risco de morte e previsão de volume

Versão 3 do projeto de análise preditiva de acidentes de trânsito, refeita do
zero. A versão anterior, do início da faculdade, está preservada na tag
[`v2-original`](https://github.com/SecreTavin/analise-preditiva-acidentes-transito-V2/tree/v2-original). A versão anterior relatava AUC-ROC de 0,98; esse
número vinha de um vazamento entre fontes de dados, explicado em
[docs/diagnostico-v2.md](docs/diagnostico-v2.md).

Esta versão usa só os dados abertos da PRF (687 mil acidentes, de 2017 a
setembro de 2026) e responde a duas perguntas:

1. **Dado o que se sabe quando um acidente é comunicado, qual o risco de haver morte?**
2. **Quantos acidentes e mortes esperar nos próximos sete dias?**

Os resultados estão em [reports/RELATORIO.md](reports/RELATORIO.md), gerado pelo
próprio pipeline.

## Como rodar

```bash
uv venv && uv pip install -e ".[dev]"
```

```bash
.venv/bin/acidentes tudo
```

`tudo` executa as etapas em ordem; cada uma também roda sozinha:

| Etapa | O que faz | Saída |
| --- | --- | --- |
| `baixar` | Baixa um zip por ano do portal da PRF (cerca de 35 MB no total) | `data/raw/` |
| `ingerir` | Limpa, tipa e harmoniza categorias | `data/processed/acidentes.parquet` |
| `treinar` | Treina e avalia os classificadores | `reports/metricas_classificacao.json`, figuras, `models/` |
| `prever` | Backtest e previsão de 7 dias | `reports/metricas_serie.json`, `reports/previsao_7_dias.csv` |
| `relatorio` | Monta o relatório a partir das métricas | `reports/RELATORIO.md` |

O pipeline inteiro leva pouco mais de um minuto num notebook, quase todo no
treino. Testes: `.venv/bin/pytest`.

### Usar o modelo treinado

`treinar` grava o modelo principal em `models/boosting_chamado.joblib`. Ele
recebe uma tabela com as colunas de `features.CHAMADO` e devolve a
probabilidade de morte:

```python
import joblib
from acidentes import config
from acidentes.ingestao import carregar

modelo = joblib.load(config.MODELOS / "boosting_chamado.joblib")
risco = modelo.predict_proba(carregar().tail(5))[:, 1]
```

## Decisões

**Uma fonte só.** Os arquivos "agrupados por ocorrência" da PRF têm o mesmo
esquema de 30 colunas desde 2017. RENAINF e RENAEST ficaram de fora: na V2
entraram sem data nem gravidade, e foi isso que gerou o vazamento.

**O que o modelo pode ver.** Só o que existe quando a ocorrência é comunicada:
local (UF, município, BR, km, coordenadas), momento (hora, dia da semana, mês, fase do
dia), via (tipo de pista, traçado, uso do solo), clima, tipo do acidente e
número de veículos. Contagens de vítimas, classificação e causa são apuradas
depois e estão numa lista de colunas proibidas em
[features.py](src/acidentes/features.py), conferida por teste. O município tem
milhares de valores e entra como a taxa de fatalidade observada no treino,
calculada com ajuste cruzado.

**Divisão temporal.** Treino em 2017–2023, escolha de limiar em 2024, teste em
2025–2026. O modelo é sempre avaliado em datas posteriores às que viu.

**Métricas para classe rara.** Cerca de 7% dos acidentes têm morte. Acurácia e
AUC-ROC escondem isso; o relatório usa PR-AUC, calibração e a fração dos fatais
capturada nos 10% de maior risco.

**Previsão só vale se bater a ingênua.** Cada modelo de série é comparado, em 52
previsões semanais fora da amostra, com "repetir a semana passada".

## Limites conhecidos

* O tipo do acidente é a variável mais forte e é registrado pelo policial no
  local. O modelo "só contexto" mostra quanto sobra sem ele.
* A base só tem acidentes atendidos pela PRF em rodovias federais. Não vale para
  vias urbanas nem estaduais.
* A PRF recodificou os tipos de acidente em 2021; as categorias foram agrupadas
  em [ingestao.py](src/acidentes/ingestao.py), com perda de detalhe.
* A previsão diária não conhece feriados, e erra neles.
* Os links do Google Drive em [config.py](src/acidentes/config.py) mudam quando a
  PRF republica os arquivos; `baixar` falha com mensagem clara se isso acontecer.

## Estrutura

```
src/acidentes/
  config.py         caminhos, fontes e anos da divisão
  baixar.py         download dos zips
  ingestao.py       CSV bruto -> Parquet tipado
  features.py       colunas permitidas, colunas proibidas, divisão temporal
  classificacao.py  modelos, limiar, métricas, figuras
  serie.py          série diária, backtest, previsão
  relatorio.py      RELATORIO.md a partir dos JSONs
  graficos.py       estilo das figuras
tests/              ingestão, vazamento, divisão, métricas, backtest
docs/               diagnóstico da versão anterior
reports/            resultados gerados
```

Fonte dos dados: [Dados abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf).
