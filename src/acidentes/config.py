"""Caminhos e decisões fixas do projeto, num lugar só."""

import os
from pathlib import Path

RAIZ = Path(os.environ.get("ACIDENTES_RAIZ", Path(__file__).resolve().parents[2]))
DADOS_BRUTOS = RAIZ / "data" / "raw"
DADOS_PROCESSADOS = RAIZ / "data" / "processed"
PARQUET = DADOS_PROCESSADOS / "acidentes.parquet"
MODELOS = RAIZ / "models"
RELATORIOS = RAIZ / "reports"
FIGURAS = RELATORIOS / "figuras"

# Arquivos "agrupados por ocorrência" do portal de dados abertos da PRF.
# O portal troca esses links de tempos em tempos; se um download falhar,
# confira em https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
# O esquema de 30 colunas é estável desde 2017, por isso o corte.
FONTES_DRIVE = {
    2017: "1HPLWt5f_l4RIX3tKjI4tUXyZOev52W0N",
    2018: "1cM4IgGMIiR-u4gBIH5IEe3DcvBvUzedi",
    2019: "1pN3fn2wY34GH6cY-gKfbxRJJBFE0lb_l",
    2020: "1esu6IiH5TVTxFoedv6DBGDd01Gvi8785",
    2021: "12xH8LX9aN2gObR766YN3cMcuycwyCJDz",
    2022: "1PRQjuV5gOn_nn6UNvaJyVURDIfbSAK4-",
    2023: "1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq",
    2024: "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO",
    2025: "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR",
    2026: "1A3IirNm0AzRaSosA1IS94DOVmvKsn0Ol",
}

# Divisão temporal: o modelo nunca vê o futuro do que está sendo avaliado.
ANO_FIM_TREINO = 2023  # treino: até este ano, inclusive
ANO_VALIDACAO = 2024  # escolha de limiar
# teste: tudo depois de ANO_VALIDACAO

SEMENTE = 42
