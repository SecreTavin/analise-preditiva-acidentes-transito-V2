import numpy as np
import pandas as pd
import pytest

from acidentes import features, serie
from acidentes.classificacao import avaliar, melhor_limiar
from acidentes.ingestao import limpar


def _bruto(**sobrescreve) -> pd.DataFrame:
    linha = {
        "id": "1e+05", "data_inversa": "2021-03-07", "dia_semana": "domingo",
        "horario": "23:15:00", "uf": "MG", "br": "381", "km": "452,5",
        "municipio": "BETIM", "causa_acidente": "Velocidade Incompatível",
        "tipo_acidente": "Colisão lateral mesmo sentido",
        "classificacao_acidente": "Com Vítimas Fatais", "fase_dia": "Plena Noite",
        "sentido_via": "Crescente", "condicao_metereologica": "Chuva",
        "tipo_pista": "Dupla", "tracado_via": "Declive;Curva", "uso_solo": "Sim",
        "pessoas": "3", "mortos": "1", "feridos_leves": "1", "feridos_graves": "0",
        "ilesos": "1", "ignorados": "0", "feridos": "1", "veiculos": "2",
        "latitude": "-19,96", "longitude": "-44,19", "regional": "SPRF-MG",
        "delegacia": "DEL01-MG", "uop": "UOP01-DEL01-MG",
    }
    return pd.DataFrame([{**linha, **sobrescreve}])


def test_limpar_converte_tipos_e_harmoniza():
    df = limpar(_bruto()).iloc[0]
    assert df["id"] == 100_000  # notação científica do arquivo original
    assert df["km"] == 452.5
    assert (df["hora"], df["dia_semana"], df["ano"]) == (23, 6, 2021)
    assert df["tipo_acidente"] == "Colisão lateral"
    assert df["tracado_curva"] == 1 and df["tracado_declive"] == 1 and df["tracado_reta"] == 0
    assert df["fatal"] == 1


def test_limpar_descarta_coordenada_fora_do_brasil():
    df = limpar(_bruto(latitude="19,96", longitude="44,19", mortos="0")).iloc[0]
    assert np.isnan(df["latitude"]) and np.isnan(df["longitude"])
    assert df["fatal"] == 0


def test_tracado_casa_o_item_inteiro():
    df = limpar(_bruto(tracado_via="Retorno Regulamentado;Aclive")).iloc[0]
    assert df["tracado_retorno_regulamentado"] == 1 and df["tracado_aclive"] == 1
    assert df["tracado_reta"] == 0 and df["tracado_declive"] == 0


@pytest.mark.parametrize("nome", features.CONJUNTOS)
def test_features_nao_incluem_desfecho(nome):
    usadas = features.conjunto(nome).todas
    disponiveis = set(limpar(_bruto()).columns)
    assert not set(usadas) & set(features.VAZAMENTO)
    assert set(usadas) <= disponiveis
    # A lista de proibidas só protege se nomear colunas que existem de fato.
    assert set(features.VAZAMENTO) <= disponiveis


def test_conjunto_com_vazamento_e_recusado(monkeypatch):
    sujo = features.Conjunto(("uf",), (), ("mortos",))
    monkeypatch.setitem(features.CONJUNTOS, "sujo", sujo)
    with pytest.raises(ValueError, match="mortos"):
        features.conjunto("sujo")


def test_divisao_temporal_nao_se_sobrepoe():
    df = pd.DataFrame({"ano": [2017, 2023, 2024, 2025, 2026]})
    treino, validacao, teste = features.dividir(df)
    assert treino["ano"].max() < validacao["ano"].min() <= validacao["ano"].max() < teste["ano"].min()
    assert len(treino) + len(validacao) + len(teste) == len(df)


def test_limiar_e_metricas():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    prob = np.array([.1, .2, .1, .9, .8, .3, .1, .2, .1, .7])
    limiar = melhor_limiar(y, prob)
    m = avaliar(y, prob, limiar)
    assert 0.3 < limiar <= 0.7
    assert m["precisao"] == m["recall"] == 1.0
    assert m["pr_auc"] == pytest.approx(1.0)


def test_serie_diaria_preenche_dias_sem_acidente():
    df = pd.DataFrame({"id": [1, 2, 3], "mortos": [0, 2, 1],
                       "data": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"])})
    diaria = serie.serie_diaria(df)
    assert diaria["acidentes"].tolist() == [2, 0, 1]
    assert diaria["mortos"].tolist() == [2, 0, 1]


def test_backtest_serie_perfeitamente_semanal():
    valores = np.tile([10., 12, 11, 13, 20, 25, 22], 120)
    resultado = serie.backtest(valores, origens=8)
    assert resultado["Ingênua sazonal"]["mae"] == 0
    assert resultado["Média de 4 semanas"]["mae"] == 0
    assert len(resultado["Ingênua sazonal"]["erros"]) == 8 * serie.HORIZONTE


def test_backtest_recusa_serie_curta():
    with pytest.raises(ValueError, match="curta"):
        serie.backtest(np.ones(60), origens=8)


def test_top10_e_exato_com_empates():
    y = np.array([1] * 5 + [0] * 95)
    m = avaliar(y, np.full(100, 0.5), limiar=0.5)
    assert m["captura_top10"] == 1.0  # os 10 primeiros, não os 100 empatados
