"""Previsão diária de acidentes e de mortes, validada por backtest.

Um modelo só entra no relatório como "previsão" se bater a referência
ingênua em dados que ele não viu.
"""

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config, graficos
from .ingestao import carregar

HORIZONTE = 7
ORIGENS = 52  # um ano de previsões semanais
ALVOS = {"acidentes": "Acidentes por dia", "mortos": "Mortes por dia"}
DIAS = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
# Histórico usado pelo ETS: dois anos bastam para o ciclo semanal e deixam o ajuste rápido.
_JANELA_ETS = 730


def serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega acidentes em contagens por dia, com zero nos dias sem registro.

    Args:
        df: precisa das colunas `data`, `id` e `mortos`.

    Returns:
        Índice diário contínuo com as colunas `acidentes` e `mortos`.
    """
    diaria = df.groupby("data").agg(acidentes=("id", "size"), mortos=("mortos", "sum"))
    dias = pd.date_range(diaria.index.min(), diaria.index.max(), freq="D")
    return diaria.reindex(dias, fill_value=0).astype(float)


# Os três previsores têm a mesma assinatura: recebem o histórico e o número de
# dias à frente, e devolvem um vetor com `h` previsões.

def ingenua_sazonal(hist: np.ndarray, h: int) -> np.ndarray:
    """Repete a última semana: a referência que qualquer modelo precisa bater."""
    return np.resize(hist[-7:], h)


def media_dia_semana(hist: np.ndarray, h: int, semanas: int = 4) -> np.ndarray:
    """Média das últimas `semanas` ocorrências de cada dia da semana."""
    return np.resize(hist[-7 * semanas:].reshape(semanas, 7).mean(axis=0), h)


def ets(hist: np.ndarray, h: int) -> np.ndarray:
    """Suavização exponencial com sazonalidade semanal aditiva, sem tendência."""
    with warnings.catch_warnings():
        # O otimizador avisa quando para perto do ótimo; a previsão ainda é válida.
        warnings.simplefilter("ignore", ConvergenceWarning)
        ajuste = ExponentialSmoothing(hist[-_JANELA_ETS:], seasonal="add", seasonal_periods=7,
                                      initialization_method="estimated").fit()
    return np.asarray(ajuste.forecast(h))


MODELOS = {
    "Ingênua sazonal": ingenua_sazonal,
    "Média de 4 semanas": media_dia_semana,
    "ETS sazonal": ets,
}


def backtest(valores: np.ndarray, h: int = HORIZONTE, origens: int = ORIGENS) -> dict:
    """Avalia cada modelo por origem móvel: prevê `h` dias, avança `h` dias, repete.

    Args:
        valores: a série diária completa.
        h: dias previstos a cada origem.
        origens: quantas previsões fazer, contadas a partir do fim da série.

    Returns:
        Por modelo: `mae`, `mase`, e os vetores `erros` e `previsoes` do
        período avaliado. MASE divide o erro pelo erro médio de repetir a
        semana anterior no histórico antes do backtest; abaixo de 1 é melhor
        que a referência.

    Raises:
        ValueError: se a série for curta demais para o backtest pedido.
    """
    inicio = len(valores) - origens * h
    if inicio < 28:
        raise ValueError("série curta demais: faltam 4 semanas de histórico antes do backtest")
    escala = np.abs(valores[7:inicio] - valores[:inicio - 7]).mean()
    erros = {nome: [] for nome in MODELOS}
    previsoes = {nome: [] for nome in MODELOS}
    for corte in range(inicio, len(valores), h):
        hist, real = valores[:corte], valores[corte:corte + h]
        for nome, modelo in MODELOS.items():
            prev = modelo(hist, h)
            erros[nome].extend(real - prev)
            previsoes[nome].extend(prev)
    return {
        nome: {
            "mae": float(np.abs(e).mean()),
            # Série sem variação semanal não tem escala; MASE fica indefinido.
            "mase": float(np.abs(e).mean() / escala) if escala > 0 else float("nan"),
            "erros": np.asarray(e),
            "previsoes": np.asarray(previsoes[nome]),
        }
        for nome, e in erros.items()
    }


def prever() -> dict:
    """Roda o backtest, prevê os próximos 7 dias com o melhor modelo e grava tudo.

    Returns:
        O mesmo dicionário gravado em `reports/metricas_serie.json`.
    """
    diaria = serie_diaria(carregar(["data", "id", "mortos"]))
    saida = {"ate": f"{diaria.index.max():%d/%m/%Y}", "horizonte": HORIZONTE,
             "origens": ORIGENS, "alvos": {}, "figuras": {}}
    futuro = pd.date_range(diaria.index.max() + pd.Timedelta(days=1), periods=HORIZONTE)
    tabela = pd.DataFrame({"data": futuro.strftime("%d/%m/%Y"),
                           "dia_semana": [DIAS[d] for d in futuro.day_of_week]})
    plt = graficos.plt

    for alvo, titulo in ALVOS.items():
        valores = diaria[alvo].to_numpy()
        resultado = backtest(valores)
        melhor = min(resultado, key=lambda n: resultado[n]["mae"])
        prev = MODELOS[melhor](valores, HORIZONTE)
        # Intervalo empírico: quantis dos erros do backtest do próprio modelo.
        baixo, alto = np.quantile(resultado[melhor]["erros"], [0.1, 0.9])
        tabela[alvo] = prev.round(1)
        tabela[f"{alvo}_p10"] = np.clip(prev + baixo, 0, None).round(1)
        tabela[f"{alvo}_p90"] = (prev + alto).round(1)
        saida["alvos"][alvo] = {
            "media_diaria": float(valores[-365:].mean()),
            "melhor": melhor,
            "modelos": {n: {"mae": r["mae"], "mase": r["mase"]} for n, r in resultado.items()},
        }

        janela = 8 * 7
        datas = diaria.index[-janela:]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(datas, valores[-janela:], color=graficos.TINTA_2, label="Observado")
        ax.plot(datas, resultado[melhor]["previsoes"][-janela:], color=graficos.AZUL,
                label=f"Previsto até 7 dias antes ({melhor})")
        ax.set(title=f"{titulo}: últimas 8 semanas do backtest", ylabel=titulo)
        ax.legend()
        fig.autofmt_xdate()
        saida["figuras"][alvo] = graficos.salvar(fig, f"backtest_{alvo}.png")

    config.RELATORIOS.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(config.RELATORIOS / "previsao_7_dias.csv", index=False)
    saida["previsao"] = tabela.to_dict(orient="records")
    (config.RELATORIOS / "metricas_serie.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    for alvo, r in saida["alvos"].items():
        print(f"{alvo}: melhor = {r['melhor']} | " + " | ".join(
            f"{n} MAE {m['mae']:.1f} (MASE {m['mase']:.2f})" for n, m in r["modelos"].items()))
    return saida
