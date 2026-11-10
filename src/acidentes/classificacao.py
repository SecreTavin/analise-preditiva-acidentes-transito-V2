"""Classificador de risco de fatalidade, avaliado no futuro em relação ao treino."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from . import config, graficos
from .features import ALVO, conjunto, dividir
from .ingestao import carregar

PRINCIPAL = "Boosting (chamado)"
ARQUIVO_MODELO = "boosting_chamado.joblib"

# Na logística, hora/dia/mês entram como categorias: o efeito não é linear
# (madrugada e fim de tarde não ficam nas pontas de uma reta).
_CICLICAS = ("hora", "dia_semana", "mes")
_CONTINUAS = ("km", "latitude", "longitude", "veiculos")
# Categoria com menos ocorrências que isto no treino é agrupada como "rara".
_FREQUENCIA_MINIMA = 30


def modelo_logistico(nome_conjunto: str) -> Pipeline:
    """Regressão logística com one-hot: a referência simples e interpretável."""
    cj = conjunto(nome_conjunto)
    continuas = [c for c in cj.numericas if c in _CONTINUAS]
    binarias = [c for c in cj.numericas if c not in _CONTINUAS and c not in _CICLICAS]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=_FREQUENCIA_MINIMA),
         [*cj.categoricas, *cj.alta_cardinalidade, *_CICLICAS]),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), continuas),
        ("bin", "passthrough", binarias),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])


def modelo_boosting(nome_conjunto: str) -> Pipeline:
    """Gradient boosting por histogramas, com categorias nativas.

    Colunas de alta cardinalidade (município) viram a taxa de fatalidade
    observada no treino. O `TargetEncoder` faz isso com ajuste cruzado, então
    nenhuma linha é codificada com o próprio desfecho.
    """
    cj = conjunto(nome_conjunto)
    categoricas = list(cj.categoricas)
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               min_frequency=_FREQUENCIA_MINIMA), categoricas),
        ("alvo", TargetEncoder(cv=StratifiedKFold(5, shuffle=True, random_state=config.SEMENTE)),
         list(cj.alta_cardinalidade)),
        ("num", "passthrough", list(cj.numericas)),
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    # A parada antecipada usa 10% do treino sorteados, não um corte temporal.
    # Isso só decide o número de árvores; limiar e métricas vêm de anos posteriores.
    clf = HistGradientBoostingClassifier(
        categorical_features=categoricas, learning_rate=0.06, max_iter=400,
        early_stopping=True, validation_fraction=0.1, random_state=config.SEMENTE)
    return Pipeline([("pre", pre), ("clf", clf)])


def melhor_limiar(y: np.ndarray, prob: np.ndarray) -> float:
    """Limiar de probabilidade que maximiza F1. Escolhido na validação, nunca no teste."""
    precisao, recall, limiares = precision_recall_curve(y, prob)
    f1 = 2 * precisao * recall / np.clip(precisao + recall, 1e-12, None)
    # O último ponto da curva (recall 0) não tem limiar correspondente.
    return float(limiares[np.argmax(f1[:-1])])


def avaliar(y: np.ndarray, prob: np.ndarray, limiar: float) -> dict:
    """Métricas de um modelo num conjunto de avaliação.

    Args:
        y: desfecho real (0/1).
        prob: probabilidade prevista de fatalidade.
        limiar: corte acima do qual o acidente é marcado como de alto risco.

    Returns:
        Métricas de ordenação (PR-AUC, ROC-AUC), de calibração (Brier), do
        limiar (precisão, recall, F1) e do decil de maior risco.
    """
    previsto = prob >= limiar
    vp = int((previsto & (y == 1)).sum())
    precisao = vp / max(int(previsto.sum()), 1)
    recall = vp / max(int(y.sum()), 1)
    # Exatamente os 10% de maior risco, mesmo que haja probabilidades empatadas.
    topo = np.argsort(-prob, kind="stable")[:max(len(y) // 10, 1)]
    return {
        "n": int(len(y)),
        "prevalencia": float(y.mean()),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "limiar": limiar,
        "precisao": precisao,
        "recall": recall,
        "f1": 2 * precisao * recall / max(precisao + recall, 1e-12),
        # Leitura operacional: olhando só os 10% de maior risco, quanto se pega?
        "captura_top10": float(y[topo].sum() / max(y.sum(), 1)),
        "precisao_top10": float(y[topo].mean()),
    }


def _figuras(y: np.ndarray, probs: dict[str, np.ndarray], importancia: pd.Series) -> dict:
    """Grava as três figuras da classificação e devolve seus caminhos relativos."""
    plt = graficos.plt
    cores = [graficos.AZUL, graficos.LARANJA, graficos.VERDE]
    prevalencia = y.mean()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (nome, prob), cor in zip(probs.items(), cores):
        precisao, recall, _ = precision_recall_curve(y, prob)
        ax.plot(recall, precisao, color=cor,
                label=f"{nome} (PR-AUC {average_precision_score(y, prob):.3f})")
    ax.axhline(prevalencia, color=graficos.TINTA_2, linewidth=1, linestyle="--")
    ax.text(0.99, prevalencia - 0.015, f"sorteio: {prevalencia:.1%}", ha="right", va="top",
            color=graficos.TINTA_2)
    ax.set(xlabel="Recall (fração dos acidentes fatais encontrada)", ylabel="Precisão",
           title="Precisão × recall no teste", ylim=(0, 1), xlim=(0, 1))
    ax.legend()
    saida = {"pr": graficos.salvar(fig, "precisao_recall.png")}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    limite = 0
    for (nome, prob), cor in zip(probs.items(), cores):
        observado, previsto = calibration_curve(y, prob, n_bins=10, strategy="quantile")
        ax.plot(previsto, observado, marker="o", markersize=5, color=cor, label=nome)
        limite = max(limite, previsto.max(), observado.max())
    ax.plot([0, limite], [0, limite], color=graficos.TINTA_2, linewidth=1, linestyle="--")
    ax.set(xlabel="Probabilidade prevista", ylabel="Frequência observada",
           title="Calibração no teste (decis)")
    ax.legend()
    saida["calibracao"] = graficos.salvar(fig, "calibracao.png")

    topo = importancia.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(topo.index, topo.values, color=graficos.AZUL, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="Queda de PR-AUC ao embaralhar a variável",
           title="O que o modelo usa (importância por permutação)")
    saida["importancia"] = graficos.salvar(fig, "importancia.png")
    return saida


def treinar() -> dict:
    """Treina os três modelos, avalia no teste e grava métricas, figuras e o modelo principal.

    Returns:
        O mesmo dicionário gravado em `reports/metricas_classificacao.json`.
    """
    usadas = sorted({*conjunto("chamado").todas, ALVO, "ano", "data"})
    treino, validacao, teste = dividir(carregar(usadas))
    y_val, y_teste = validacao[ALVO].to_numpy(), teste[ALVO].to_numpy()
    anos_teste = teste["ano"].to_numpy()

    candidatos = {
        "Logística (chamado)": modelo_logistico("chamado"),
        PRINCIPAL: modelo_boosting("chamado"),
        "Boosting (só contexto)": modelo_boosting("contexto"),
    }
    resultados, probs, ajustados = {}, {}, {}
    for nome, modelo in candidatos.items():
        modelo.fit(treino, treino[ALVO])
        limiar = melhor_limiar(y_val, modelo.predict_proba(validacao)[:, 1])
        prob = modelo.predict_proba(teste)[:, 1]
        resultados[nome] = avaliar(y_teste, prob, limiar)
        resultados[nome]["por_ano"] = {
            int(ano): float(average_precision_score(y_teste[anos_teste == ano],
                                                    prob[anos_teste == ano]))
            for ano in np.unique(anos_teste)
        }
        probs[nome], ajustados[nome] = prob, modelo
        print(f"{nome}: PR-AUC {resultados[nome]['pr_auc']:.3f} | "
              f"ROC-AUC {resultados[nome]['roc_auc']:.3f} | "
              f"top 10% captura {resultados[nome]['captura_top10']:.1%}")

    # Importância medida no teste: quanto a PR-AUC cai ao embaralhar cada coluna.
    # O pré-processamento mantém uma coluna por variável, então dá para aplicá-lo
    # uma vez e permutar direto na entrada do classificador, sem recodificar a cada rodada.
    amostra = teste.sample(min(20_000, len(teste)), random_state=config.SEMENTE)
    codificada = ajustados[PRINCIPAL]["pre"].transform(amostra)
    perm = permutation_importance(
        ajustados[PRINCIPAL]["clf"], codificada, amostra[ALVO],
        scoring="average_precision", n_repeats=3, random_state=config.SEMENTE)
    importancia = pd.Series(perm.importances_mean,
                            index=codificada.columns).sort_values(ascending=False)

    config.MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(ajustados[PRINCIPAL], config.MODELOS / ARQUIVO_MODELO)

    saida = {
        "divisao": {
            "treino": {"anos": f"{treino['ano'].min()}–{treino['ano'].max()}", "n": len(treino)},
            "validacao": {"anos": str(config.ANO_VALIDACAO), "n": len(validacao)},
            "teste": {"de": f"{teste['data'].min():%d/%m/%Y}",
                      "ate": f"{teste['data'].max():%d/%m/%Y}", "n": len(teste)},
        },
        "principal": PRINCIPAL,
        "modelos": resultados,
        "importancia": importancia.round(5).to_dict(),
        "figuras": _figuras(y_teste, probs, importancia),
    }
    config.RELATORIOS.mkdir(parents=True, exist_ok=True)
    (config.RELATORIOS / "metricas_classificacao.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    return saida
