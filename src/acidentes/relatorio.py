"""Monta reports/RELATORIO.md a partir das métricas gravadas.

Nenhum número do relatório é digitado à mão: se o pipeline mudar, o texto muda junto.
"""

import json

from . import config
from .serie import ALVOS


def _pct(x: float) -> str:
    """0.072 -> "7,2%"."""
    return f"{x:.1%}".replace(".", ",")


def _num(x: float, casas: int = 3) -> str:
    """Número com vírgula decimal."""
    return f"{x:.{casas}f}".replace(".", ",")


def _mil(n: int) -> str:
    """486982 -> "486.982"."""
    return f"{n:,}".replace(",", ".")


def gerar() -> str:
    """Escreve `reports/RELATORIO.md` e devolve o caminho.

    Depende dos JSONs gravados por `treinar` e `prever`.
    """
    clf = json.loads((config.RELATORIOS / "metricas_classificacao.json").read_text(encoding="utf-8"))
    serie = json.loads((config.RELATORIOS / "metricas_serie.json").read_text(encoding="utf-8"))
    div, principal = clf["divisao"], clf["modelos"][clf["principal"]]
    contexto = clf["modelos"]["Boosting (só contexto)"]

    linhas = [
        "# Relatório de resultados",
        "",
        "Gerado por `acidentes relatorio`. Não edite à mão.",
        "",
        "## 1. Risco de fatalidade",
        "",
        f"Treino em {div['treino']['anos']} ({_mil(div['treino']['n'])} acidentes), limiar escolhido "
        f"em {div['validacao']['anos']} ({_mil(div['validacao']['n'])}) e teste de "
        f"{div['teste']['de']} a {div['teste']['ate']} ({_mil(div['teste']['n'])}). "
        f"No teste, {_pct(principal['prevalencia'])} dos acidentes tiveram morte.",
        "",
        "| Modelo | PR-AUC | ROC-AUC | Brier | Precisão | Recall | F1 | Captura no top 10% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for nome, m in clf["modelos"].items():
        linhas.append(
            f"| {nome} | {_num(m['pr_auc'])} | {_num(m['roc_auc'])} | {_num(m['brier'], 4)} | "
            f"{_pct(m['precisao'])} | {_pct(m['recall'])} | {_num(m['f1'])} | "
            f"{_pct(m['captura_top10'])} |")
    por_ano = ", ".join(f"{ano}: {_num(v)}" for ano, v in principal["por_ano"].items())
    linhas += [
        "",
        f"Um classificador que sorteia teria PR-AUC igual à prevalência, {_num(principal['prevalencia'])}. "
        "Precisão, recall e F1 usam o limiar que maximiza F1 na validação.",
        "",
        f"Leitura do modelo principal ({clf['principal']}):",
        "",
        f"* Os 10% de acidentes com maior risco previsto concentram {_pct(principal['captura_top10'])} "
        f"dos fatais; dentro desse grupo, {_pct(principal['precisao_top10'])} têm morte.",
        f"* O desempenho é estável entre os anos de teste (PR-AUC {por_ano}).",
        f"* Sem o tipo do acidente e o número de veículos, só com local e horário, a PR-AUC é "
        f"{_num(contexto['pr_auc'])}, contra {_num(principal['pr_auc'])} do modelo principal.",
        "",
        f"![Precisão × recall]({clf['figuras']['pr']})",
        "",
        f"![Calibração]({clf['figuras']['calibracao']})",
        "",
        f"![Importância]({clf['figuras']['importancia']})",
        "",
        "## 2. Previsão diária",
        "",
        f"Backtest de origem móvel: {serie['origens']} previsões de {serie['horizonte']} dias, "
        f"uma por semana, no último ano de dados (até {serie['ate']}). MASE abaixo de 1 "
        "significa erro menor que repetir a semana anterior.",
        "",
    ]
    for alvo, r in serie["alvos"].items():
        linhas += [
            f"**{ALVOS[alvo]}** (média do último ano: {_num(r['media_diaria'], 1)})",
            "",
            "| Modelo | MAE | MASE |",
            "| --- | --- | --- |",
            *[f"| {n}{' ✔' if n == r['melhor'] else ''} | {_num(m['mae'], 1)} | {_num(m['mase'], 2)} |"
              for n, m in r["modelos"].items()],
            "",
            f"![Backtest de {alvo}]({serie['figuras'][alvo]})",
            "",
        ]
    linhas += [
        "### Próximos 7 dias",
        "",
        "Intervalo de 80% tirado dos erros do backtest do modelo escolhido.",
        "",
        "| Data | Dia | Acidentes | Intervalo | Mortes | Intervalo |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for p in serie["previsao"]:
        linhas.append(
            f"| {p['data']} | {p['dia_semana']} | {p['acidentes']:.0f} | "
            f"{p['acidentes_p10']:.0f}–{p['acidentes_p90']:.0f} | {p['mortos']:.0f} | "
            f"{p['mortos_p10']:.0f}–{p['mortos_p90']:.0f} |")
    linhas.append("")

    destino = config.RELATORIOS / "RELATORIO.md"
    destino.write_text("\n".join(linhas), encoding="utf-8")
    print(f"relatório -> {destino}")
    return str(destino)
