"""Quais colunas o modelo pode ver, e a divisão temporal.

A regra é uma só: entra o que se sabe quando a ocorrência é comunicada.
Tudo o que é apurado depois (vítimas, classificação, causa) fica de fora,
porque descreve o desfecho que se quer prever.
"""

from dataclasses import dataclass

import pandas as pd

from . import config
from .ingestao import TRACADOS, coluna_tracado

ALVO = "fatal"


@dataclass(frozen=True)
class Conjunto:
    """Colunas de um conjunto de features, separadas pelo tratamento que pedem.

    Attributes:
        categoricas: poucas categorias; o boosting as trata nativamente.
        alta_cardinalidade: milhares de categorias; viram taxa do alvo (target encoding).
        numericas: usadas como estão.
    """

    categoricas: tuple[str, ...]
    alta_cardinalidade: tuple[str, ...]
    numericas: tuple[str, ...]

    @property
    def todas(self) -> list[str]:
        return [*self.categoricas, *self.alta_cardinalidade, *self.numericas]


_TRACADO_COLS = tuple(coluna_tracado(t) for t in TRACADOS)

# Onde e quando: disponível antes de qualquer informação sobre o acidente em si.
CONTEXTO = Conjunto(
    categoricas=("uf", "br", "fase_dia", "condicao_metereologica", "tipo_pista",
                 "sentido_via", "uso_solo"),
    alta_cardinalidade=("municipio",),
    numericas=("hora", "dia_semana", "mes", "km", "latitude", "longitude", *_TRACADO_COLS),
)

# O que quem liga costuma informar a mais: o tipo do acidente e quantos veículos.
CHAMADO = Conjunto(
    categoricas=(*CONTEXTO.categoricas, "tipo_acidente"),
    alta_cardinalidade=CONTEXTO.alta_cardinalidade,
    numericas=(*CONTEXTO.numericas, "veiculos"),
)

CONJUNTOS = {"contexto": CONTEXTO, "chamado": CHAMADO}

# Nunca podem virar feature. `pessoas` e `ilesos` são contagens fechadas após
# o atendimento; `causa_acidente` é atribuída na apuração.
VAZAMENTO = ("mortos", "feridos_leves", "feridos_graves", "feridos", "ilesos",
             "ignorados", "pessoas", "classificacao_acidente", "causa_acidente", ALVO)


def conjunto(nome: str) -> Conjunto:
    """Devolve o conjunto de features pedido, recusando qualquer coluna de desfecho.

    Raises:
        ValueError: se alguma coluna do conjunto estiver em `VAZAMENTO`.
    """
    escolhido = CONJUNTOS[nome]
    proibidas = set(escolhido.todas) & set(VAZAMENTO)
    if proibidas:
        raise ValueError(f"colunas com vazamento no conjunto {nome}: {sorted(proibidas)}")
    return escolhido


def dividir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separa treino, validação e teste em ordem cronológica, sem sobreposição.

    Os anos de corte vêm de `config`. Divisão aleatória misturaria passado e
    futuro e inflaria o resultado.
    """
    treino = df[df["ano"] <= config.ANO_FIM_TREINO]
    validacao = df[df["ano"] == config.ANO_VALIDACAO]
    teste = df[df["ano"] > config.ANO_VALIDACAO]
    return treino, validacao, teste
