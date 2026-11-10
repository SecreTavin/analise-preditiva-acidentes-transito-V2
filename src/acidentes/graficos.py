"""Estilo único para as figuras do relatório."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

AZUL, LARANJA, VERDE = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRADE, "axes.labelcolor": TINTA_2, "text.color": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def salvar(fig, nome: str) -> str:
    """Grava a figura em `reports/figuras` e devolve o caminho relativo ao relatório."""
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = config.FIGURAS / nome
    fig.savefig(caminho, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return f"figuras/{nome}"
