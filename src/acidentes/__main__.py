"""Linha de comando: `acidentes <etapa>` ou `python -m acidentes <etapa>`."""

import argparse


def main() -> None:
    """Executa a etapa pedida; `tudo` roda todas, na ordem do pipeline."""
    parser = argparse.ArgumentParser(prog="acidentes", description=__doc__)
    parser.add_argument("etapa", choices=["baixar", "ingerir", "treinar", "prever",
                                          "relatorio", "tudo"])
    etapa = parser.parse_args().etapa

    # Imports tardios: `baixar` não deve exigir scikit-learn carregado.
    if etapa in ("baixar", "tudo"):
        from .baixar import baixar
        baixar()
    if etapa in ("ingerir", "tudo"):
        from .ingestao import ingerir
        ingerir()
    if etapa in ("treinar", "tudo"):
        from .classificacao import treinar
        treinar()
    if etapa in ("prever", "tudo"):
        from .serie import prever
        prever()
    if etapa in ("relatorio", "tudo"):
        from .relatorio import gerar
        gerar()


if __name__ == "__main__":
    main()
