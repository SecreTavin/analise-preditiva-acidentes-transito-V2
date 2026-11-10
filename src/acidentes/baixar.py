"""Baixa os arquivos anuais da PRF para data/raw."""

import shutil
import urllib.request
import zipfile

from . import config

_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
_TEMPO_LIMITE = 120  # segundos por arquivo


def baixar(forcar: bool = False) -> None:
    """Baixa um zip por ano listado em `config.FONTES_DRIVE`.

    Args:
        forcar: baixa de novo mesmo que o arquivo já exista.

    Raises:
        RuntimeError: se um link não devolver um zip válido.
    """
    config.DADOS_BRUTOS.mkdir(parents=True, exist_ok=True)
    for ano, id_drive in config.FONTES_DRIVE.items():
        destino = config.DADOS_BRUTOS / f"datatran{ano}.zip"
        if destino.exists() and not forcar:
            print(f"{ano}: já existe")
            continue
        # Grava num arquivo temporário: um download interrompido não pode
        # deixar um zip pela metade com o nome final.
        parcial = destino.with_suffix(".parcial")
        try:
            with urllib.request.urlopen(_URL.format(id=id_drive), timeout=_TEMPO_LIMITE) as resposta, \
                    open(parcial, "wb") as saida:
                shutil.copyfileobj(resposta, saida)
            # O Drive devolve uma página HTML quando o link morre; falhar aqui
            # é melhor do que descobrir na ingestão.
            if not zipfile.is_zipfile(parcial):
                raise RuntimeError(f"{ano}: o link não devolveu um zip; atualize FONTES_DRIVE")
            parcial.replace(destino)
        finally:
            parcial.unlink(missing_ok=True)
        print(f"{ano}: {destino.stat().st_size / 1e6:.1f} MB")
