"""Lê os CSVs brutos da PRF, limpa e grava um único Parquet tipado."""

import io
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

from . import config

# A PRF recodificou os tipos de acidente em 2021. Sem harmonizar, o modelo
# treinado até 2023 veria categorias que mudam de nome no meio da série.
TIPOS_HARMONIZADOS = {
    "Colisão lateral mesmo sentido": "Colisão lateral",
    "Colisão lateral sentido oposto": "Colisão lateral",
    "Colisão com objeto estático": "Colisão com objeto",
    "Colisão com objeto em movimento": "Colisão com objeto",
    "Danos eventuais": "Outros",
    "Eventos atípicos": "Outros",
    "Sinistro pessoal de trânsito": "Outros",
}

# `tracado_via` vem como lista separada por ";" (ex.: "Aclive;Reta").
TRACADOS = [
    "Reta", "Curva", "Aclive", "Declive", "Interseção de Vias", "Rotatória",
    "Retorno Regulamentado", "Ponte", "Viaduto", "Túnel", "Em Obras",
    "Desvio Temporário",
]

_CONTAGENS = ["pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos",
              "ignorados", "feridos", "veiculos"]
_CATEGORICAS = ["uf", "br", "municipio", "causa_acidente", "tipo_acidente",
                "classificacao_acidente", "fase_dia", "sentido_via",
                "condicao_metereologica", "tipo_pista", "uso_solo"]


def coluna_tracado(tracado: str) -> str:
    """Nome da coluna 0/1 de um traçado: "Interseção de Vias" -> "tracado_intersecao_de_vias"."""
    sem_acento = unicodedata.normalize("NFKD", tracado).encode("ascii", "ignore").decode()
    return "tracado_" + sem_acento.lower().replace(" ", "_")


def _decimal(serie: pd.Series) -> pd.Series:
    """Converte texto com vírgula decimal ("452,5") em número; inválidos viram NaN."""
    return pd.to_numeric(serie.str.replace(",", ".", regex=False), errors="coerce")


def ler_zip(caminho: Path) -> pd.DataFrame:
    """Lê o CSV de dentro de um zip anual da PRF, com todas as colunas como texto."""
    with zipfile.ZipFile(caminho) as zf:
        conteudo = zf.read(zf.namelist()[0])
    return pd.read_csv(io.BytesIO(conteudo), sep=";", encoding="latin-1", dtype=str)


def limpar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos, harmoniza categorias e cria o alvo. Não descarta acidentes.

    Args:
        bruto: uma tabela no esquema de 30 colunas da PRF, tudo como texto.

    Returns:
        Uma linha por acidente, com data/hora decompostas, uma coluna 0/1 por
        traçado de via, contagens inteiras e `fatal` (1 se houve morte).
    """
    # Alguns ids vêm em notação científica ("1e+05") no arquivo da PRF.
    df = pd.DataFrame({"id": pd.to_numeric(bruto["id"]).astype("int64")})
    df["data_hora"] = pd.to_datetime(bruto["data_inversa"] + " " + bruto["horario"],
                                     format="%Y-%m-%d %H:%M:%S")
    df["data"] = df["data_hora"].dt.normalize()
    df["ano"] = df["data_hora"].dt.year.astype("int16")
    df["mes"] = df["data_hora"].dt.month.astype("int8")
    df["hora"] = df["data_hora"].dt.hour.astype("int8")
    # Derivado da data (0 = segunda): a coluna textual da PRF é redundante.
    df["dia_semana"] = df["data_hora"].dt.day_of_week.astype("int8")

    for col in _CATEGORICAS:
        df[col] = bruto[col].str.strip()
    df["tipo_acidente"] = df["tipo_acidente"].replace(TIPOS_HARMONIZADOS)
    df["km"] = _decimal(bruto["km"])

    # Coordenadas fora do Brasil são erro de digitação; viram ausentes.
    lat, lon = _decimal(bruto["latitude"]), _decimal(bruto["longitude"])
    valida = lat.between(-34, 6) & lon.between(-74, -34)
    df["latitude"] = lat.where(valida)
    df["longitude"] = lon.where(valida)

    # Os ";" nas pontas garantem casar o item inteiro, não um pedaço de outro.
    tracado = ";" + bruto["tracado_via"].fillna("") + ";"
    for nome in TRACADOS:
        df[coluna_tracado(nome)] = tracado.str.contains(f";{nome};", regex=False).astype("int8")

    for col in _CONTAGENS:
        df[col] = pd.to_numeric(bruto[col], errors="coerce").fillna(0).astype("int16")

    df["fatal"] = (df["mortos"] > 0).astype("int8")
    return df


def ingerir() -> pd.DataFrame:
    """Junta todos os zips de `data/raw`, limpa e grava `acidentes.parquet`.

    Raises:
        FileNotFoundError: se não houver zip baixado.
        ValueError: se o mesmo id de acidente aparecer em mais de um arquivo.
    """
    arquivos = sorted(config.DADOS_BRUTOS.glob("datatran*.zip"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum zip em {config.DADOS_BRUTOS}; rode `acidentes baixar`")
    df = pd.concat([limpar(ler_zip(a)) for a in arquivos], ignore_index=True)
    if df["id"].duplicated().any():
        raise ValueError("ids de acidente repetidos entre arquivos")
    df = df.sort_values("data_hora").reset_index(drop=True)
    config.DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PARQUET, index=False)
    print(f"{len(df):,} acidentes, {df['data'].min():%d/%m/%Y} a {df['data'].max():%d/%m/%Y}, "
          f"{df['fatal'].mean():.2%} com morte -> {config.PARQUET}")
    return df


def carregar(colunas: list[str] | None = None) -> pd.DataFrame:
    """Lê o Parquet processado; `colunas` restringe a leitura ao que for usado.

    Raises:
        FileNotFoundError: se a ingestão ainda não foi executada.
    """
    if not config.PARQUET.exists():
        raise FileNotFoundError(f"{config.PARQUET} não existe; rode `acidentes ingerir`")
    return pd.read_parquet(config.PARQUET, columns=colunas)
