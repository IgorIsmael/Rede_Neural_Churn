"""Carregamento seguro e cacheado dos artefatos externos."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

LOGGER = logging.getLogger(__name__)


class ArtefactError(RuntimeError):
    """Erro legível relacionado a um artefato da aplicação."""


def _validate_model_file(path: Path) -> tuple[int, int]:
    """Valida o arquivo e devolve dados usados para invalidar o cache."""
    if not path.is_file():
        raise ArtefactError(f"Modelo não encontrado: {path.name}")
    stat = path.stat()
    if stat.st_size == 0:
        raise ArtefactError(f"O arquivo {path.name} está vazio. Envie novamente o artefato treinado.")
    try:
        with path.open("rb") as file:
            header = file.read(200)
    except OSError as exc:
        raise ArtefactError(f"O arquivo {path.name} existe, mas não pôde ser lido.") from exc
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ArtefactError(
            f"{path.name} é apenas um ponteiro do Git LFS, não o modelo. "
            "Confirme que o arquivo binário foi enviado e está disponível no deploy."
        )
    if header.lstrip().startswith((b"<!DOCTYPE html", b"<html")):
        raise ArtefactError(f"{path.name} contém uma página HTML em vez de um modelo binário.")
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner=False)
def _load_model_cached(path: Path, modified_ns: int, size: int) -> Any:
    """Carrega o binário; metadados fazem o cache expirar após substituição."""
    del modified_ns, size
    try:
        return joblib.load(path)
    except Exception as exc:
        LOGGER.exception("Falha ao desserializar o artefato %s", path)
        detail = str(exc).strip().replace("\n", " ")
        if len(detail) > 180:
            detail = detail[:177] + "..."
        suffix = f" Detalhe: {type(exc).__name__}: {detail}" if detail else f" Detalhe: {type(exc).__name__}."
        raise ArtefactError(
            f"Não foi possível desserializar {path.name}. O arquivo foi encontrado ({path.stat().st_size:,} bytes), "
            "mas pode ter sido gerado com versões diferentes de Python, Scikit-learn ou XGBoost."
            + suffix
        ) from exc


def load_model(path: Path) -> Any:
    """Carrega um modelo e renova o cache quando o arquivo é substituído."""
    modified_ns, size = _validate_model_file(path)
    return _load_model_cached(path, modified_ns, size)


@st.cache_data(show_spinner=False)
def _load_json_cached(path: Path, modified_ns: int, size: int) -> Any:
    """Lê JSON e converte erros de leitura em mensagens amigáveis."""
    del modified_ns, size
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtefactError(f"O arquivo {path.name} não contém um JSON válido.") from exc


def load_json(path: Path) -> Any:
    """Lê JSON e renova o cache quando o arquivo é substituído."""
    if not path.is_file():
        raise ArtefactError(f"Arquivo não encontrado: {path.name}")
    stat = path.stat()
    return _load_json_cached(path, stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _load_csv_cached(path: Path, modified_ns: int, size: int) -> pd.DataFrame:
    """Lê um CSV e rejeita resultados vazios."""
    del modified_ns, size
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ArtefactError(f"Não foi possível ler {path.name}.") from exc
    if data.empty:
        raise ArtefactError(f"O arquivo {path.name} está vazio.")
    return data


def load_csv(path: Path) -> pd.DataFrame:
    """Lê CSV e renova o cache quando o arquivo é substituído."""
    if not path.is_file():
        raise ArtefactError(f"Arquivo não encontrado: {path.name}")
    stat = path.stat()
    return _load_csv_cached(path, stat.st_mtime_ns, stat.st_size)


def availability(paths: dict[str, Path]) -> tuple[list[Path], list[Path]]:
    """Separa os caminhos configurados entre encontrados e ausentes."""
    found = [path for path in paths.values() if path.is_file()]
    missing = [path for path in paths.values() if not path.is_file()]
    return found, missing
