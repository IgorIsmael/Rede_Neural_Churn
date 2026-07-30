"""Carregamento seguro e cacheado dos artefatos externos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


class ArtefactError(RuntimeError):
    """Erro legível relacionado a um artefato da aplicação."""


@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any:
    """Carrega um modelo pickle/joblib sem modificar o arquivo."""
    if not path.is_file():
        raise ArtefactError(f"Modelo não encontrado: {path.name}")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ArtefactError(f"Não foi possível carregar {path.name}; verifique as versões das bibliotecas.") from exc


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> Any:
    """Lê JSON e converte erros de leitura em mensagens amigáveis."""
    if not path.is_file():
        raise ArtefactError(f"Arquivo não encontrado: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtefactError(f"O arquivo {path.name} não contém um JSON válido.") from exc


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV e rejeita resultados vazios."""
    if not path.is_file():
        raise ArtefactError(f"Arquivo não encontrado: {path.name}")
    try:
        data = pd.read_csv(path)
    except Exception as exc:
        raise ArtefactError(f"Não foi possível ler {path.name}.") from exc
    if data.empty:
        raise ArtefactError(f"O arquivo {path.name} está vazio.")
    return data


def availability(paths: dict[str, Path]) -> tuple[list[Path], list[Path]]:
    """Separa os caminhos configurados entre encontrados e ausentes."""
    found = [path for path in paths.values() if path.is_file()]
    missing = [path for path in paths.values() if not path.is_file()]
    return found, missing
