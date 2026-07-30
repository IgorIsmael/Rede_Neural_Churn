"""Validação, montagem da entrada e inferência — nunca treinamento."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_COLUMNS, DEFAULT_THRESHOLD, PATHS, RISK_LIMITS
from .loaders import ArtefactError, load_json, load_model


def validate_inputs(values: dict[str, Any]) -> list[str]:
    """Valida presença, tipos e limites do formulário."""
    errors = [f"Campo ausente: {column}." for column in DEFAULT_COLUMNS if column not in values]
    try:
        tenure = int(values.get("tenure", -1))
        monthly = float(values.get("MonthlyCharges", -1))
        total = float(values.get("TotalCharges", -1))
        if not 0 <= tenure <= 100:
            errors.append("O tempo como cliente deve estar entre 0 e 100 meses.")
        if not np.isfinite(monthly) or monthly < 0:
            errors.append("O valor mensal deve ser um número maior ou igual a zero.")
        if not np.isfinite(total) or total < 0:
            errors.append("O valor total deve ser um número maior ou igual a zero.")
    except (TypeError, ValueError):
        errors.append("Revise os valores numéricos informados.")
    return errors


def input_columns() -> tuple[list[str], str | None]:
    """Obtém a ordem oficial ou usa a ordem Telco documentada."""
    try:
        raw = load_json(PATHS["input_columns"])
        columns = raw.get("colunas", raw.get("columns")) if isinstance(raw, dict) else raw
        if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
            raise ArtefactError("colunas_entrada.json deve conter uma lista de nomes.")
        forbidden = {"customerID", "Churn"} & set(columns)
        missing = set(DEFAULT_COLUMNS) - set(columns)
        extra = set(columns) - set(DEFAULT_COLUMNS)
        if forbidden or missing or extra or len(columns) != len(DEFAULT_COLUMNS):
            raise ArtefactError("A lista de colunas é incompatível com a entrada Telco esperada.")
        return columns, None
    except ArtefactError as exc:
        return DEFAULT_COLUMNS.copy(), f"{exc} Foi usada a ordem padrão documentada."


def build_dataframe(values: dict[str, Any]) -> tuple[pd.DataFrame, str | None]:
    """Monta uma única linha com nomes técnicos na ordem esperada."""
    errors = validate_inputs(values)
    if errors:
        raise ValueError(" ".join(errors))
    columns, warning = input_columns()
    frame = pd.DataFrame([{column: values[column] for column in columns}], columns=columns)
    frame["SeniorCitizen"] = frame["SeniorCitizen"].astype(int)
    frame["tenure"] = frame["tenure"].astype(int)
    frame[["MonthlyCharges", "TotalCharges"]] = frame[["MonthlyCharges", "TotalCharges"]].astype(float)
    return frame, warning


def select_model() -> tuple[Any, str]:
    """Seleciona o modelo avançado e recorre ao modelo básico."""
    errors = []
    for key, label in (("advanced_model", "XGBoost calibrado"), ("basic_model", "modelo final (fallback)")):
        try:
            return load_model(PATHS[key]), label
        except ArtefactError as exc:
            errors.append(str(exc))
    raise ArtefactError("Nenhum modelo de previsão está disponível. " + " ".join(errors))


def select_threshold() -> tuple[float, str, str | None]:
    """Seleciona threshold avançado, básico ou o fallback seguro 0,5."""
    problems = []
    for key, label in (("advanced_threshold", "threshold XGBoost"), ("basic_threshold", "threshold básico")):
        try:
            data = load_json(PATHS[key])
            value = data.get("threshold") if isinstance(data, dict) else None
            value = float(value)
            if not 0 <= value <= 1:
                raise ValueError
            return value, label, None
        except (ArtefactError, TypeError, ValueError):
            problems.append(PATHS[key].name)
    return DEFAULT_THRESHOLD, "fallback seguro", f"Threshold válido não encontrado ({', '.join(problems)}); foi usado 0,50."


def predict(frame: pd.DataFrame) -> dict[str, Any]:
    """Calcula a probabilidade positiva e aplica o threshold persistido."""
    model, model_source = select_model()
    threshold, threshold_source, threshold_warning = select_threshold()
    try:
        probabilities = np.asarray(model.predict_proba(frame))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or probabilities.shape[1] < 2:
            raise ValueError("saída inesperada")
        classes = list(getattr(model, "classes_", []))
        positive_index = classes.index(1) if 1 in classes else (classes.index("Yes") if "Yes" in classes else 1)
        probability = float(probabilities[0, positive_index])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probabilidade fora do intervalo")
    except Exception as exc:
        raise ArtefactError("O modelo é incompatível com a entrada ou falhou durante predict_proba.") from exc
    return {
        "probability": probability, "threshold": threshold,
        "classification": "Risco de churn" if probability >= threshold else "Tendência de permanência",
        "risk": risk_band(probability), "model_source": model_source,
        "threshold_source": threshold_source, "warning": threshold_warning,
    }


def risk_band(probability: float) -> str:
    """Classifica a probabilidade nas faixas de risco da interface."""
    if probability < RISK_LIMITS[0]:
        return "Baixo risco"
    if probability < RISK_LIMITS[1]:
        return "Risco moderado"
    return "Alto risco"
