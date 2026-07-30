"""Explicações SHAP do pipeline XGBoost original."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from .translations import translate_feature


class ExplainabilityError(RuntimeError):
    """Indica que a explicação não pôde ser produzida sem afetar a previsão."""


def _pipeline_parts(pipeline: Any) -> tuple[Any, Any]:
    steps = getattr(pipeline, "named_steps", None)
    if not steps:
        raise ExplainabilityError("O artefato não possui um pipeline reconhecível.")
    preprocessor = next((steps[key] for key in ("preprocessor", "preprocessador", "preprocessing") if key in steps), None)
    estimator = next((steps[key] for key in ("classifier", "classificador", "model", "modelo", "xgboost") if key in steps), None)
    if preprocessor is None and len(steps) >= 2:
        preprocessor = list(steps.values())[-2]
    if estimator is None:
        estimator = list(steps.values())[-1]
    if preprocessor is None or estimator is None:
        raise ExplainabilityError("Pré-processador ou estimador não localizado.")
    return preprocessor, estimator


def explain_client(pipeline: Any, client: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    """Calcula e traduz as maiores contribuições SHAP absolutas."""
    try:
        preprocessor, estimator = _pipeline_parts(pipeline)
        transformed = preprocessor.transform(client)
        transformed_array = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            names = [f"Feature {index + 1}" for index in range(transformed_array.shape[1])]
        values = shap.TreeExplainer(estimator).shap_values(transformed_array)
        if isinstance(values, list):
            values = values[-1]
        array = np.asarray(values)
        if array.ndim == 3:
            array = array[:, :, -1]
        contributions = array[0]
        if len(contributions) != len(names):
            raise ValueError("dimensões incompatíveis")
        result = pd.DataFrame({"Fator": [translate_feature(name) for name in names], "Contribuição SHAP": contributions})
        result["Impacto absoluto"] = result["Contribuição SHAP"].abs()
        result = result.nlargest(limit, "Impacto absoluto").drop(columns="Impacto absoluto")
        result["Associação"] = np.where(result["Contribuição SHAP"] >= 0, "Aumentou o risco previsto", "Reduziu o risco previsto")
        return result.reset_index(drop=True)
    except ExplainabilityError:
        raise
    except Exception as exc:
        raise ExplainabilityError("O SHAP não é compatível com o artefato disponível.") from exc
