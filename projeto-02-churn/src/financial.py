"""Cálculos didáticos da simulação financeira."""
from __future__ import annotations

import pandas as pd


def individual_simulation(probability: float, cost: float, preserved_value: float, success_rate: float) -> dict[str, float]:
    """Calcula benefício esperado = p(churn) × sucesso × valor − custo."""
    if not 0 <= probability <= 1 or cost < 0 or preserved_value < 0 or not 0 <= success_rate <= 1:
        raise ValueError("Os parâmetros financeiros estão fora dos intervalos permitidos.")
    gross = probability * success_rate * preserved_value
    return {"cost": cost, "gross_benefit": gross, "net_result": gross - cost}


def find_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Localiza uma coluna independentemente de maiúsculas e separadores comuns."""
    normalized = {str(column).lower().replace(" ", "_").replace("-", "_"): str(column) for column in data.columns}
    return next((normalized[candidate] for candidate in candidates if candidate in normalized), None)


def aggregate_summary(data: pd.DataFrame) -> dict[str, object]:
    """Identifica o melhor cenário apenas quando as colunas reais existem."""
    net = find_column(data, ("resultado_liquido", "lucro_liquido", "net_result"))
    threshold = find_column(data, ("threshold", "limiar"))
    if not net or not threshold:
        raise ValueError("O CSV precisa conter colunas de threshold e resultado líquido.")
    numeric_net = pd.to_numeric(data[net], errors="coerce")
    if numeric_net.notna().sum() == 0:
        raise ValueError("A coluna de resultado líquido não possui valores numéricos.")
    row = data.loc[numeric_net.idxmax()]
    return {"row": row, "net_column": net, "threshold_column": threshold}


def brl(value: float) -> str:
    """Formata moeda no padrão brasileiro sem depender do locale do servidor."""
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"
