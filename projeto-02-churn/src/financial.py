"""Cálculos didáticos da simulação financeira."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd


def individual_simulation(probability: float, cost: float, preserved_value: float, success_rate: float) -> dict[str, float]:
    """Calcula benefício esperado = p(churn) × sucesso × valor − custo."""
    if not 0 <= probability <= 1 or cost < 0 or preserved_value < 0 or not 0 <= success_rate <= 1:
        raise ValueError("Os parâmetros financeiros estão fora dos intervalos permitidos.")
    gross = probability * success_rate * preserved_value
    return {"cost": cost, "gross_benefit": gross, "net_result": gross - cost}


def normalize_column(name: object) -> str:
    """Normaliza acentos, caixa e separadores para comparar cabeçalhos reais."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", ascii_name.lower())).strip("_")


def find_column(data: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Localiza uma coluna independentemente de acentos, caixa e separadores."""
    normalized = {normalize_column(column): str(column) for column in data.columns}
    return next((normalized[normalize_column(candidate)] for candidate in candidates if normalize_column(candidate) in normalized), None)


def aggregate_summary(data: pd.DataFrame) -> dict[str, object]:
    """Identifica o melhor cenário apenas quando as colunas reais existem."""
    net = find_column(data, (
        "resultado_liquido", "resultado_liquido_estimado", "beneficio_liquido",
        "beneficio_liquido_estimado", "lucro_liquido", "lucro_estimado",
        "retorno_liquido", "valor_liquido", "net_result", "net_benefit", "profit",
    ))
    threshold = find_column(data, ("threshold", "threshold_otimizado", "limiar", "ponto_de_corte"))
    if not net or not threshold:
        available = ", ".join(map(str, data.columns))
        raise ValueError(f"Não foi possível identificar threshold e resultado líquido. Colunas recebidas: {available}.")
    numeric_net = pd.to_numeric(data[net], errors="coerce")
    if numeric_net.notna().sum() == 0:
        raise ValueError("A coluna de resultado líquido não possui valores numéricos.")
    row = data.loc[numeric_net.idxmax()]
    return {"row": row, "net_column": net, "threshold_column": threshold}


def brl(value: float) -> str:
    """Formata moeda no padrão brasileiro sem depender do locale do servidor."""
    formatted = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"
