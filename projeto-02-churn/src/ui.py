"""Elementos visuais reutilizáveis da aplicação."""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st


def inject_css() -> None:
    """Aplica pequenos ajustes responsivos sem substituir componentes nativos."""
    st.markdown("""<style>
    .block-container {padding-top: 2rem; max-width: 1280px}
    [data-testid="stMetric"] {
        background:var(--secondary-background-color);
        border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
        color:var(--text-color);
        padding:1rem;
        border-radius:.75rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {color:var(--text-color) !important}
    .prediction-summary {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:1rem;
        margin:.5rem 0 1.25rem;
    }
    .prediction-card {
        min-width:0;
        min-height:8rem;
        background:var(--secondary-background-color);
        border:1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
        border-radius:.75rem;
        padding:1rem 1.15rem;
    }
    .prediction-card-label {font-size:.9rem; font-weight:600; margin-bottom:.65rem; color:var(--text-color)}
    .prediction-card-value {
        color:var(--text-color);
        font-size:clamp(1.55rem, 2.4vw, 2.35rem);
        line-height:1.18;
        overflow-wrap:anywhere;
        white-space:normal;
    }
    @media(max-width:900px){.prediction-summary{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:640px){
        .block-container{padding:1rem}
        .stTabs [data-baseweb="tab"]{font-size:.8rem}
        .prediction-summary{grid-template-columns:1fr}
        .prediction-card{min-height:auto}
    }
    </style>""", unsafe_allow_html=True)


def show_prediction(result: dict[str, object]) -> None:
    """Apresenta o resultado probabilístico com linguagem não determinística."""
    probability = float(result["probability"])
    summary = (
        ("Probabilidade estimada", f"{probability:.1%}"),
        ("Classificação", str(result["classification"])),
        ("Faixa", str(result["risk"])),
        ("Threshold do modelo", f"{float(result['threshold']):.1%}"),
    )
    cards = "".join(
        f'<div class="prediction-card"><div class="prediction-card-label">{escape(label)}</div>'
        f'<div class="prediction-card-value">{escape(value)}</div></div>'
        for label, value in summary
    )
    st.markdown(f'<div class="prediction-summary">{cards}</div>', unsafe_allow_html=True)
    st.progress(probability, text="Risco previsto de churn")
    recommendations = {
        "Baixo risco": "Manter o acompanhamento regular do relacionamento.",
        "Risco moderado": "Considerar uma análise preventiva de retenção.",
        "Alto risco": "Priorizar este cliente para uma análise de retenção.",
    }
    message = f"**Recomendação geral:** {recommendations[str(result['risk'])]} Esta é uma tendência indicada pelo modelo."
    if result["risk"] == "Alto risco": st.error(message)
    elif result["risk"] == "Risco moderado": st.warning(message)
    else: st.success(message)
    st.caption(f"Modelo: {result['model_source']} · Threshold: {result['threshold_source']}")


def metric_chart(data: pd.DataFrame, columns: list[str]) -> None:
    """Exibe gráfico somente para métricas realmente presentes no arquivo."""
    usable = [column for column in columns if column in data.columns]
    if not usable:
        st.info("As colunas de métricas esperadas não foram encontradas neste arquivo.")
        return
    label = next((column for column in data.columns if column.lower() in ("modelo", "model")), data.columns[0])
    chart = data.set_index(label)[usable].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if chart.empty: st.info("Não há valores numéricos para o gráfico.")
    else: st.bar_chart(chart)
