"""Elementos visuais reutilizáveis da aplicação."""
from __future__ import annotations

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
    @media(max-width:640px){.block-container{padding:1rem}.stTabs [data-baseweb="tab"]{font-size:.8rem}}
    </style>""", unsafe_allow_html=True)


def show_prediction(result: dict[str, object]) -> None:
    """Apresenta o resultado probabilístico com linguagem não determinística."""
    probability = float(result["probability"])
    columns = st.columns(4)
    columns[0].metric("Probabilidade estimada", f"{probability:.1%}")
    columns[1].metric("Classificação", str(result["classification"]))
    columns[2].metric("Faixa", str(result["risk"]))
    columns[3].metric("Threshold do modelo", f"{float(result['threshold']):.1%}")
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
