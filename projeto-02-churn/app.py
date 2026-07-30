"""Interface Streamlit do Projeto 02 — somente inferência e análise."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import ADVANCED_ARTEFACTS_DIR, ARTEFACTS_DIR, GITHUB_URL, PATHS
from src.explainability import ExplainabilityError, explain_client
from src.financial import aggregate_summary, brl, find_column, individual_simulation
from src.loaders import ArtefactError, availability, load_csv, load_model
from src.prediction import build_dataframe, predict
from src.translations import FIELD_LABELS, HELP, category_label
from src.ui import inject_css, metric_chart, show_prediction

st.set_page_config(page_title="Previsão de Churn de Clientes", page_icon="📊", layout="wide")
inject_css()

OPTIONS = {
    "gender": ["Female", "Male"], "SeniorCitizen": [0, 1], "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"], "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def select(name: str) -> object:
    """Cria um seletor que traduz apenas a apresentação."""
    return st.selectbox(FIELD_LABELS[name], OPTIONS[name], format_func=category_label, help=HELP.get(name))


def configuration_status() -> None:
    """Mostra uma página de configuração quando nenhum modelo existe."""
    found, missing = availability(PATHS)
    with st.expander("Status dos artefatos", expanded=not bool(found)):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Arquivos encontrados")
            if found:
                for path in found: st.success(f"✓ {path.parent.name}/{path.name}")
            else: st.info("Nenhum artefato foi encontrado ainda.")
        with right:
            st.markdown("#### Arquivos ausentes")
            for path in missing: st.warning(f"• {path.parent.name}/{path.name}")
        st.markdown(
            f"Adicione os arquivos em `{ARTEFACTS_DIR.name}/` e `{ADVANCED_ARTEFACTS_DIR.name}/`, "
            "sem alterar seus nomes, e recarregue a página. A aplicação não treina nem reconstrói modelos."
        )


def prediction_tab() -> None:
    """Renderiza formulário, inferência e último resultado."""
    st.header("Previsão individual")
    st.caption("Informe os dados originais do cliente Telco. Nenhum dado é usado para treinamento.")
    if not PATHS["advanced_model"].is_file() and not PATHS["basic_model"].is_file():
        st.info("A interface está pronta, mas adicione ao menos um modelo para calcular uma previsão.")
        configuration_status()
    with st.form("churn_form"):
        st.subheader("1 — Perfil do cliente")
        cols = st.columns(5)
        with cols[0]: gender = select("gender")
        with cols[1]: senior = select("SeniorCitizen")
        with cols[2]: partner = select("Partner")
        with cols[3]: dependents = select("Dependents")
        with cols[4]: tenure = st.number_input(FIELD_LABELS["tenure"], 0, 100, 12, 1, help=HELP["tenure"])

        st.subheader("2 — Telefonia e internet")
        telecom = {}
        names = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for index, name in enumerate(names):
            if index % 3 == 0: columns = st.columns(3)
            with columns[index % 3]: telecom[name] = select(name)

        st.subheader("3 — Contrato e cobrança")
        cols = st.columns(3)
        with cols[0]: contract = select("Contract")
        with cols[1]: paperless = select("PaperlessBilling")
        with cols[2]: payment = select("PaymentMethod")
        cols = st.columns(2)
        with cols[0]: monthly = st.number_input(FIELD_LABELS["MonthlyCharges"], min_value=0.0, value=70.0, step=0.01, help=HELP["MonthlyCharges"])
        suggested_total = float(tenure) * float(monthly)
        with cols[1]: total = st.number_input(FIELD_LABELS["TotalCharges"], min_value=0.0, value=suggested_total, step=0.01, help=f"{HELP['TotalCharges']} Sugestão atual: {suggested_total:.2f}.")
        submitted = st.form_submit_button("Calcular risco de churn", type="primary", use_container_width=True)

    if submitted:
        values = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                  "tenure": tenure, **telecom, "Contract": contract, "PaperlessBilling": paperless,
                  "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total}
        expected = suggested_total
        if abs(total - expected) > max(100.0, expected * 0.5):
            st.warning("O valor total difere bastante de tempo como cliente × valor mensal. A previsão não foi bloqueada; confirme o dado.")
        try:
            frame, column_warning = build_dataframe(values)
            result = predict(frame)
            st.session_state["last_prediction"] = {**result, "input": frame}
            if column_warning: st.warning(column_warning)
            if result["warning"]: st.warning(result["warning"])
        except (ValueError, ArtefactError) as exc:
            st.error(str(exc))
    if "last_prediction" in st.session_state:
        st.divider()
        show_prediction(st.session_state["last_prediction"])


def explanation_tab() -> None:
    """Renderiza fatores SHAP da última previsão quando possível."""
    st.header("Explicação da última previsão")
    if "last_prediction" not in st.session_state:
        st.info("Faça uma previsão na primeira aba para visualizar a explicação.")
        return
    if not PATHS["shap_model"].is_file():
        st.warning("O arquivo xgboost_original.pkl ainda não está disponível. A previsão continua funcionando, mas a explicação SHAP não pode ser exibida.")
        return
    try:
        model = load_model(PATHS["shap_model"])
        factors = explain_client(model, st.session_state["last_prediction"]["input"])
        positive = factors[factors["Contribuição SHAP"] >= 0]
        negative = factors[factors["Contribuição SHAP"] < 0]
        left, right = st.columns(2)
        with left:
            st.subheader("Fatores associados ao aumento do risco")
            st.dataframe(positive, hide_index=True, use_container_width=True)
        with right:
            st.subheader("Fatores associados à redução do risco")
            st.dataframe(negative, hide_index=True, use_container_width=True)
        st.bar_chart(factors.set_index("Fator")["Contribuição SHAP"])
        st.info("SHAP explica o comportamento do modelo, mas não comprova relação de causa e efeito.")
    except (ArtefactError, ExplainabilityError) as exc:
        st.warning(f"Não foi possível gerar a explicação: {exc} A previsão continua disponível normalmente.")


def financial_tab() -> None:
    """Renderiza simulação individual e cenários agregados opcionais."""
    st.header("Simulação financeira")
    st.info("Simulação didática baseada em premissas informadas; não representa garantia financeira.")
    probability = float(st.session_state.get("last_prediction", {}).get("probability", 0.0))
    if "last_prediction" not in st.session_state:
        st.warning("Faça uma previsão para usar a probabilidade individual. Enquanto isso, o cálculo considera 0%.")
    columns = st.columns(3)
    cost = columns[0].number_input("Custo da abordagem (R$)", min_value=0.0, value=25.0, step=1.0)
    value = columns[1].number_input("Valor estimado preservado (R$)", min_value=0.0, value=600.0, step=10.0)
    success = columns[2].slider("Taxa estimada de sucesso", 0, 100, 30) / 100
    result = individual_simulation(probability, cost, value, success)
    st.caption("Fórmula: probabilidade de churn × taxa de sucesso × valor preservado − custo da abordagem.")
    columns = st.columns(3)
    columns[0].metric("Custo da abordagem", brl(result["cost"]))
    columns[1].metric("Benefício bruto esperado", brl(result["gross_benefit"]))
    columns[2].metric("Resultado líquido esperado", brl(result["net_result"]))
    (st.success if result["net_result"] >= 0 else st.warning)("O resultado estimado é positivo." if result["net_result"] >= 0 else "O resultado estimado é negativo.")

    st.subheader("Cenários agregados")
    try:
        data = load_csv(PATHS["financial"])
        st.dataframe(data, hide_index=True, use_container_width=True)
        summary = aggregate_summary(data)
        threshold_col, net_col = summary["threshold_column"], summary["net_column"]
        chart = data[[threshold_col, net_col]].copy().set_index(threshold_col)
        chart[net_col] = pd.to_numeric(chart[net_col], errors="coerce")
        st.line_chart(chart)
        row = summary["row"]
        available_metrics = [("Melhor threshold", threshold_col)]
        optional_metrics = [
            ("Clientes abordados", ("clientes_abordados", "quantidade_clientes_abordados", "qtd_abordados", "abordados")),
            ("Churns reais alcançados", ("churns_reais_alcancados", "churns_alcancados", "churns_capturados", "churns_retidos")),
            ("ROI estimado", ("roi", "roi_estimado", "retorno_sobre_investimento")),
        ]
        available_metrics.extend((title, column) for title, candidates in optional_metrics if (column := find_column(data, candidates)))
        metric_columns = st.columns(len(available_metrics))
        for container, (title, column) in zip(metric_columns, available_metrics):
            container.metric(title, str(row[column]))
    except ArtefactError as exc:
        st.info(f"A análise agregada ainda não foi adicionada: {exc}")
    except ValueError as exc:
        st.info(f"A tabela foi carregada, mas o resumo automático não pôde ser montado. {exc}")


def _first_available_csv(keys: tuple[str, ...]) -> tuple[pd.DataFrame | None, str | None]:
    for key in keys:
        try: return load_csv(PATHS[key]), PATHS[key].name
        except ArtefactError: continue
    return None, None


def performance_tab() -> None:
    """Renderiza métricas existentes sem fabricar valores ausentes."""
    st.header("Desempenho dos modelos")
    st.markdown("### 1. Comparação geral")
    comparison, source = _first_available_csv(("advanced_comparison", "comparison"))
    if comparison is None: st.info("Adicione o CSV de comparação para visualizar esta seção.")
    else:
        st.caption(f"Fonte: {source}")
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        normalized = {str(c).lower().replace("-", "_").replace(" ", "_"): c for c in comparison.columns}
        metrics = [normalized[key] for key in ("f1_score", "f1", "recall", "roc_auc") if key in normalized]
        metric_chart(comparison, list(dict.fromkeys(metrics)))
        for metric in dict.fromkeys(metrics):
            values = pd.to_numeric(comparison[metric], errors="coerce")
            if values.notna().any(): st.success(f"Melhor {metric}: {values.max():.4f}")
    st.markdown("### 2. Validação cruzada")
    try:
        cross = load_csv(PATHS["cross_validation"])
        st.dataframe(cross, hide_index=True, use_container_width=True)
        normalized = {str(c).lower().replace(" ", "_"): c for c in cross.columns}
        f1_cols = [c for key, c in normalized.items() if "f1" in key and ("media" in key or "mean" in key or "desvio" in key or "std" in key)]
        metric_chart(cross, f1_cols)
        st.caption("Menor variação entre as partições indica maior estabilidade do desempenho estimado.")
    except ArtefactError as exc: st.info(str(exc))
    st.markdown("### 3. Calibração das probabilidades")
    try:
        calibration = load_csv(PATHS["calibration"])
        st.dataframe(calibration, hide_index=True, use_container_width=True)
        normalized = {str(c).lower().replace(" ", "_"): c for c in calibration.columns}
        calibration_metrics = [c for key, c in normalized.items() if "brier" in key or "log_loss" in key or "logloss" in key]
        metric_chart(calibration, calibration_metrics)
        st.caption("Para Brier Score e Log Loss, valores menores são melhores. ROC AUC não mede diretamente a qualidade das probabilidades.")
    except ArtefactError as exc: st.info(str(exc))
    st.info("A accuracy não é suficiente para avaliar churn porque um modelo pode acertar muitos clientes que permanecem e, ainda assim, deixar de identificar uma grande parte dos clientes que cancelam.")


def about_tab() -> None:
    """Documenta escopo, método e limitações do projeto."""
    st.header("Sobre o projeto")
    st.markdown("""
**Churn** é o encerramento do relacionamento de um cliente com a empresa. Este projeto educacional estima o risco de churn no dataset **Telco Customer Churn**, apoiando a priorização de análises de retenção.

Os dados originais passam, no pipeline treinado externamente, por preparação de variáveis numéricas e categóricas (incluindo One-Hot Encoding). Foram considerados **Dummy Classifier, Regressão Logística, Random Forest, Random Forest regularizado, Rede Neural, Gradient Boosting e XGBoost**.

As métricas incluem accuracy, precision, recall, F1-score e ROC AUC. O threshold converte a probabilidade em uma tendência e pode ser ajustado ao objetivo de retenção. A validação cruzada estima estabilidade entre partições; a calibração avalia a qualidade das probabilidades. SHAP descreve associações locais com a saída do modelo, sem estabelecer causalidade. A simulação financeira combina probabilidade e premissas hipotéticas de custo, sucesso e valor preservado.

### Limitações
- O dataset é educacional e pode não representar uma operação real.
- As previsões são probabilísticas, não certezas.
- Associações do modelo não demonstram causalidade.
- Desempenho e calibração precisam de monitoramento contínuo.
- É necessária validação independente em dados reais e recentes.
- As premissas financeiras são hipotéticas.

> **Esta aplicação possui finalidade educacional e não deve ser utilizada isoladamente para decisões comerciais reais.**
""")
    st.link_button("Ver repositório no GitHub", GITHUB_URL)


st.title("📊 Previsão de Churn de Clientes")
st.markdown("Aplicação educacional para estimar risco, interpretar fatores associados e explorar cenários de retenção — sem treinamento em produção.")

tabs = st.tabs(["Previsão", "Explicação", "Simulação financeira", "Desempenho dos modelos", "Sobre o projeto"])
with tabs[0]: prediction_tab()
with tabs[1]: explanation_tab()
with tabs[2]: financial_tab()
with tabs[3]: performance_tab()
with tabs[4]: about_tab()
