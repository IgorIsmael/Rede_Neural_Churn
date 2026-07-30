"""Rótulos exclusivamente visuais; valores técnicos permanecem inalterados."""
from __future__ import annotations

FIELD_LABELS = {
    "gender": "Gênero", "SeniorCitizen": "Cliente idoso", "Partner": "Possui parceiro(a)",
    "Dependents": "Possui dependentes", "tenure": "Tempo como cliente (meses)",
    "PhoneService": "Serviço telefônico", "MultipleLines": "Múltiplas linhas",
    "InternetService": "Tipo de internet", "OnlineSecurity": "Segurança online",
    "OnlineBackup": "Backup online", "DeviceProtection": "Proteção do dispositivo",
    "TechSupport": "Suporte técnico", "StreamingTV": "Streaming de TV",
    "StreamingMovies": "Streaming de filmes", "Contract": "Tipo de contrato",
    "PaperlessBilling": "Cobrança sem papel", "PaymentMethod": "Forma de pagamento",
    "MonthlyCharges": "Valor mensal", "TotalCharges": "Valor total",
}
HELP = {
    "tenure": "Quantidade de meses desde o início do relacionamento.",
    "SeniorCitizen": "Indicador original do dataset: 1 para cliente idoso e 0 caso contrário.",
    "MonthlyCharges": "Cobrança mensal atual, em unidades monetárias.",
    "TotalCharges": "Total acumulado; pode ser ajustado manualmente.",
    "Contract": "Prazo contratual vigente.",
    "PaymentMethod": "Método usado para pagamento da fatura.",
}
CATEGORY_LABELS = {
    "Female": "Feminino", "Male": "Masculino", "Yes": "Sim", "No": "Não",
    "No phone service": "Sem serviço telefônico", "Fiber optic": "Fibra óptica",
    "No internet service": "Sem serviço de internet", "Month-to-month": "Mensal",
    "One year": "Um ano", "Two year": "Dois anos", "Electronic check": "Cheque eletrônico",
    "Mailed check": "Cheque enviado", "Bank transfer (automatic)": "Transferência bancária (automática)",
    "Credit card (automatic)": "Cartão de crédito (automático)", "DSL": "DSL",
}
FEATURE_TRANSLATIONS = {
    "tenure": "Tempo como cliente", "MonthlyCharges": "Valor mensal", "TotalCharges": "Valor total",
    "Contract_Month-to-month": "Contrato mensal", "TechSupport_No": "Sem suporte técnico",
    "OnlineSecurity_No": "Sem segurança online", "PaymentMethod_Electronic check": "Pagamento por cheque eletrônico",
    "InternetService_Fiber optic": "Internet por fibra óptica",
}


def category_label(value: object) -> str:
    """Traduz uma categoria somente para exibição."""
    if value in (0, 1):
        return "Sim" if value == 1 else "Não"
    return CATEGORY_LABELS.get(str(value), str(value))


def translate_feature(name: str) -> str:
    """Remove prefixos de pipelines e traduz uma feature transformada."""
    clean = name
    for prefix in ("numerico__", "categorico__", "onehot__", "pipeline__"):
        clean = clean.replace(prefix, "")
    if clean in FEATURE_TRANSLATIONS:
        return FEATURE_TRANSLATIONS[clean]
    for technical, label in FIELD_LABELS.items():
        if clean == technical:
            return label
        if clean.startswith(f"{technical}_"):
            category = clean[len(technical) + 1:]
            return f"{label}: {category_label(category)}"
    return clean.replace("_", " ")
