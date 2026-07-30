"""Configurações e caminhos portáveis da aplicação."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefatos"
ADVANCED_ARTEFACTS_DIR = BASE_DIR / "artefatos_avancados"
ASSETS_DIR = BASE_DIR / "assets"

PATHS = {
    "advanced_model": ADVANCED_ARTEFACTS_DIR / "xgboost_calibrado.pkl",
    "shap_model": ADVANCED_ARTEFACTS_DIR / "xgboost_original.pkl",
    "advanced_threshold": ADVANCED_ARTEFACTS_DIR / "threshold_xgboost.json",
    "basic_model": ARTEFACTS_DIR / "modelo_final_churn.pkl",
    "basic_threshold": ARTEFACTS_DIR / "threshold.json",
    "input_columns": ARTEFACTS_DIR / "colunas_entrada.json",
    "metadata": ARTEFACTS_DIR / "metadados_modelo.json",
    "comparison": ARTEFACTS_DIR / "comparacao_modelos.csv",
    "advanced_comparison": ADVANCED_ARTEFACTS_DIR / "comparacao_modelos_avancada.csv",
    "cross_validation": ADVANCED_ARTEFACTS_DIR / "validacao_cruzada.csv",
    "calibration": ADVANCED_ARTEFACTS_DIR / "calibracao_modelos.csv",
    "financial": ADVANCED_ARTEFACTS_DIR / "simulacao_financeira.csv",
}

DEFAULT_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
GITHUB_URL = "https://github.com/IgorIsmael/deep-learning-lab"
DEFAULT_THRESHOLD = 0.5
RISK_LIMITS = (0.40, 0.70)
