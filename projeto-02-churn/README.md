# Projeto 02 — Previsão de Churn

Aplicação Streamlit profissional e educacional para inferência de churn no dataset **Telco Customer Churn**. A aplicação somente carrega artefatos, recebe uma entrada, calcula previsões e apresenta explicações e análises; ela **não treina ou reconstrói modelos**.

## Funcionalidades

- formulário em português com os 19 campos técnicos originais;
- prioridade para XGBoost calibrado e fallback para o modelo final básico;
- threshold persistido com fallback explícito e seguro de 0,5;
- explicação local SHAP opcional com o XGBoost original;
- simulação financeira individual e análise agregada opcional;
- comparação, validação cruzada e calibração a partir dos CSVs reais;
- diagnóstico amigável quando artefatos ainda não existem.

## Estrutura

```text
projeto-02-churn/
├── app.py
├── requirements.txt
├── src/                    # configuração, loaders e regras da aplicação
├── assets/
├── artefatos/              # fallback básico (adicionar manualmente)
└── artefatos_avancados/    # XGBoost e resultados (adicionar manualmente)
```

## Artefatos necessários

Adicione manualmente em `artefatos/`:

- `modelo_final_churn.pkl`
- `threshold.json`
- `colunas_entrada.json`
- `comparacao_modelos.csv`
- `metadados_modelo.json`

Adicione manualmente em `artefatos_avancados/`:

- `xgboost_calibrado.pkl`
- `xgboost_original.pkl`
- `threshold_xgboost.json`
- `comparacao_modelos_avancada.csv`
- `validacao_cruzada.csv`
- `calibracao_modelos.csv`
- `simulacao_financeira.csv`

Não é necessário ter todos os arquivos para a página abrir. Para prever, basta o modelo calibrado ou o modelo básico; sem `xgboost_original.pkl`, apenas a explicação SHAP fica indisponível. Não renomeie os arquivos.

> **Compatibilidade de pickle:** modelos serializados podem depender da mesma versão de Python, Scikit-learn, XGBoost e demais bibliotecas usada no treinamento. O `requirements.txt` usa uma combinação conservadora baseada em Python 3.11, NumPy 1.26 e Scikit-learn 1.5. Se o carregamento falhar, use o detalhe exibido pela aplicação e alinhe as versões às registradas no ambiente que gerou o artefato; não tente editar o `.pkl`.

Se o modelo foi versionado com **Git LFS**, confirme que o deploy recebeu o conteúdo binário, e não somente o pequeno arquivo de ponteiro. A aplicação identifica explicitamente arquivos vazios, páginas HTML e ponteiros LFS. Depois de substituir um artefato, basta recarregar a aplicação: o cache é invalidado pelo tamanho e pela data de modificação do arquivo.

## Instalação local

Na raiz desta pasta:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instale e execute:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Todos os caminhos partem da localização dos próprios módulos via `pathlib.Path`, portanto a aplicação não depende do diretório corrente nem de caminhos específicos de Windows ou Linux.

## Deploy no Streamlit Community Cloud

1. Publique o repositório que contém seus artefatos autorizados.
2. Crie um app no Streamlit Community Cloud.
3. Selecione `projeto-02-churn/app.py` como arquivo principal.
4. Escolha **Python 3.11**, compatível com as versões fixadas, e faça o deploy.

## Abas

1. **Previsão:** entrada e probabilidade estimada, classificação, faixa e recomendação geral.
2. **Explicação:** até oito associações SHAP da última previsão.
3. **Simulação financeira:** cenário individual e cenários agregados, quando fornecidos.
4. **Desempenho dos modelos:** métricas reais dos CSVs adicionados.
5. **Sobre o projeto:** método, conceitos e limitações.

## Limitações e uso responsável

O dataset é educacional; as previsões são probabilísticas; SHAP não demonstra causalidade; modelos exigem monitoramento e validação em dados reais; e premissas financeiras são hipotéticas.

**Esta aplicação possui finalidade educacional e não deve ser utilizada isoladamente para decisões comerciais reais.**
