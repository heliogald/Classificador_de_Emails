# Classificador Inteligente de Emails (Streamlit + OpenRouter + Heurística + opcional local ML)

Uma aplicação para classificar emails em **Produtivo** / **Improdutivo** e gerar respostas automáticas.
Combina LLM (OpenRouter / OpenAI client) + heurísticas + (opcional) classificador local TF-IDF + LogisticRegression.

---

## Conteúdo do repositório

- `app.py` - Aplicação Streamlit completa (upload de .txt/.pdf, spaCy preprocessing, ensemble LLM+heurística, opção FastAPI).
- `train_local_classifier.py` - Script para treinar e salvar um classificador local (opcional).
- `models/` - (gerado) onde o classificador local e TF-IDF serão salvos (`local_clf.joblib`, `local_tfidf.joblib`).
- `requirements.txt` - Dependências.
- `.github/workflows/ci.yml` - GitHub Actions para CI (instala deps e roda pytest).
- `examples/` - (recomendado) coloque exemplos .txt rotulados para demonstração.

---

## Requisitos

- Python 3.10+ recomendado.
- Variáveis de ambiente:
  - `OPENROUTER_API_KEY` — chave do OpenRouter / OpenAI (se quiser usar LLM).
  - `MODEL_NAME` (opcional) — ex.: `openai/gpt-4o-mini`.

---

## Primeiros passos (local)

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo

Crie e ative ambiente virtual:

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows


Instale dependências:

pip install -r requirements.txt
# se falhar com o wheel de pt_core_news_sm, instale manualmente:
# python -m pip install pt_core_news_sm
# ou rode: python -m spacy download pt_core_news_sm


Configure a chave da API:

export OPENROUTER_API_KEY="sua_chave_aqui"    # macOS/Linux
setx OPENROUTER_API_KEY "sua_chave_aqui"     # Windows (em CMD)
# Ou coloque em um .env (usando python-dotenv)


(Opcional) Treine um classificador local:

python train_local_classifier.py --data data/dataset.csv --out-dir models
# dataset.csv deve conter colunas: "text","label" (label: Produtivo/Improdutivo)


Rode a aplicação Streamlit:

streamlit run app.py
```
