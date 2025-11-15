# 📧 Classificador Inteligente de Emails  
### *Streamlit + OpenRouter + Heurísticas + Classificador ML Opcional*

Uma aplicação que classifica emails em **Produtivo** ou **Improdutivo** e gera **respostas automáticas** usando Inteligência Artificial.  
A solução combina:

- Modelos LLM via **OpenRouter**
- Regras heurísticas para maior precisão
- (Opcional) Classificador local usando **TF-IDF + LogisticRegression**
- Interface intuitiva via **Streamlit**

---

## 📁 Conteúdo do Repositório

- `app.py` — Aplicação Streamlit (upload de .txt e .pdf, heurísticas, LLM e resposta automática)
- `train_local_classifier.py` — Treinamento do classificador local
- `models/` — Armazena `local_clf.joblib` e `local_tfidf.joblib`
- `requirements.txt` — Dependências
- `examples/` — Arquivos de email para teste
- `.github/workflows/ci.yml` — Pipeline CI

---

## 🧰 Requisitos

- Python **3.10+**
- Variáveis de ambiente:
  - `OPENROUTER_API_KEY`
  - `MODEL_NAME` (opcional)

---

## 🚀 Como Executar o Projeto Localmente

### 1. Clone o repositório
```bash
git clone https://github.com/heliogald/Classificador_de_Emails.git
cd seu-repo
```

### 2. Crie e ative ambiente virtual

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instale dependências
```bash
pip install -r requirements.txt
```

Se necessário:
```bash
python -m spacy download pt_core_news_sm
```

### 4. Configure variáveis
**Linux/Mac**
```bash
export OPENROUTER_API_KEY="sua_chave_aqui"
```

**Windows**
```cmd
setx OPENROUTER_API_KEY "sua_chave_aqui"
```

Arquivo `.env` (opcional):
```
OPENROUTER_API_KEY=sua_chave
MODEL_NAME=openai/gpt-4o-mini
```

### 5. (Opcional) Treine o classificador local
```bash
python train_local_classifier.py --data data/dataset.csv --out-dir models
```

### 6. Execute o Streamlit
```bash
streamlit run app.py
```

---

## 🌐 Deploy (Hugging Face Spaces)

1. Crie um Space (tipo **Streamlit**)
2. Faça upload dos arquivos
3. Adicione o secret:
   - `OPENROUTER_API_KEY`
4. Deploy automático

---

## 🤖 Funcionalidades

- Upload de `.txt` e `.pdf`
- Classificação híbrida
- Geração de resposta automática
- Download da resposta
- Heurísticas inteligentes + LLM
- CI com GitHub Actions

---

## 📄 Licença

MIT License
