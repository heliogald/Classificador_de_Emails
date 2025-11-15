import os
import re
import json
import math
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# LLM client (OpenRouter wrapper via openai.OpenAI)
try:
    from openai import OpenAI
except Exception:
    # if openai package not available, will error later on LLM calls
    OpenAI = None

# PDF extraction
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# spaCy
try:
    import spacy
except Exception:
    spacy = None

# Optional local classifier
try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    joblib = None

# FastAPI (optional)
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
except Exception:
    FastAPI = None

# ---------------------------
# Config / Load env
# ---------------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

# Init LLM client if possible
client = None
if OPENROUTER_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    except Exception as e:
        client = None

# ---------------------------
# Constants / Keywords
# ---------------------------
PRODUTIVO_KEYWORDS = [
    "protocolo", "status", "erro", "falha", "incidente", "recuperar",
    "reclamação", "problema", "reembolso", "fatura", "cobrança",
    "documento", "anexo", "contrato", "solicitação", "suporte",
    "ajuda", "urgente", "prazo", "vencimento", "cancelar", "ativar",
    "acesso", "senha", "restaurar", "processamento", "autenticação",
    "número", "numero", "id", "identificador"
]

IMPRODUTIVO_KEYWORDS = [
    "parabéns", "feliz", "boas festas", "obrigado", "agradeço",
    "saudações", "ótimo trabalho", "cumprimentos", "newsletter",
    "informativo", "spam", "divulgação", "promoção", "felicitações"
]

# ---------------------------
# NLP Setup (spaCy) - attempt to load Portuguese model
# ---------------------------
nlp = None
SPACY_MODEL_NAME = "pt_core_news_sm"
if spacy is not None:
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
    except Exception:
        # if model not installed, try to fallback to blank Portuguese pipeline
        try:
            nlp = spacy.blank("pt")
        except Exception:
            nlp = None

# ---------------------------
# Optional local classifier load
# If you have models/local_clf.joblib and models/tfidf.joblib, they will be used.
# ---------------------------
local_clf = None
local_tfidf = None
MODEL_DIR = "models"
LOCAL_CLF_PATH = os.path.join(MODEL_DIR, "local_clf.joblib")
LOCAL_TFIDF_PATH = os.path.join(MODEL_DIR, "local_tfidf.joblib")
if joblib is not None:
    try:
        if os.path.exists(LOCAL_CLF_PATH) and os.path.exists(LOCAL_TFIDF_PATH):
            local_clf = joblib.load(LOCAL_CLF_PATH)
            local_tfidf = joblib.load(LOCAL_TFIDF_PATH)
    except Exception:
        local_clf = None
        local_tfidf = None

# ---------------------------
# Helpers: text extraction
# ---------------------------
def extract_text_from_uploaded(uploaded) -> str:
    """Recebe um Streamlit UploadedFile e retorna string com o texto extraído."""
    filename = uploaded.name.lower()
    try:
        content = uploaded.read()
    except Exception:
        content = None

    # TXT
    if filename.endswith(".txt") or (uploaded.type == "text/plain"):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            try:
                return content.decode("latin-1", errors="ignore")
            except Exception:
                return str(content)

    # PDF
    if filename.endswith(".pdf"):
        if pdfplumber is None:
            return "Erro: pdfplumber não instalado no ambiente."
        try:
            text_pages = []
            # pdfplumber accepts file-like objects
            uploaded.seek(0)
            with pdfplumber.open(uploaded) as pdf:
                for p in pdf.pages:
                    pg_text = p.extract_text()
                    if pg_text:
                        text_pages.append(pg_text)
            return "\n".join(text_pages)
        except Exception as e:
            return f"Erro ao extrair PDF: {e}"

    # fallback: try decode
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return str(content)

# ---------------------------
# Helpers: preprocessing
# ---------------------------
def sanitize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def spacy_preprocess(text: str) -> str:
    """
    Lematiza e remove stopwords/punctuacao usando spaCy quando disponível.
    Fallback: lower + remove múltiplos espaços.
    """
    if not text:
        return ""
    if nlp is None:
        return sanitize_text(text).lower()
    try:
        doc = nlp(text)
        lemmas = []
        for t in doc:
            if getattr(t, "is_stop", False) or t.is_punct or t.is_space:
                continue
            lemma = t.lemma_.lower() if t.lemma_ else t.text.lower()
            # remove tokens with only punctuation or digits
            if re.fullmatch(r"[\W_]+", lemma) or re.fullmatch(r"\d+", lemma):
                continue
            lemmas.append(lemma)
        return " ".join(lemmas)
    except Exception:
        return sanitize_text(text).lower()

# ---------------------------
# Helpers: heuristic scoring
# ---------------------------
def heuristic_score(text: str) -> dict:
    txt = text.lower()
    prod_count = sum(txt.count(k) for k in PRODUTIVO_KEYWORDS)
    impr_count = sum(txt.count(k) for k in IMPRODUTIVO_KEYWORDS)
    question_marks = txt.count("?")
    numbers = len(re.findall(r"\d{2,}", txt))
    has_attachment = bool(re.search(r"\banexo\b|\banexado\b|\battachment\b", txt))
    has_deadline = bool(re.search(r"\b(?:dia|até|amanhã|hoje|prazo|útil|úteis|horas)\b", txt))
    word_count = len(txt.split())
    score = (prod_count * 1.3) - (impr_count * 1.0) + (0.9 * question_marks) + (0.8 * has_attachment) + (0.6 * has_deadline) + (0.35 * numbers)
    if word_count > 0:
        score = score / math.log(word_count + 2)
    scaled = max(min(score / 3.0, 1.0), -1.0)
    features = {
        "prod_count": prod_count,
        "impr_count": impr_count,
        "question_marks": question_marks,
        "numbers": numbers,
        "has_attachment": bool(has_attachment),
        "has_deadline": bool(has_deadline),
        "word_count": word_count,
        "raw_score": score,
        "scaled_score": scaled
    }
    return {"score": scaled, "features": features}

# ---------------------------
# Helpers: LLM classification (JSON output)
# ---------------------------
def parse_model_json_output(text: str) -> dict:
    txt = text.strip()
    try:
        json_match = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group(0))
            return {"ok": True, "data": obj}
        else:
            if re.search(r"\bProdutivo\b", txt, flags=re.IGNORECASE):
                return {"ok": True, "data": {"category": "Produtivo", "explanation": txt, "confidence": 0.6}}
            if re.search(r"\bImprodutivo\b", txt, flags=re.IGNORECASE):
                return {"ok": True, "data": {"category": "Improdutivo", "explanation": txt, "confidence": 0.6}}
    except Exception:
        pass
    return {"ok": False, "raw_text": txt}

def llm_classify(email_text: str) -> dict:
    """
    Faz a requisição ao LLM pedindo JSON estrito:
    { "category": "Produtivo"|"Improdutivo", "confidence": 0.0..1.0, "explanation": "..." }
    """
    if client is None:
        return {"ok": False, "error": "LLM client não inicializado (OPENROUTER_API_KEY ausente ou client libs faltando)."}
    prompt = f"""
Você é um classificador binário de emails para uma empresa financeira.
Retorne apenas um objeto JSON válido com as chaves:
- category: "Produtivo" ou "Improdutivo"
- confidence: número entre 0.0 e 1.0
- explanation: textual curto (<=25 palavras)

Regras:
Considere 'Produtivo' quando o email requer ação específica (status, erro, documento, suporte).
Considere 'Improdutivo' para felicitações, agradecimentos, newsletters ou sem necessidade de ação.
Se estiver incerto, defina confidence baixa (ex.: 0.3-0.6).

Email:
---BEGIN EMAIL---
{email_text}
---END EMAIL---
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Você é um classificador de textos. Forneça apenas um JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=220
        )
        model_text = response.choices[0].message.content
        parsed = parse_model_json_output(model_text)
        if parsed.get("ok"):
            data = parsed["data"]
            cat_raw = data.get("category", "")
            if isinstance(cat_raw, str) and cat_raw.lower().startswith("prod"):
                category = "Produtivo"
            elif isinstance(cat_raw, str) and cat_raw.lower().startswith("improd"):
                category = "Improdutivo"
            else:
                category = None
            confidence = float(data.get("confidence", 0.0) if data.get("confidence") is not None else 0.0)
            explanation = data.get("explanation", "").strip()
            return {"ok": True, "category": category, "confidence": confidence, "explanation": explanation, "raw": model_text}
        else:
            return {"ok": False, "raw": parsed.get("raw_text", "")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---------------------------
# Combine decision (LLM + heur + local model)
# ---------------------------
def combine_decision(llm_result: dict, heur: dict, local_pred: Optional[str] = None) -> dict:
    heur_score = heur["score"]
    heur_features = heur["features"]
    heur_label = "Produtivo" if heur_score > 0.15 else ("Improdutivo" if heur_score < -0.05 else "Indeterminado")
    # no LLM
    if not llm_result.get("ok"):
        final_label = heur_label if heur_label != "Indeterminado" else ("Improdutivo")
        confidence = 0.5 + (abs(heur_score) * 0.4)
        explanation = f"Decisão por heurística (features: {heur_features})"
        source = "heuristic"
        # if local model exists, use it to boost confidence
        if local_pred:
            if local_pred == final_label:
                confidence = min(0.95, confidence + 0.2)
                explanation += f" + local_model({local_pred})"
            else:
                # conflict: mark indeterminate lower confidence
                confidence = max(0.45, confidence - 0.15)
                explanation += f" (conflito local_model={local_pred})"
        return {"category": final_label, "confidence": round(min(confidence, 0.99), 2), "explanation": explanation, "source": source}

    # LLM ok
    llm_cat = llm_result.get("category")
    llm_conf = llm_result.get("confidence", 0.0)
    llm_expl = llm_result.get("explanation", "")
    source = "llm"
    # If very confident LLM, use it but apply strong heuristic override
    if llm_conf >= 0.75:
        if llm_cat == "Produtivo" and heur_score < -0.4:
            return {"category": "Improdutivo", "confidence": 0.7, "explanation": "Override heurístico forte (conteúdo claramente improdutivo)", "source": "override"}
        # local model can confirm
        if local_pred and local_pred != llm_cat and local_pred is not None:
            # slight demotion if conflict
            llm_conf = max(0.55, llm_conf - 0.15)
            source = "llm+local_conflict"
            return {"category": llm_cat, "confidence": round(llm_conf, 2), "explanation": f"{llm_expl} (conflito local_model={local_pred})", "source": source}
        return {"category": llm_cat, "confidence": round(llm_conf, 2), "explanation": llm_expl, "source": source}

    # medium/low LLM confidence: combine
    heur_norm = (heur_score + 1) / 2
    combined_conf = round(min(0.95, 0.6 * llm_conf + 0.4 * heur_norm), 2)

    # local model acts as tie-breaker
    if local_pred:
        if local_pred != llm_cat:
            # if heur favours local_pred, choose it
            if (heur_score > 0.1 and local_pred == "Produtivo") or (heur_score < -0.1 and local_pred == "Improdutivo"):
                return {"category": local_pred, "confidence": max(0.55, combined_conf), "explanation": f"Tie-breaker: local model votou por {local_pred}", "source": "local_tiebreak"}
            else:
                # follow combined vote
                vote = (llm_conf * (1 if llm_cat == "Produtivo" else -1)) + (heur_score * 0.5)
                final_label = "Produtivo" if vote > 0 else "Improdutivo"
                return {"category": final_label, "confidence": combined_conf, "explanation": f"Combinação LLM+heur (local_model={local_pred})", "source": "combined"}
        else:
            # agreement: boost confidence
            return {"category": llm_cat, "confidence": min(0.99, combined_conf + 0.15), "explanation": f"LLM + local model concordam ({llm_expl})", "source": "combined_agree"}

    # no local model
    vote = (llm_conf * (1 if llm_cat == "Produtivo" else -1)) + (heur_score * 0.5)
    final_label = "Produtivo" if vote > 0 else "Improdutivo"
    return {"category": final_label, "confidence": combined_conf, "explanation": f"Combinação LLM+heur ({llm_expl})", "source": "combined"}

# ---------------------------
# Response templates + polish
# ---------------------------
def generate_response_template(email_text: str, category: str, confidence: float) -> str:
    if category == "Produtivo":
        resp = (
            "Recebemos sua solicitação e iniciamos a análise. "
            "Nossa equipe retornará com um posicionamento em até 24 horas úteis. "
            "Se possuir documentos adicionais, responda a este e-mail."
        )
        if confidence < 0.6:
            resp = (
                "Recebemos sua mensagem e vamos analisá-la. "
                "A equipe poderá solicitar mais informações caso necessário. "
                "Retornaremos em até 24 horas úteis."
            )
        return resp
    else:
        resp = "Agradecemos o contato. Sua mensagem foi recebida. Caso precise de suporte técnico, por favor utilize o canal de atendimento."
        if confidence < 0.5:
            resp = "Obrigado pelo contato. Se precisar de suporte, estamos à disposição."
        return resp

def polish_text_with_llm(text: str) -> str:
    if client is None:
        return text
    prompt = f"Reescreva a mensagem abaixo em no máximo 3 linhas, tom profissional e conciso. Não mencione datas comemorativas.\n\n{text}"
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Você reescreve mensagens de forma concisa e profissional."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=180
        )
        polished = resp.choices[0].message.content.strip()
        if len(polished) < 5:
            return text
        return polished
    except Exception:
        return text

# ---------------------------
# Main wrapper: process email
# ---------------------------
def processar_email_com_ia(texto_email: str, do_polish: bool = True) -> Dict[str, Any]:
    texto_email = sanitize_text(texto_email)
    if not texto_email:
        return {"status": "error", "message": "Texto vazio."}

    # 1) Preprocess
    preprocessed = spacy_preprocess(texto_email)

    # 2) Heuristic
    heur = heuristic_score(texto_email)

    # 3) Local model prediction if available
    local_pred = None
    if local_clf is not None and local_tfidf is not None:
        try:
            X = local_tfidf.transform([preprocessed])
            pred = local_clf.predict(X)[0]
            # expect model labels to be "Produtivo" / "Improdutivo"
            local_pred = pred
        except Exception:
            local_pred = None

    # 4) LLM classification
    llm_res = llm_classify(texto_email)

    # 5) Combine
    decision = combine_decision(llm_res, heur, local_pred)

    # 6) Generate response
    suggested = generate_response_template(texto_email, decision["category"], decision["confidence"])
    polished_text = suggested
    if do_polish:
        polished_text = polish_text_with_llm(suggested)

    result = {
        "category": decision["category"],
        "confidence": decision["confidence"],
        "explanation": decision["explanation"],
        "source": decision.get("source", ""),
        "suggested_response": polished_text,
        "debug": {
            "preprocessed": preprocessed,
            "heuristic": heur,
            "llm": llm_res,
            "local_model": {"pred": local_pred} if local_pred else None
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    return {"status": "ok", "result": result}

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🤖 Classificador Inteligente de Emails", layout="wide")
st.title("📧 Classificador e Sugestão de Resposta de Emails (AI)")
st.markdown("Disponibiliza upload de `.txt` / `.pdf`, pré-processamento com spaCy, ensemble LLM+heurística e resposta automática.")

st.sidebar.markdown("### Opções")
show_debug = st.sidebar.checkbox("Mostrar debug", value=False)
do_polish = st.sidebar.checkbox("Usar reescrita (polish) via LLM", value=True)
use_local_model = st.sidebar.checkbox("Considerar classificador local (se disponível)", value=True)

# upload area
st.markdown("#### 1. Carregue um email (.txt ou .pdf) ou cole o texto abaixo")
uploaded = st.file_uploader("Upload (.txt, .pdf) ou escolha exemplo", type=["txt", "pdf"])
col_text, col_actions = st.columns([3,1])

if uploaded:
    texto_in = extract_text_from_uploaded(uploaded)
else:
    texto_in = col_text.text_area("Ou cole o e-mail aqui", height=260, placeholder="Ex.: Olá, qual o status do protocolo 12345? Obrigado.")

# examples download / quick fill
with col_actions:
    st.write("Exemplos")
    if st.button("Carregar exemplo Produtivo"):
        texto_in = "Olá, solicito atualização do protocolo 12345. O prazo expirou e precisamos de posição. Favor responder com o status."
    if st.button("Carregar exemplo Improdutivo"):
        texto_in = "Quero desejar feliz natal para toda a equipe! Obrigado pelo suporte."
    if st.button("Download exemplos"):
        ex_zip = False
        # generate a small zip or text bundle for download
        exemplos = {
            "prod_1.txt": "Solicito atualização do protocolo 98765. Houve erro no processamento e anexo comprovante.",
            "prod_2.txt": "Boa tarde, o boleto com ID 44321 não foi registrado. Preciso de reemissão.",
            "improd_1.txt": "Parabéns a toda a equipe pelo ótimo trabalho neste ano! Boas festas!",
            "improd_2.txt": "Recebi o informativo mensal. Obrigado pelas informações."
        }
        # prepare plain text output
        multi = "\n\n---\n\n".join([f"{k}\n\n{v}" for k,v in exemplos.items()])
        st.download_button("Baixar exemplos (.txt)", data=multi, file_name="exemplos_emails.txt", mime="text/plain")

# main action
if st.button("Processar Email", type="primary"):
    if not texto_in or not texto_in.strip():
        st.warning("Por favor, insira ou carregue o texto do email antes de processar.")
    else:
        with st.spinner("Analisando o email..."):
            outcome = processar_email_com_ia(texto_in, do_polish=do_polish if client else False)
            st.markdown("---")
            if outcome.get("status") != "ok":
                st.error(outcome.get("message", "Erro desconhecido"))
            else:
                res = outcome["result"]
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Categoria de Classificação:**")
                    cat = res["category"]
                    conf = res["confidence"]
                    if cat == "Produtivo":
                        st.success(f"## 🛠️ {cat}  (confiança: {conf})")
                    elif cat == "Improdutivo":
                        st.info(f"## 🥳 {cat}  (confiança: {conf})")
                    else:
                        st.warning(f"## ⚠️ {cat}  (confiança: {conf})")
                    st.markdown(f"**Fonte da decisão:** {res.get('source')}")
                    st.markdown("**Explicação curta:**")
                    st.write(res.get("explanation"))

                with col2:
                    st.markdown("**Resposta Automática Sugerida:**")
                    st.code(res.get("suggested_response", ""), language='text')
                    st.download_button(
                        label="Baixar / Copiar Resposta Sugerida",
                        data=res.get("suggested_response", ""),
                        file_name="resposta_automatica.txt",
                        mime="text/plain"
                    )

                if show_debug:
                    st.markdown("---")
                    st.subheader("🔍 Debug")
                    st.json(res["debug"])
                    # show raw LLM output if present
                    if res["debug"].get("llm"):
                        st.markdown("**LLM raw (se disponível):**")
                        st.text(res["debug"]["llm"].get("raw", ""))

st.markdown("---")
st.markdown("Aplicação usando **OpenRouter** (se a chave estiver configurada) e processamento local (spaCy, heurística).")

# ---------------------------
# Optional FastAPI server for batch processing (only if explicitly requested via CLI)
# ---------------------------
def run_fastapi():
    if FastAPI is None:
        print("FastAPI/uvicorn não disponível. Instale 'fastapi' e 'uvicorn' para habilitar endpoint.")
        return

    api = FastAPI(title="Email Classifier API")

    class EmailIn(BaseModel):
        text: str
        polish: Optional[bool] = True

    @api.post("/process")
    def process_email(payload: EmailIn):
        return processar_email_com_ia(payload.text, do_polish=payload.polish)

    @api.get("/health")
    def health():
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}

    # run uvicorn programmatically - won't run if this script executed by Streamlit normally
    uvicorn.run(api, host="0.0.0.0", port=8000)

# ---------------------------
# CLI: start FastAPI if asked (only when running python app.py --with-api)
# ---------------------------
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-api", action="store_true", help="Start FastAPI endpoint on 0.0.0.0:8000 (useful for integration).")
    args = parser.parse_args()
    if args.with_api:
        run_fastapi()

if __name__ == "__main__":
    # Only run CLI server when executed directly from command line (not when imported by Streamlit).
    # For streamlit run app.py, __name__ == "__main__" but Streamlit loads file differently. To avoid starting API unintentionally,
    # require explicit flag --with-api.
    main_cli()
