import pandas as pd
import numpy as np
import joblib
import argparse
import os
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Funções de Pré-processamento (do seu app.py) ---

try:
    nlp = spacy.load("pt_core_news_sm")
except:
    print("Aviso: Modelo spaCy 'pt_core_news_sm' não encontrado. Usando pipeline em branco.")
    try:
        nlp = spacy.blank("pt")
    except:
        nlp = None

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def spacy_preprocess(text: str) -> str:
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
            if re.fullmatch(r"[\W_]+", lemma) or re.fullmatch(r"\d+", lemma):
                continue
            lemmas.append(lemma)
        return " ".join(lemmas)
    except Exception:
        return sanitize_text(text).lower()

# --- Simulação de Dados de Treinamento ---

# Crie esta pasta e inclua este arquivo.
# Exemplo de dataset simples para fins de demonstração (data/dataset.csv)
def generate_sample_data(filename="data/dataset.csv"):
    if os.path.exists(filename):
        return
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    data = [
        # Produtivo (1)
        ("Qual o status do meu protocolo 45678?", "Produtivo"),
        ("Houve um erro na fatura. Preciso de suporte urgente.", "Produtivo"),
        ("O anexo contém os documentos solicitados. Por favor, confirme.", "Produtivo"),
        ("Minha senha expirou, como posso restaurar o acesso?", "Produtivo"),
        ("Solicito o cancelamento da conta com ID 999.", "Produtivo"),
        # Improdutivo (0)
        ("Feliz Natal a toda a equipe! Obrigado.", "Improdutivo"),
        ("Recebi a newsletter e achei o conteúdo ótimo, parabéns.", "Improdutivo"),
        ("Agradeço a atenção, voltamos a falar em breve.", "Improdutivo"),
        ("Cumprimentos pelo sucesso do último trimestre.", "Improdutivo"),
        ("Esta é apenas uma mensagem de teste. Ignore.", "Improdutivo"),
    ]
    
    df = pd.DataFrame(data, columns=['email_text', 'category'])
    df.to_csv(filename, index=False)
    print(f"Arquivo de dados de exemplo criado em: {filename}")


# --- Script Principal de Treinamento ---
def main():
    parser = argparse.ArgumentParser(description="Treina e salva classificador local de emails.")
    parser.add_argument("--data", default="data/dataset.csv", help="Caminho para o arquivo CSV de dados de treinamento.")
    parser.add_argument("--out-dir", default="models", help="Diretório para salvar os modelos treinados.")
    args = parser.parse_args()

    # Gera dados de exemplo se o arquivo não existir
    generate_sample_data(args.data)
    
    try:
        df = pd.read_csv(args.data)
        if df.empty or 'email_text' not in df.columns or 'category' not in df.columns:
            print(f"Erro: O arquivo de dados {args.data} está vazio ou faltando colunas 'email_text'/'category'.")
            return
    except Exception as e:
        print(f"Erro ao ler o arquivo de dados: {e}")
        return

    # Pré-processamento e tokenização
    print("Iniciando pré-processamento dos dados...")
    df['processed_text'] = df['email_text'].apply(spacy_preprocess)

    # Codificação dos rótulos
    df['label'] = df['category'].apply(lambda x: 1 if x == 'Produtivo' else 0)

    X = df['processed_text']
    y = df['label']

    # Divisão para teste (opcional, mas bom para validação)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Treinamento do TF-IDF Vectorizer
    print("Treinando TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    # 2. Treinamento do Classificador (Regressão Logística)
    print("Treinando Classificador de Regressão Logística...")
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Avaliação (para debug)
    X_test_tfidf = tfidf.transform(X_test)
    y_pred = clf.predict(X_test_tfidf)
    print("\nRelatório de Classificação (Exemplo):")
    print(classification_report(y_test, y_pred, target_names=['Improdutivo', 'Produtivo'], zero_division=0))

    # 3. Salvando os modelos
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.out_dir, "local_clf.joblib"))
    joblib.dump(tfidf, os.path.join(args.out_dir, "local_tfidf.joblib"))
    print(f"\nModelos salvos com sucesso na pasta: {args.out_dir}")

if __name__ == "__main__":
    main()