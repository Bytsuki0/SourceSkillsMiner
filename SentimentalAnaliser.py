import os
import time
import requests
import nltk
from langdetect import detect, LangDetectException
from translate import Translator
from nltk.sentiment import SentimentIntensityAnalyzer
import configparser as cfgparser


import nltk

# Garante que o vader_lexicon está disponível
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

def setup_nltk():
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        print("Baixando recursos do NLTK necessários...")
        nltk.download('vader_lexicon')
        print("Download concluído.")

# Inicialização única de tradutor e analisador de sentimento
translator = Translator(from_lang="pt", to_lang="en")
sia        = SentimentIntensityAnalyzer()

# Sessão HTTP com possível autenticação GitHub

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
session = requests.Session()
config = cfgparser.ConfigParser()
config.read('config.ini')
username = config['github']['username']
token = config['github']['token']
if token:
    session.headers.update({"Authorization": f"token {token}"})
else:
    print("⚠️  Atenção: variável GITHUB_TOKEN não definida — você ficará limitado a 60 requisições/hora.")


def safe_get(url, max_retries=3, backoff_factor=2):
    """
    Tenta obter a URL até max_retries vezes; em caso de 429, faz backoff exponencial.
    Retorna o objeto Response ou None se falhar.
    """
    for attempt in range(max_retries):
        resp = session.get(url)
        if resp.status_code == 429:
            wait = backoff_factor ** attempt
            print(f"429 recebido, aguardando {wait}s antes da próxima tentativa...")
            time.sleep(wait)
            continue
        return resp
    print(f"❌ Falha ao obter {url} após {max_retries} tentativas")
    return None


def is_english(text: str) -> bool:
    """
    Returns True if langdetect thinks the text is English.
    Falls back to False on very short or undetectable texts.
    """
    try:
        # langdetect can choke on empty or super‑short strings:
        if len(text.strip()) < 5:
            return False
        return detect(text) == "en"
    except LangDetectException:
        return False

def get_user_activity_sentiment(repo_full_name, num_events=10):
    """
    Coleta comentários de issues, PRs, commits e discussões de um repositório GitHub
    e retorna a média geral de sentimento (em escala de -1 a 1).

    Parâmetros:
      - repo_full_name: str no formato "usuario/repositorio"
      - num_events: número máximo de comentários por categoria

    Retorna:
      - float: média composta dos escores de sentimento.
      - dict: escores por repo chave = repos, valor = lista de escores.
    """
    # Extract username and repo
    username, repo = repo_full_name.split("/")

    # Endpoints públicos para cada tipo de comentário
    endpoints = {
        "issues_comments":      f"https://api.github.com/repos/{username}/{repo}/issues/comments",
        "pr_comments":          f"https://api.github.com/repos/{username}/{repo}/pulls/comments",
        "commit_comments":      f"https://api.github.com/repos/{username}/{repo}/comments",
        "comments_url": f"https://api.github.com/repos/{username}/{repo}/comments"
    }

    sentiment_scores = {key: [] for key in endpoints}

    # Para cada categoria, faz a requisição e processa os comentários
    for key, url in endpoints.items():
        resp = safe_get(url)
        if resp and resp.status_code == 200:
            comments = resp.json()[:num_events]
            for c in comments:
                body = c.get('body', '') or ''
                if is_english(body):
                    text_en = body
                else:
                        text_en = translator.translate(body)
                score = sia.polarity_scores(text_en)['compound']
                sentiment_scores[key].append(score)
        else:
            status = resp.status_code if resp else 'erro'
            print(f"Aviso: não foi possível obter {key} (status {status})")

    # Cálculo das médias por categoria e geral
    all_scores = []
    averages = {}
    for cat, scores in sentiment_scores.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        averages[cat] = avg
        all_scores.extend(scores)

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    averages['geral'] = overall

    # Exibir resumo
    for cat, avg in averages.items():
        label = "Positivo" if avg > 0.05 else "Negativo" if avg < -0.05 else "Neutro"

    return overall, sentiment_scores

def fetch_user_issue_comments_rest(username: str,
                                   token: str = None,
                                   max_pages: int = 20):
    """
    Busca comentários de issue feitos por `username` via GitHub REST Events API.

    Params:
      - username:   GitHub login
      - token:      Personal Access Token (ou setar GITHUB_TOKEN no ambiente)
      - max_pages:  Quantas páginas de eventos buscar (100 eventos/página).
                    Até ~300 eventos públicos serão retornados.

    Returns:
      List[dict] onde cada dict contém:
        - repo_full_name (e.g. "owner/repo")
        - issue_number   (int)
        - comment_url    (str)
        - created_at     (ISO timestamp)
        - body           (str)
    """
    token = token or os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("Defina GITHUB_TOKEN ou passe `token` como argumento")

    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json"
    }

    comments = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/users/{username}/events/public"
        params = {"per_page": 100, "page": page}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            break  # sem mais eventos

        for ev in events:
            if ev.get("type") != "IssueCommentEvent":
                continue

            c = ev["payload"]["comment"]
            issue = ev["payload"]["issue"]
            repo = ev["repo"]["name"]

            comments.append({
                "repo_full_name": repo,
                "issue_number":   issue["number"],
                "comment_url":    c["html_url"],
                "created_at":     c["created_at"],
                "body":           c.get("body", "").strip()
            })

        # Se você quiser parar após um certo # comentários:
        # if len(comments) >= desired_count: break

        # Se vier menos de 100 eventos, não há mais páginas
        if len(events) < 100:
            break

    return comments

def get_user_comments_sentiment(comments):
    """
    Recebe uma lista de dicts de comentários (cada um com chave 'body'),
    traduz e calcula o sentimento de cada um, e retorna:
      - overall_avg: float, média de todos os compound scores
      - scores:      List[float], score de cada comentário na ordem original
    """
    scores = []

    for c in comments:
        text = c.get("body") or ""
        if is_english(text):
            text_en = text
        else:
            text_en = translator.translate(text)
        compound = sia.polarity_scores(text_en)["compound"]
        scores.append(compound)

    # d) calcula média final
    overall = sum(scores) / len(scores) if scores else 0.0

    return overall, scores


if __name__ == '__main__':
    setup_nltk()
    rep = "pystardust/sbar"
    f2 = get_user_activity_sentiment(rep, num_events=10) 
    print(f2)
   

    
