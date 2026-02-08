# commit_frequency.py
# ----------------
# Analisa a frequência de commits de um usuário do GitHub por trimestre anual

import configparser
from github import Github, GithubException
import pandas as pd
import matplotlib.pyplot as plt


def load_github_client(config_path: str = "config.ini") -> (Github, str):
    """
    Carrega username e token de config.ini e retorna um cliente PyGithub e o username.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    username = cfg["github"].get("username", "").strip()
    token = cfg["github"].get("token", "").strip()
    if not (username and token):
        raise ValueError("Preencha username e token em config.ini.")
    return Github(token), username


def fetch_commit_dates(gh: Github, username: str) -> pd.DatetimeIndex:
    """
    Para cada repositório do usuário, coleta a lista de datas de commits feitos por ele.
    Retorna um DatetimeIndex com todas as datas.
    """
    user = gh.get_user(username)
    all_dates = []
    for repo in user.get_repos():
        # Tentar obter commits; repositório vazio lança GithubException 409
        try:
            commits = repo.get_commits(author=username)
        except GithubException as e:
            if e.status == 409:
                continue
            else:
                raise
        # Iterar commits, capturando possíveis 409 durante paginação
        try:
            for commit in commits:
                try:
                    date = commit.commit.author.date
                    all_dates.append(date)
                except Exception:
                    continue
        except GithubException as e:
            if e.status == 409:
                # pular este repositório
                continue
            else:
                raise
    return pd.to_datetime(all_dates)


def frequency_by_quarter(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Agrupa commits em trimestres (Q1, Q2, Q3, Q4) por ano.
    Retorna uma Series indexada por período 'YYYYQX'.
    """
    if dates.empty:
        return pd.Series(dtype=int)
    quarters = dates.to_series().dt.to_period('Q')
    freq = quarters.value_counts().sort_index()
    freq.index = freq.index.astype(str)
    return freq


def plot_quarterly_commits(freq: pd.Series):
    """
    Plota commits totais por trimestre.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(freq.index, freq.values)
    ax.set_title('Commits por Trimestre (ano-trimestre)')
    ax.set_xlabel('Trimestre')
    ax.set_ylabel('Número de commits')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    gh, username = load_github_client()
    print(f"Coletando commits para {username}...")
    dates = fetch_commit_dates(gh, username)
    if dates.empty:
        print("Nenhum commit encontrado.")
        return

    freq = frequency_by_quarter(dates)
    print("\nCommits por trimestre (YYYYQX):")
    print(freq)
    plot_quarterly_commits(freq)

if __name__ == "__main__":
    main()
