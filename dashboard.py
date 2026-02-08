import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import OSSanaliser
from StatusAnaliser import GitHubStatsAnalyzerAllTime, GitHubLanguageCommitAnalyzer
import SentimentalAnaliser
import configparser
from github import Github, GithubException

import configparser


cfg = configparser.ConfigParser()
cfg.read("config.ini")
USERNAME = cfg["github"].get("username", "").strip()
TOKEN    = cfg["github"].get("token", "").strip()

if not (USERNAME and TOKEN):
    st.error("Preencha seu username e token GitHub.")
    st.stop()

@st.cache_data(ttl=3600)
def load_participation_stats(username: str):
    """
    Chama OSSanaliser.get_repo_participation_stats(username)
    Retorna: (owned_list, non_owned_list)
    """
    return OSSanaliser.get_repo_participation_stats(username)


@st.cache_data(ttl=3600)
def load_status_per_repo(username: str, token: str):
    """
    Instancia GitHubStatsAnalyzerAllTime e chama analyze_all().
    Retorna: dict { reponame: { ..., "Total_commits": int, ... }, ... }
    """
    analyzer = GitHubStatsAnalyzerAllTime(username, token)
    return analyzer.analyze_all()

@st.cache_data(ttl=3600)
def load_status_total(username: str, token: str):
    analyzer =  GitHubStatsAnalyzerAllTime(username, token)
    return analyzer.get_aggregate_stats()

@st.cache_data(ttl=3600)
def load_sentiment_for_repos(repo_list, num_events=10):
    """
    Para cada full repo (formato 'username/reponame'), chama
    SentimentalAnaliser.get_user_activity_sentiment.
    Retorna: dict { "username/reponame": sentiment_float }
    """
    SentimentalAnaliser.setup_nltk()
    result = {}
    for full_name in repo_list:
        score, all = SentimentalAnaliser.get_user_activity_sentiment(full_name, num_events)
        result[full_name] = score
    return result

@st.cache_data(ttl=3600)
def load_sentiment_for_comments(username: str,
                                token: str,
                                num_events: int = 10):
    """
    1) Busca comentários (até ~300) via REST Events API
    2) Seleciona os primeiros `num_events`
    3) Calcula sentimento médio desses comentários
    Retorna: (quantidade_analisada, média_sentimento)
    """

    comments = SentimentalAnaliser.fetch_user_issue_comments_rest(username, token)
    selected = comments[:num_events]
    overall_avg, _scores = SentimentalAnaliser.get_user_comments_sentiment(selected)
    return len(selected), overall_avg



@st.cache_data(ttl=3600)
def fetch_all_commit_dates(username: str, token: str) -> pd.DatetimeIndex:
    """
    Coleta todas as datas de commits de um usuário em todos os repositórios.
    Agrupa falhas 409 para repositórios vazios.
    """
    gh = Github(token)
    user = gh.get_user(username)
    dates = []
    for repo in user.get_repos():
        try:
            commits = repo.get_commits(author=username)
        except GithubException as e:
            if e.status == 409:
                continue
            else:
                raise
        try:
            for commit in commits:
                try:
                    dates.append(commit.commit.author.date)
                except:
                    continue
        except GithubException as e:
            if e.status == 409:
                continue
            else:
                raise
    return pd.to_datetime(dates)


def frequency_by_quarter(dates: pd.DatetimeIndex) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype=int)
    quarters = dates.to_series().dt.to_period('Q')
    freq = quarters.value_counts().sort_index()
    freq.index = freq.index.astype(str)
    return freq


def load_issues_and_prs_stats(username, token):
    """
    Usa as funções de OSSanaliser para obter dados de Issues e PRs abertos/fechados/mesclados.
    Retorna um DataFrame com colunas:
    [Repositório, Open_Issues, Closed_Issues, Open_PRs, Merged_PRs]
    """
    import OSSanaliser
    from collections import defaultdict

    num_events = 100  # ou qualquer outro valor adequado

    open_data = OSSanaliser.get_user_opened_issues_and_prs(username, num_events)
    closed_data = OSSanaliser.get_user_resolved_issues_and_prs(username, num_events)

    # Estrutura para acumular por repositório
    repo_stats = defaultdict(lambda: {
        "Open_Issues": 0,
        "Closed_Issues": 0,
        "Open_PRs": 0,
        "Merged_PRs": 0
    })

    for issue in open_data.get("issues", []):
        repo = issue["repo"]
        repo_stats[repo]["Open_Issues"] += 1

    for pr in open_data.get("prs", []):
        repo = pr["repo"]
        repo_stats[repo]["Open_PRs"] += 1

    for issue in closed_data.get("resolved_issues", []):
        repo = issue["repo"]
        repo_stats[repo]["Closed_Issues"] += 1

    for pr in closed_data.get("closed_prs", []):
        repo = pr["repo"]
        repo_stats[repo]["Merged_PRs"] += 1

    # Caso merged_prs_count seja global e não por repo, distribuímos uniformemente (ou ignoramos)
    # Neste caso, ignoraremos pois já estamos contando merged PRs como fechados.

    # Montar DataFrame final
    data = []
    for repo, counts in repo_stats.items():
        row = {"Repositório": repo}
        row.update(counts)
        data.append(row)

    return pd.DataFrame(data)


def load_language_distribution(username, token):
    analyzer = GitHubLanguageCommitAnalyzer(username, token)
    return analyzer.analyze_language_usage()



def main():
    st.title("📊 SoftSkillMiner Dashboard")
    st.markdown(
        """
        Dashboard:
        1. Número de commits, linhas trocadas e repositórios do usuario e distribuição percentual de commits por repositório.
        2. Repositórios próprios/colaborados.
        3. Sentimento médio (Issues, Pull-Requests, Commits) em cada repositório.
        4. Distruibuição de Issues(Abertos e Fechados) e Pull-Requests (Abertos e merged).
        5. Distribuição por Linguagem de Programação.
        """
    )

    st.subheader("1. Distribuição de commits por repositório")
    with st.spinner("Buscando estatísticas de commits por repositório..."):
        status_per_repo = load_status_per_repo(USERNAME, TOKEN)

    commit_data = {"Repositório": [], "Total_Commits": []}
    for repo_name, stats in status_per_repo.items():
        commit_data["Repositório"].append(repo_name)
        commit_data["Total_Commits"].append(stats.get("Total_commits", 0))


    df_commits = pd.DataFrame(commit_data)
    total_all_commits = df_commits["Total_Commits"].sum()


    total_stats = load_status_total(USERNAME,TOKEN)

    # Calcular totais
    if 'df_commits' in locals():
        total_commits = int(df_commits["Total_Commits"].sum())
        total_lines = total_stats["Linhas_trocas"]
        total_repos = df_commits.shape[0]
    else:
        total_commits = total_lines = total_repos = 0

    # Estilo tipo "card":
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Commits", total_commits)
    col2.metric("Repositórios Analisados", total_repos)
    col3.metric("Linhas de Código Trocadas", total_lines)

    
    if total_all_commits == 0:
        st.warning("⚠️ Nenhum commit encontrado (soma total de commits = 0).")
    else:
        df_commits["Percentual"] = df_commits["Total_Commits"] / total_all_commits * 100
        df_commits = df_commits.sort_values("Total_Commits", ascending=False).reset_index(drop=True)

        st.write("Tabela: commits por repositório")
        st.dataframe(df_commits[["Repositório", "Total_Commits", "Percentual"]])

        top_n = 5
        df_top = df_commits.head(top_n).copy()
        if len(df_commits) > top_n:
            others = df_commits.iloc[top_n:]
            outros_row = pd.DataFrame([{
                "Repositório": "Outros",
                "Total_Commits": others["Total_Commits"].sum(),
                "Percentual":   others["Percentual"].sum()
            }])
            df_top = pd.concat([df_top, outros_row], ignore_index=True)


        fig2, ax2 = plt.subplots()
        ax2.pie(
            df_top["Percentual"],
            labels=df_top["Repositório"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax2.axis("equal")
        st.pyplot(fig2)

    st.write("---")


    #st.subheader("2. Seus repositórios: Próprios vs. Colaborações")
    #with st.spinner("Carregando lista de repositórios..."):
    #    owned_repos, non_owned_repos = load_participation_stats(USERNAME)

#    st.write(
 #       f"- Você possui **{len(owned_repos)}** repositórios.\n"
  ## )
#
 #   df_participation = pd.DataFrame({
  #      "Tipo": ["Próprios", "Colaborando"],
   #     "Quantidade": [len(owned_repos), len(non_owned_repos)]
    #})

    #fig1, ax1 = plt.subplots()
#    ax1.pie(
#        df_participation["Quantidade"],
#        labels=df_participation["Tipo"],
#        autopct="%1.1f%%",
#        startangle=90
#    )
#    ax1.axis("equal")
#    st.pyplot(fig1)
#
#    st.write("---")
#

    st.subheader("2. Sentimento médio geral dos repositórios")
    repo_full_names = [f"{USERNAME}/{repo_name}" for repo_name in status_per_repo.keys()]
    with st.spinner("Calculando sentimento nos comentários (pode demorar um pouco)..."):
        sentiment_dict = load_sentiment_for_repos(repo_full_names, num_events=10)
    # Filtrar apenas os sentimentos diferentes de zero
    valid_sentiments = [v for v in sentiment_dict.values() if v != 0]
    if not valid_sentiments:
        st.warning("Nenhum dado de sentimento válido para calcular a média (todos zero ou ausentes).")
    else:
        # Calcula média geral sem considerar zeros
        mean_sentiment = sum(valid_sentiments) / len(valid_sentiments)
        st.metric("Sentimento Médio Geral", f"{mean_sentiment:.3f}")

        # Plot único de barra horizontal
        fig_sent, ax_sent = plt.subplots(figsize=(6, 2))
        ax_sent.barh([0], [mean_sentiment]*2, color='steelblue')
        ax_sent.set_xlim(-1, 1)
        ax_sent.set_yticks([0])
        ax_sent.set_yticklabels(["Média Geral"])
        ax_sent.set_xlabel("Sentimento (-1 a +1)")
        ax_sent.set_title("Sentimento médio geral de todos os repositórios")
        ax_sent.axvline(x=0, color='red', linestyle='--', linewidth=1)
        plt.tight_layout()
        st.pyplot(fig_sent)

        st.write("---")
    
    
    st.subheader("3. Comentarios em issues outros issues com analise de sentimento")

    with st.spinner("Procurando todos os comentarios e calculando sentimento (pode demorar um pouco)..."):
        numero_comments, avg_score_comments = load_sentiment_for_comments(
        USERNAME, TOKEN, num_events=20
    )
    
    st.metric("Número de comentarios", f"{numero_comments}")
    st.metric("Média de sentimento dos comentários", f"{avg_score_comments:.2f}")
    # Plot único de barra horizontal
    fig_sent, ax_sent = plt.subplots(figsize=(6, 2))
    ax_sent.barh([0], [avg_score_comments]*2, color='steelblue')
    ax_sent.set_xlim(-1, 1)
    ax_sent.set_yticks([0])
    ax_sent.set_yticklabels(["Média Geral"])
    ax_sent.set_xlabel("Sentimento (-1 a +1)")
    ax_sent.set_title("Sentimento médio geral de todos os repositórios")
    ax_sent.axvline(x=0, color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    st.pyplot(fig_sent)

    st.write("---")
    
    
    st.subheader("4. Estado atual de Issues e Pull Requests")

    with st.spinner("Coletando informações de Issues e PRs..."):
        df_issues_prs = load_issues_and_prs_stats(USERNAME, TOKEN)

    if df_issues_prs.empty:
        st.warning("Nenhum dado de Issues/PRs encontrado.")
    else:
        st.dataframe(df_issues_prs)

        # Plot de barras agrupadas (open vs closed issues / open vs merged PRs)
        fig4, ax4 = plt.subplots(figsize=(10, 0.5 * len(df_issues_prs)))

        width = 0.35
        x = range(len(df_issues_prs))
        ax4.barh([i + width for i in x], df_issues_prs["Open_Issues"], height=width, label="Issues Abertas", color="tab:orange")
        ax4.barh(x, df_issues_prs["Closed_Issues"], height=width, label="Issues Fechadas", color="tab:blue")

        ax4.set(yticks=[i + width / 2 for i in x], yticklabels=df_issues_prs["Repositório"])
        ax4.invert_yaxis()
        ax4.set_xlabel("Quantidade")
        ax4.set_title("Issues abertas vs. fechadas por repositório")
        ax4.legend()
        st.pyplot(fig4)

        # Segundo gráfico: PRs
        fig5, ax5 = plt.subplots(figsize=(10, 0.5 * len(df_issues_prs)))

        ax5.barh([i + width for i in x], df_issues_prs["Open_PRs"], height=width, label="PRs Abertos", color="tab:red")
        ax5.barh(x, df_issues_prs["Merged_PRs"], height=width, label="PRs Mesclados", color="tab:green")

        ax5.set(yticks=[i + width / 2 for i in x], yticklabels=df_issues_prs["Repositório"])
        ax5.invert_yaxis()
        ax5.set_xlabel("Quantidade")
        ax5.set_title("Pull Requests abertos vs. mesclados por repositório")
        ax5.legend()
        st.pyplot(fig5)


        st.subheader("5. Distribuição por Linguagem de Programação")
        with st.spinner("Calculando distribuição por linguagem..."):
            lang_stats = load_language_distribution(USERNAME, TOKEN)

        if not lang_stats:
            st.warning("Nenhum dado encontrado de linguagens para este usuário.")
            return

        # Transformar para DataFrame
        df_lang = pd.DataFrame([
            {"Linguagem": lang, "Linhas_editadas": vals[0], "Commits": vals[1]}
            for lang, vals in lang_stats.items()
        ])

        df_lang = df_lang.sort_values("Linhas_editadas", ascending=False).reset_index(drop=True)

        st.dataframe(df_lang)
        
        fig1, ax1 = plt.subplots(figsize=(8, max(3, 0.5 * len(df_lang))))
        ax1.barh(df_lang["Linguagem"], df_lang["Linhas_editadas"], color="tab:blue")
        ax1.set_xlabel("Linhas Editadas")
        ax1.set_title("Distribuição de Linhas Editadas por Linguagem")
        plt.tight_layout()
        st.pyplot(fig1)

        
        fig2, ax2 = plt.subplots(figsize=(8, max(3, 0.5 * len(df_lang))))
        ax2.barh(df_lang["Linguagem"], df_lang["Commits"], color="tab:green")
        ax2.set_xlabel("Número de Commits")
        ax2.set_title("Distribuição de Commits por Linguagem")
        plt.tight_layout()
        st.pyplot(fig2)

    st.write("---")

    st.subheader("6. Frequência de Commits por Trimestre")
    with st.spinner("Coletando datas de commits..."):
        dates = fetch_all_commit_dates(USERNAME, TOKEN)
    if dates.empty:
        st.warning("Nenhum commit encontrado para análise trimestral.")
    else:
        freq = frequency_by_quarter(dates)
        df_freq = freq.reset_index()
        df_freq.columns = ["Trimestre", "Commits"]
        st.dataframe(df_freq)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df_freq["Trimestre"], df_freq["Commits"])
        ax.set_title("Commits por Trimestre (YYYYQX)")
        ax.set_xlabel("Trimestre")
        ax.set_ylabel("Número de Commits")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)



if __name__ == "__main__":
    main()

