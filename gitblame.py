import os
import csv
import re
import time
import stat
import requests
import platform 
import subprocess
import shutil
import json
from pathlib import Path
import configparser as cfgparser
import OSSanaliser as f1
import SentimentalAnaliser as f2
import StatusAnaliser as f3


config = cfgparser.ConfigParser()
config.read('config.ini')
username = config['github']['username']
token = config['github']['token']

data_rows = []


#função de calculo preliminar caso passe o usuario é apto para o projeto e ira pra segunda parte do projeto
def preliminary(commits_non_owned,lines_of_code, pull_issues, merge_solved):
    i = 0
    print("commits_non_owned:", commits_non_owned)
    print("lines_of_code:", lines_of_code)
    print("pull_issues:", pull_issues)
    print("merge_solved:", merge_solved)
   
    return True
    
    

def get_git_commits_info(num_commits=20, repoPath = ""):
    commits_info = ""
    if f1.is_git_repo(repoPath):
        try:
            log_output = subprocess.check_output(
                ["git", "log", "--pretty=format:%H | %an | %ae | %s", "-n", str(num_commits)],
                text=True
            )
            header = f"\nÚltimos {num_commits} commits:\nCommit Hash | Autor | E-mail | Mensagem\n"
            print(header)
            print(log_output)
            commits_info += header + log_output
        except subprocess.CalledProcessError as e:
            err = f"Erro ao executar o comando Git: {e}"
            print(err)
            commits_info += err
    else:
        err = "Erro: O diretório não é um repositório Git válido."
        print(err)
        commits_info += err
    data_rows.append(("Git Commits", commits_info))

def softskillpoints(repoPath,repoName, n):
    
    repos = f1.get_repo_participation_stats(username)
    resolved_issues= f1.get_user_resolved_issues_and_prs(username, n)
    
    get_git_commits_info(n, repoPath)
    f1.get_user_activity(username, token, repoName, n)

#linha de codigo para que o programa funcione tanto no meu linux e no meu windows
def os_user(username):
    if platform.system() == "Windows":
        path = Path(__file__).resolve().parent
        return path / username
    
    if platform.system() == "Linux":
        path = Path(__file__).resolve().parent
        return path / username

def force_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Modificar a função main para usar a nova função de análise de sentimentos
def main(): 
    n = 20

    gitUserName = username
    pathProjects = os_user(gitUserName)

    if not os.path.exists(pathProjects):
        os.makedirs(pathProjects)

    #Retorna uma lista: [lista_de_proprios, lista_de_colaborando]
    repoName = f1.get_repo_participation_stats(username)
    repo1 = repoName[0]
    repo2 = repoName[1]
    repos = repo1 + repo2

    print("Digite 1 se quer baixar todos os repositórios localmente")
    if int(input()) == 1:
        for r in repos:
            repo_folder_name = r.split('/')[-1]
            repoPath = os.path.join(pathProjects, repo_folder_name)
            # Sair da pasta para evitar erro ao deletar
            os.chdir(os.path.expanduser("~"))

            if os.path.exists(repoPath):
                shutil.rmtree(repoPath, onerror=force_remove_readonly)

            subprocess.run(["git", "clone", f"https://github.com/{r}.git", repoPath])
    
    #retorna {"issues": [], "prs": []}
    pulls_issues = f1.get_user_opened_issues_and_prs(username, n)

    #retorna {"resolved_issues": [], "closed_prs": [], "merged_prs_count": 0}
    resolsed_pull_issues = f1.get_user_resolved_issues_and_prs(username, n)

    #retorna o numero de commits do usuario
    info = f1.get_commit_stats_total(username)

    #retorna os commits não proprios 
    info_non_owned = f1.get_commit_stats_non_owned(username, info)  
    
    
    reposi = "D:/Code/GitBlame/Bytsuki0/Python"
    f2.setup_nltk()
    sentiment_scores = []
    for r in repos:
        sentiment_scores.append(f2.get_user_activity_sentiment(r, n))
    print(sentiment_scores)

    status = f3.analyzer.get_results_as_json()
    print("Estatísticas individuais por repositório:\n", status)

    agregados = f3.analyzer.get_aggregate_stats()
    print("Estatísticas agregadas (últimos 365 dias):\n", json.dumps(agregados, indent=2))

if __name__ == "__main__":
    main()
