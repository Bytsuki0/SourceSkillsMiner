import os
import csv
import re
import time
import requests
import platform
import subprocess
import configparser as cfgparser
import json  # Adicionado para tratar JSONDecodeError
import gitblame as gb

gb.data_rows = []

# Carrega configuração do GitHub
config = cfgparser.ConfigParser()
config.read('config.ini')
GITHUB_TOKEN = config.get('github', 'token', fallback='')

HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'github-public-client'
}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'


def is_git_repo(path):
    """Verifica se um diretório é um repositório Git."""
    try:
        subprocess.run(
            ["git", "-C", path, "status"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


# Todos os comentários em commits, issues e pull requests
def get_user_activity(username, repoName, num_events):
    activity_info = "\n"
    # Issues
    issues_url = f"https://api.github.com/repos/{username}/{repoName}/issues/comments"
    issues_response = requests.get(issues_url, headers=HEADERS)
    issues_info = "Comentários em Issues:\n"
    if issues_response.ok:
        for comment in issues_response.json()[:num_events]:
            issues_info += (
                f"- Comentário: {comment.get('body', 'Sem conteúdo')}\n"
                f"  Autor: {comment['user'].get('login')} ({comment['user'].get('html_url')})\n"
                f"  Criado em: {comment.get('created_at')}\n"
            )
    else:
        issues_info += f"Erro ao obter comentários em Issues: {issues_response.status_code}\n"
    activity_info += issues_info + "\n"

    # Pull Request comments
    pr_comments_url = f"https://api.github.com/repos/{username}/{repoName}/pulls/comments"
    pr_comments_response = requests.get(pr_comments_url, headers=HEADERS)
    pr_info = "Comentários em Pull Requests:\n"
    if pr_comments_response.ok:
        for comment in pr_comments_response.json()[:num_events]:
            pr_info += (
                f"- Comentário: {comment.get('body', 'Sem conteúdo')}\n"
                f"  Autor: {comment['user'].get('login')} ({comment['user'].get('html_url')})\n"
                f"  Criado em: {comment.get('created_at')}\n"
            )
    else:
        pr_info += f"Erro ao obter comentários em Pull Requests: {pr_comments_response.status_code}\n"
    activity_info += pr_info + "\n"

    # Commit comments
    commit_comments_url = f"https://api.github.com/repos/{username}/{repoName}/comments"
    commit_comments_response = requests.get(commit_comments_url, headers=HEADERS)
    commit_info = "Comentários em Commits:\n"
    if commit_comments_response.ok:
        for comment in commit_comments_response.json()[:num_events]:
            commit_info += (
                f"- Comentário: {comment.get('body', 'Sem conteúdo')}\n"
                f"  Autor: {comment['user'].get('login')} ({comment['user'].get('html_url')})\n"
                f"  Criado em: {comment.get('created_at')}\n"
            )
    else:
        commit_info += f"Erro ao obter comentários em Commits: {commit_comments_response.status_code}\n"
    activity_info += commit_info
    return activity_info


# Todos os últimos issues e pull requests abertos
def get_user_opened_issues_and_prs(username, num_events):
    issues_url = f"https://api.github.com/search/issues?q=is:issue+author:{username}+state:open"
    prs_url = f"https://api.github.com/search/issues?q=is:pr+author:{username}+state:open"
    result = {"issues": [], "prs": []}

    issues_response = requests.get(issues_url, headers=HEADERS)
    if issues_response.ok:
        for issue in issues_response.json().get('items', [])[:num_events]:
            result['issues'].append({
                'title': issue['title'],
                'repo': '/'.join(issue['repository_url'].split('/')[-2:]),
                'created_at': issue['created_at']
            })

    pr_response = requests.get(prs_url, headers=HEADERS)
    if pr_response.ok:
        for pr in pr_response.json().get('items', [])[:num_events]:
            result['prs'].append({
                'title': pr['title'],
                'repo': '/'.join(pr['repository_url'].split('/')[-2:]),
                'created_at': pr['created_at']
            })

    return result


# Todos os últimos issues e pull requests resolvidos
def get_user_resolved_issues_and_prs(username, num_events):
    issues_url = f"https://api.github.com/search/issues?q=is:issue+author:{username}+state:closed"
    prs_url = f"https://api.github.com/search/issues?q=is:pr+author:{username}+state:closed"
    merged_url = f"https://api.github.com/search/issues?q=is:pr+author:{username}+state:closed+is:merged"
    result = {"resolved_issues": [], "closed_prs": [], "merged_prs_count": 0}

    issues_response = requests.get(issues_url, headers=HEADERS)
    if issues_response.ok:
        for issue in issues_response.json().get('items', [])[:num_events]:
            result['resolved_issues'].append({
                'title': issue['title'],
                'repo': '/'.join(issue['repository_url'].split('/')[-2:]),
                'closed_at': issue['closed_at']
            })

    pr_response = requests.get(prs_url, headers=HEADERS)
    if pr_response.ok:
        for pr in pr_response.json().get('items', [])[:num_events]:
            result['closed_prs'].append({
                'title': pr['title'],
                'repo': '/'.join(pr['repository_url'].split('/')[-2:]),
                'closed_at': pr['closed_at']
            })

    merged_response = requests.get(merged_url, headers=HEADERS)
    if merged_response.ok:
        data = merged_response.json()
        result['merged_prs_count'] = data.get('total_count', 0)

    return result


# Estatísticas totais de commits
def get_commit_stats_total(username):
    repo_url = f"https://api.github.com/users/{username}/repos?per_page=100"
    response = requests.get(repo_url, headers=HEADERS)
    total_commits = 0
    stats_info = []
    if not response.ok:
        print(f"Erro ao buscar repositórios: {response.status_code}")
        return []

    for repo in response.json():
        owner = repo.get('owner', {}).get('login')
        reponame = repo.get('name')
        full_name = f"{owner}/{reponame}"  # Agora inclui o owner
        contribs_url = f"https://api.github.com/repos/{full_name}/contributors"
        contribs_resp = requests.get(contribs_url, headers=HEADERS)

        if not contribs_resp.ok:
            print(f"Falha ao obter contribuidores para {full_name}: {contribs_resp.status_code}")
            continue

        try:
            contributors = contribs_resp.json()
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON de {full_name}")
            continue

        for c in contributors:
            if c.get('login') == username:
                contrib_count = c.get('contributions', 0)
                total_commits += contrib_count
                stats_info.append((full_name, contrib_count))

    print(f"Total de commits públicos: {total_commits}")
    return total_commits


# Estatísticas de commits em repositórios não próprios
def get_commit_stats_non_owned(username, commits):
    stats = commits
    non_owned = [(repo, cnt) for repo, cnt in stats if not repo.startswith(f"{username}/")]
    return non_owned


# Estatísticas de participação em repositórios
def get_repo_participation_stats(username):
    headers = HEADERS
    page = 1
    all_repos = []
    while True:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&page={page}"
        response = requests.get(url, headers)
        if not response.ok:
            print(f"Erro ao buscar repositórios: {response.status_code}")
            return []

        page_data = response.json()
        if not page_data:
            break

        all_repos.extend(page_data)
        page += 1

    owned_repos = []
    non_owned_repos = []

    for repo in all_repos:
        fullname = repo.get('full_name')
        if repo['owner']['login'] == username:
            owned_repos.append(fullname)
        else:
            non_owned_repos.append(fullname)

    print(f"Total de repositórios públicos: {len(all_repos)}")
    print(f"Repositórios próprios: {len(owned_repos)}")
    print(f"Repositórios colaborando: {len(non_owned_repos)}")

    return [owned_repos,non_owned_repos]


def main():
    username = "Gictorbit"

    print(get_repo_participation_stats(username))






