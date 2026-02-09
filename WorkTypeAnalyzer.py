import requests
import json
import time
import re
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple


class GitHubLanguageCommitAnalyzer:
    """
    Analyzes programming language usage across a user's GitHub repositories.
    Distributes commit and line change statistics proportionally based on 
    language byte counts in each repository.
    """
    
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }

    def _get_user_repositories(self):
        """Fetch all repositories for the user."""
        repos = []
        page = 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            repos.extend(data)
            page += 1
        return repos

    def _get_repo_languages(self, full_name):
        """Get language statistics for a repository."""
        url = f"{self.api_url}/repos/{full_name}/languages"
        resp = requests.get(url, headers=self.headers)
        if resp.status_code != 200:
            return {}
        return resp.json()

    def _get_commit_stats(self, full_name):
        """Get commit statistics for the user in a specific repository."""
        url = f"{self.api_url}/repos/{full_name}/stats/contributors"
        max_retries = 10
        for attempt in range(max_retries):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 202:
                time.sleep(2)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data:
                return None
            for contributor in data:
                author = contributor.get('author')
                if author and author.get('login', '').lower() == self.username.lower():
                    total_lines = sum(week.get('a', 0) + week.get('d', 0) for week in contributor.get('weeks', []))
                    total_commits = sum(week.get('c', 0) for week in contributor.get('weeks', []))
                    return total_lines, total_commits
        return None

    def analyze_language_usage(self) -> Dict[str, List[int]]:
        """
        Analyze language usage across all repositories.
        
        Returns:
            Dict mapping language names to [lines_changed, commits]
        """
        language_stats = defaultdict(lambda: [0, 0])  # language: [lines_changed, commits]
        repos = self._get_user_repositories()
        
        for repo in repos:
            full_name = repo['full_name']
            languages = self._get_repo_languages(full_name)
            commit_stats = self._get_commit_stats(full_name)
            
            if not commit_stats:
                continue
            
            lines_changed, commits = commit_stats
            total_bytes = sum(languages.values())
            
            if total_bytes == 0:
                continue
            
            # Distribute commits and lines proportionally based on language bytes
            for lang, bytes_count in languages.items():
                proportion = bytes_count / total_bytes
                language_stats[lang][0] += int(lines_changed * proportion)
                language_stats[lang][1] += int(commits * proportion)
        
        return dict(language_stats)
    
    def get_results_as_json(self) -> str:
        """Return analysis results as formatted JSON string."""
        data = self.analyze_language_usage()
        return json.dumps(data, indent=2)


class GitHubWorkTypeAnalyzer:
    """
    Analyzes import statements in code files to determine the type of work
    a developer does (frontend, backend, data analysis, DevOps, ML/AI, etc.).
    
    Examines the first 20 lines of code files across all repositories to
    extract import statements and classify them into work categories.
    """
    
    # Classification patterns for different work types
    WORK_TYPE_PATTERNS = {
        'Frontend': {
            'libraries': [
                'react', 'vue', 'angular', 'svelte', 'jquery', 'next', 'nuxt',
                'redux', 'mobx', 'recoil', 'zustand', 'styled-components',
                'emotion', 'sass', 'less', 'webpack', 'vite', 'parcel',
                'tailwind', 'bootstrap', 'material-ui', '@mui', 'chakra-ui',
                'antd', 'semantic-ui', 'framer-motion', 'gsap', 'd3',
                'chart.js', 'recharts', 'three', 'babylon', 'pixi'
            ],
            'keywords': ['component', 'jsx', 'tsx', 'css', 'html', 'dom', 'ui']
        },
        'Backend': {
            'libraries': [
                'express', 'koa', 'hapi', 'fastify', 'nest', 'django',
                'flask', 'fastapi', 'spring', 'gin', 'echo', 'fiber',
                'actix', 'rocket', 'rails', 'laravel', 'symfony',
                'sequelize', 'typeorm', 'prisma', 'mongoose', 'gorm',
                'sqlalchemy', 'hibernate', 'jwt', 'passport', 'bcrypt',
                'cors', 'helmet', 'morgan', 'dotenv', 'config'
            ],
            'keywords': ['server', 'api', 'route', 'middleware', 'controller', 'auth']
        },
        'Data_Analysis': {
            'libraries': [
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                'plotly', 'bokeh', 'altair', 'statsmodels', 'pingouin',
                'jupyter', 'notebook', 'ipython', 'xlrd', 'openpyxl',
                'pyarrow', 'dask', 'polars', 'datashader'
            ],
            'keywords': ['dataframe', 'analysis', 'statistics', 'visualization']
        },
        'Machine_Learning': {
            'libraries': [
                'tensorflow', 'keras', 'torch', 'pytorch', 'sklearn',
                'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
                'transformers', 'huggingface', 'jax', 'flax', 'optax',
                'mlflow', 'wandb', 'tensorboard', 'ray', 'dvc',
                'gym', 'stable-baselines', 'opencv', 'cv2', 'pillow',
                'albumentations', 'spacy', 'nltk', 'gensim'
            ],
            'keywords': ['model', 'training', 'neural', 'learning', 'predict', 'inference']
        },
        'DevOps': {
            'libraries': [
                'docker', 'kubernetes', 'kubectl', 'terraform', 'ansible',
                'chef', 'puppet', 'jenkins', 'gitlab', 'circleci',
                'travis', 'prometheus', 'grafana', 'elasticsearch', 'logstash',
                'kibana', 'redis', 'rabbitmq', 'kafka', 'celery',
                'airflow', 'dagster', 'prefect', 'nginx', 'apache'
            ],
            'keywords': ['deploy', 'container', 'orchestration', 'pipeline', 'infra']
        },
        'Database': {
            'libraries': [
                'mysql', 'postgresql', 'psycopg2', 'pymysql', 'sqlite3',
                'mongodb', 'pymongo', 'redis', 'cassandra', 'dynamodb',
                'neo4j', 'influxdb', 'timescaledb', 'clickhouse',
                'elasticsearch', 'prisma', 'knex', 'sequelize'
            ],
            'keywords': ['query', 'schema', 'migration', 'orm', 'database']
        },
        'Mobile': {
            'libraries': [
                'react-native', 'expo', 'flutter', 'dart', 'swift',
                'swiftui', 'uikit', 'kotlin', 'jetpack', 'compose',
                'ionic', 'cordova', 'capacitor', 'xamarin'
            ],
            'keywords': ['mobile', 'ios', 'android', 'native', 'app']
        },
        'Game_Development': {
            'libraries': [
                'unity', 'unreal', 'godot', 'pygame', 'phaser',
                'pixi', 'babylon', 'three', 'webgl', 'opengl',
                'vulkan', 'directx', 'monogame', 'raylib'
            ],
            'keywords': ['game', 'physics', 'render', 'sprite', 'scene']
        },
        'Testing': {
            'libraries': [
                'jest', 'mocha', 'chai', 'jasmine', 'pytest', 'unittest',
                'nose', 'testify', 'junit', 'testng', 'selenium',
                'puppeteer', 'playwright', 'cypress', 'webdriver',
                'locust', 'jmeter', 'gatling', 'mockito', 'sinon'
            ],
            'keywords': ['test', 'mock', 'assert', 'expect', 'describe', 'spec']
        },
        'Security': {
            'libraries': [
                'cryptography', 'pycryptodome', 'bcrypt', 'jwt', 'oauth',
                'openssl', 'hashlib', 'secrets', 'vault', 'keyring',
                'passlib', 'argon2', 'scrypt'
            ],
            'keywords': ['encrypt', 'decrypt', 'hash', 'auth', 'security', 'crypto']
        }
    }
    
    # File extensions to analyze
    CODE_EXTENSIONS = [
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
        '.cpp', '.c', '.h', '.hpp', '.cs', '.php', '.rb', '.swift',
        '.kt', '.scala', '.r', '.m', '.dart', '.lua'
    ]
    
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.api_url = "https://api.github.com"
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.import_cache = defaultdict(set)  # Cache imports per work type
        self.file_count = defaultdict(int)  # Count files per work type
    
    def _get_user_repositories(self) -> List[dict]:
        """Fetch all repositories for the user."""
        repos = []
        page = 1
        while True:
            url = f"{self.api_url}/users/{self.username}/repos?per_page=100&page={page}"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code != 200:
                break
            try:
                data = resp.json()
            except (ValueError, json.JSONDecodeError):
                break
            if not isinstance(data, list) or not data:
                break
            repos.extend(data)
            page += 1
        return repos
    
    def _get_repo_tree(self, full_name: str, branch: str = 'main') -> Optional[List[dict]]:
        """Get the file tree of a repository."""
        # Try main branch first, then master
        for branch_name in [branch, 'master', 'develop']:
            url = f"{self.api_url}/repos/{full_name}/git/trees/{branch_name}?recursive=1"
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    return data.get('tree', [])
                except (ValueError, json.JSONDecodeError):
                    continue
        return None
    
    def _get_file_content(self, full_name: str, file_path: str) -> Optional[str]:
        """Get the content of a file from a repository."""
        url = f"{self.api_url}/repos/{full_name}/contents/{file_path}"
        resp = requests.get(url, headers=self.headers)
        
        if resp.status_code != 200:
            return None
        
        try:
            data = resp.json()
            if data.get('encoding') == 'base64':
                import base64
                content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                return content
        except Exception:
            return None
        
        return None
    
    def _extract_imports_from_lines(self, lines: List[str], language: str) -> Set[str]:
        """Extract import statements from the first 20 lines of code."""
        imports = set()
        
        for line in lines[:20]:
            line = line.strip()
            
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            
            # Python imports
            if language in ['.py']:
                # import module
                match = re.match(r'^import\s+([a-zA-Z0-9_.,\s]+)', line)
                if match:
                    modules = match.group(1).split(',')
                    for module in modules:
                        imports.add(module.strip().split()[0].split('.')[0])
                
                # from module import ...
                match = re.match(r'^from\s+([a-zA-Z0-9_.,\s]+)\s+import', line)
                if match:
                    module = match.group(1).strip().split('.')[0]
                    imports.add(module)
            
            # JavaScript/TypeScript imports
            elif language in ['.js', '.ts', '.jsx', '.tsx']:
                # import ... from 'module'
                match = re.search(r"from\s+['\"]([^'\"]+)['\"]", line)
                if match:
                    module = match.group(1).split('/')[0]
                    if not module.startswith('.'):
                        imports.add(module.lstrip('@'))
                
                # const/var = require('module')
                match = re.search(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", line)
                if match:
                    module = match.group(1).split('/')[0]
                    if not module.startswith('.'):
                        imports.add(module.lstrip('@'))
            
            # Go imports
            elif language in ['.go']:
                match = re.search(r'import\s+"([^"]+)"', line)
                if match:
                    module = match.group(1).split('/')[-1]
                    imports.add(module)
            
            # Java imports
            elif language in ['.java']:
                match = re.match(r'^import\s+([a-zA-Z0-9_.]+);', line)
                if match:
                    parts = match.group(1).split('.')
                    if len(parts) >= 2:
                        imports.add(parts[0] + '.' + parts[1])
            
            # Ruby imports/requires
            elif language in ['.rb']:
                match = re.match(r"^require\s+['\"]([^'\"]+)['\"]", line)
                if match:
                    imports.add(match.group(1))
            
            # PHP imports
            elif language in ['.php']:
                match = re.match(r'^use\s+([a-zA-Z0-9_\\\\]+)', line)
                if match:
                    parts = match.group(1).split('\\')
                    if parts:
                        imports.add(parts[0])
        
        return imports
    
    def _classify_imports(self, imports: Set[str]) -> Dict[str, int]:
        """Classify imports into work type categories."""
        work_type_scores = defaultdict(int)
        
        for imp in imports:
            imp_lower = imp.lower()
            
            for work_type, patterns in self.WORK_TYPE_PATTERNS.items():
                # Check against libraries
                for library in patterns['libraries']:
                    if library in imp_lower or imp_lower in library:
                        work_type_scores[work_type] += 2
                        self.import_cache[work_type].add(imp)
                        break
                
                # Check against keywords (lower weight)
                for keyword in patterns['keywords']:
                    if keyword in imp_lower:
                        work_type_scores[work_type] += 1
                        break
        
        return dict(work_type_scores)
    
    def analyze_work_types(self, max_repos: int = 100, max_files_per_repo: int = 100) -> Dict[str, any]:
        """
        Analyze import statements across repositories to determine work types.
        
        Args:
            max_repos: Maximum number of repositories to analyze
            max_files_per_repo: Maximum number of files to analyze per repository
        
        Returns:
            Dict with work type statistics and classifications
        """
        repos = self._get_user_repositories()[:max_repos]
        work_type_totals = defaultdict(int)
        repo_classifications = []
        total_files_analyzed = 0
        
        print(f"Analyzing {len(repos)} repositories...")
        
        for repo_idx, repo in enumerate(repos):
            full_name = repo['full_name']
            print(f"  [{repo_idx + 1}/{len(repos)}] Analyzing {full_name}...")
            
            tree = self._get_repo_tree(full_name)
            if not tree:
                continue
            
            # Filter code files
            code_files = [
                item for item in tree
                if item['type'] == 'blob' and
                any(item['path'].endswith(ext) for ext in self.CODE_EXTENSIONS)
            ][:max_files_per_repo]
            
            repo_work_scores = defaultdict(int)
            files_analyzed_in_repo = 0
            
            for file_item in code_files:
                file_path = file_item['path']
                
                # Skip files in common directories to ignore
                skip_dirs = ['node_modules', 'vendor', 'dist', 'build', '.git', 'test', 'tests']
                if any(skip_dir in file_path for skip_dir in skip_dirs):
                    continue
                
                content = self._get_file_content(full_name, file_path)
                if not content:
                    continue
                
                lines = content.split('\n')
                extension = '.' + file_path.split('.')[-1] if '.' in file_path else ''
                
                imports = self._extract_imports_from_lines(lines, extension)
                if imports:
                    classifications = self._classify_imports(imports)
                    for work_type, score in classifications.items():
                        repo_work_scores[work_type] += score
                        work_type_totals[work_type] += score
                        self.file_count[work_type] += 1
                    
                    files_analyzed_in_repo += 1
                    total_files_analyzed += 1
                
                # Rate limiting
                time.sleep(0.1)
            
            if repo_work_scores:
                # Determine primary work type for this repo
                primary_type = max(repo_work_scores, key=repo_work_scores.get)
                repo_classifications.append({
                    'repository': full_name,
                    'primary_work_type': primary_type,
                    'scores': dict(repo_work_scores),
                    'files_analyzed': files_analyzed_in_repo
                })
        
        # Calculate percentages
        total_score = sum(work_type_totals.values())
        work_type_percentages = {}
        if total_score > 0:
            work_type_percentages = {
                wt: round((score / total_score) * 100, 2)
                for wt, score in work_type_totals.items()
            }
        
        # Sort by score
        sorted_work_types = sorted(
            work_type_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'username': self.username,
            'analysis_date': datetime.now().isoformat(),
            'total_files_analyzed': total_files_analyzed,
            'total_repos_analyzed': len(repos),
            'work_type_scores': dict(sorted_work_types),
            'work_type_percentages': work_type_percentages,
            'file_count_by_type': dict(self.file_count),
            'primary_work_type': sorted_work_types[0][0] if sorted_work_types else 'Unknown',
            'repository_classifications': repo_classifications,
            'common_imports_by_type': {
                wt: sorted(list(imports)[:10])
                for wt, imports in self.import_cache.items()
            }
        }
    
    def get_results_as_json(self, **kwargs) -> str:
        """Return analysis results as formatted JSON string."""
        data = self.analyze_work_types(**kwargs)
        return json.dumps(data, indent=2)
    
    def get_summary(self, **kwargs) -> Dict[str, any]:
        """Get a simplified summary of work types."""
        data = self.analyze_work_types(**kwargs)
        
        return {
            'username': self.username,
            'primary_work_type': data['primary_work_type'],
            'work_type_distribution': data['work_type_percentages'],
            'total_files_analyzed': data['total_files_analyzed'],
            'top_3_work_types': [
                {'type': wt, 'percentage': data['work_type_percentages'].get(wt, 0)}
                for wt, _ in sorted(
                    data['work_type_scores'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ]
        }


if __name__ == "__main__":
    import configparser
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    username = config['github']['username']
    token = config['github']['token']
    
    print("LANGUAGE USAGE ANALYSIS")
    lang_analyzer = GitHubLanguageCommitAnalyzer(username, token)
    print(lang_analyzer.get_results_as_json())
    
    print("WORK TYPE ANALYSIS (Based on Imports)")
    work_analyzer = GitHubWorkTypeAnalyzer(username, token)
    
    # Get summary for quick overview
    summary = work_analyzer.get_summary(max_repos=100, max_files_per_repo=100)
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    
    # Uncomment for full analysis
    # print("\nFULL ANALYSIS:")
    # print(work_analyzer.get_results_as_json(max_repos=20, max_files_per_repo=30))
