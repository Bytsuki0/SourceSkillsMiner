# WorkTypeAnalyzer Documentation

This module contains two classes for analyzing GitHub repositories:

## 1. GitHubLanguageCommitAnalyzer

Analyzes programming language usage across a user's repositories by distributing commits and line changes proportionally based on language byte counts.

### Usage:

```python
from WorkTypeAnalyzer import GitHubLanguageCommitAnalyzer

analyzer = GitHubLanguageCommitAnalyzer(username, token)

# Get language statistics as dict
results = analyzer.analyze_language_usage()
# Returns: {'Python': [5000, 120], 'JavaScript': [3000, 80], ...}
#           where each value is [lines_changed, commits]

# Get as JSON string
json_results = analyzer.get_results_as_json()
```

## 2. GitHubWorkTypeAnalyzer

Analyzes the first 20 lines of code files to extract import statements and classify the type of work being done.

### Work Types Detected:

- **Frontend**: React, Vue, Angular, jQuery, etc.
- **Backend**: Express, Django, Flask, FastAPI, etc.
- **Data_Analysis**: pandas, numpy, matplotlib, seaborn, etc.
- **Machine_Learning**: TensorFlow, PyTorch, scikit-learn, etc.
- **DevOps**: Docker, Kubernetes, Terraform, Ansible, etc.
- **Database**: MySQL, PostgreSQL, MongoDB, Redis, etc.
- **Mobile**: React Native, Flutter, Swift, Kotlin, etc.
- **Game_Development**: Unity, Pygame, Phaser, etc.
- **Testing**: Jest, Pytest, Selenium, Cypress, etc.
- **Security**: Cryptography, JWT, OAuth, bcrypt, etc.

### Usage:

```python
from WorkTypeAnalyzer import GitHubWorkTypeAnalyzer

analyzer = GitHubWorkTypeAnalyzer(username, token)

# Get quick summary (recommended for initial analysis)
summary = analyzer.get_summary(max_repos=20, max_files_per_repo=30)
print(summary)
# Returns:
# {
#   'username': 'yourusername',
#   'primary_work_type': 'Backend',
#   'work_type_distribution': {
#     'Backend': 45.5,
#     'Frontend': 30.2,
#     'Data_Analysis': 15.3,
#     ...
#   },
#   'total_files_analyzed': 450,
#   'top_3_work_types': [...]
# }

# Get full detailed analysis
full_results = analyzer.analyze_work_types(max_repos=50, max_files_per_repo=50)

# Get as JSON string
json_results = analyzer.get_results_as_json(max_repos=20, max_files_per_repo=30)
```

### Parameters:

- `max_repos`: Maximum number of repositories to analyze (default: 50)
- `max_files_per_repo`: Maximum code files to analyze per repository (default: 50)

### Output Structure:

```json
{
  "username": "username",
  "analysis_date": "2026-02-08T...",
  "total_files_analyzed": 450,
  "total_repos_analyzed": 20,
  "work_type_scores": {
    "Backend": 350,
    "Frontend": 280,
    "Data_Analysis": 150
  },
  "work_type_percentages": {
    "Backend": 44.87,
    "Frontend": 35.90,
    "Data_Analysis": 19.23
  },
  "file_count_by_type": {
    "Backend": 120,
    "Frontend": 95,
    "Data_Analysis": 45
  },
  "primary_work_type": "Backend",
  "repository_classifications": [
    {
      "repository": "user/repo1",
      "primary_work_type": "Backend",
      "scores": {"Backend": 50, "Database": 20},
      "files_analyzed": 15
    }
  ],
  "common_imports_by_type": {
    "Backend": ["express", "django", "flask", ...],
    "Frontend": ["react", "vue", ...],
    ...
  }
}
```

## How Import Analysis Works:

1. **File Discovery**: Scans repository tree for code files (`.py`, `.js`, `.ts`, `.java`, `.go`, etc.)
2. **Import Extraction**: Reads first 20 lines of each file to extract import statements
3. **Pattern Matching**: Matches imports against predefined library patterns for each work type
4. **Scoring**: Awards points based on library matches (exact match = 2 points, keyword match = 1 point)
5. **Classification**: Aggregates scores to determine primary work type per repository and overall

## Integration with ScoringSys:

To integrate with your existing scoring system:

```python
# In ScoringSys.py, add:
from WorkTypeAnalyzer import GitHubWorkTypeAnalyzer

def compute_work_diversity_score(username: str, token: str) -> Dict:
    """Computes score based on diversity of work types."""
    try:
        analyzer = GitHubWorkTypeAnalyzer(username, token)
        summary = analyzer.get_summary(max_repos=20, max_files_per_repo=30)
        
        # Score based on number of work types (diversity is good)
        work_types = summary['work_type_distribution']
        num_work_types = len([k for k, v in work_types.items() if v > 5])  # >5% contribution
        
        # Normalize: 1 type = 0.0, 5+ types = 1.0
        diversity_score = min(1.0, num_work_types / 5.0)
        
        return {
            'score': _to_signed01(diversity_score),
            'details': {
                'primary_work_type': summary['primary_work_type'],
                'work_type_count': num_work_types,
                'distribution': work_types
            }
        }
    except Exception as e:
        return {'score': 0.0, 'details': {'error': str(e)}}
```

## Performance Considerations:

- The analyzer caches imports to avoid redundant API calls
- Rate limiting: 0.1s delay between file fetches, 0.5s for user queries
- Recommended limits: 20 repos, 30 files per repo for quick analysis
- Full analysis (50 repos, 50 files) may take 10-15 minutes depending on API rate limits

## Notes:

- Skips common build directories: `node_modules`, `vendor`, `dist`, `build`, `test`
- Handles multiple languages: Python, JavaScript/TypeScript, Java, Go, Ruby, PHP
- Import patterns are continuously improved based on common libraries
- GitHub API rate limits apply (5000 requests/hour for authenticated users)
