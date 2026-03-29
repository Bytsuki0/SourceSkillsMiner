# SourceSkillsMiner — Web Dashboard

Full-stack developer profile analysis tool.

## Project layout

```
SourceSkillsMiner/
├── ScoringSys.py                     ← mines + scores a GitHub user
├── config_main.ini                   ← GitHub token(s)
├── json/                             ← scored profiles saved here
│
├── Bayers_Classifier/
│   ├── classify_profile.py           ← Naive Bayes classifier
│   └── models/
│       └── developer_classifier.joblib
│
├── backend/
│   ├── api.py                        ← Flask API (port 5000)
│   └── requirements.txt
│
└── frontend/
    ├── server.js                     ← Node.js proxy (port 3000)
    ├── package.json
    └── public/
        └── index.html                ← Dashboard UI
```

## Setup

### 1. Python backend

```bash
pip install flask flask-cors
# (joblib, scikit-learn, requests already required by other scripts)
```

### 2. Node.js frontend

```bash
cd frontend
npm install
```

## Running

Start both servers in separate terminals from the **project root**:

```bash
# Terminal 1 — Python backend
python backend/api.py

# Terminal 2 — Node.js frontend
cd frontend && npm start
```

Open **http://localhost:3000** in your browser.

## Configuration

`config_main.ini` (in project root) must have:

```ini
[github]
token = ghp_yourPersonalAccessToken
```

The token needs `public_repo` and `read:user` scopes.

## How it works

1. You type a GitHub username and click **Analyze**.
2. The Node.js server forwards the request to Flask.
3. Flask writes a temporary `config.ini`, runs `ScoringSys.py`, then `classify_profile.py`.
4. The combined result is streamed back via Server-Sent Events.
5. The dashboard renders scores, classification probabilities, language usage, libraries, and sentiment.

## Retraining the classifier

```bash
python generate_synthetic.py --per-category 50 --output synthetic.csv
python train_classifier.py --input synthetic.csv
```

The updated model is saved to `Bayers_Classifier/models/developer_classifier.joblib`.
