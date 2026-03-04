<h1 align="center">SourceSkillsMiner</h1>

<p align="center">
Sistema de mineração e análise de habilidades técnicas baseado em atividade pública no GitHub.
</p>

<hr>

<h2>Visão Geral</h2>

<p>
O <strong>SourceSkillsMiner</strong> é um conjunto modular de ferramentas desenvolvidas em Python para análise quantitativa e qualitativa de perfis de desenvolvedores no GitHub.
O sistema coleta dados públicos via API oficial, processa métricas estruturais e comportamentais e produz uma pontuação agregada representando o perfil técnico do usuário.
</p>

<h2>Arquitetura do Projeto</h2>

<pre><code>
SourceSkillsMiner/
│
├── OSSanaliser.py
├── ScoringSys.py
├── SentimentalAnaliser.py
├── StatusAnaliser.py
├── WorkTypeAnalyzer.py
├── commit_frequency.py
├── dashboard.py
├── gitblame.py
│
├── Bayers_Classifier/
├── Flame Graphs/
├── Research/
├── json/
│
├── RunParallel.ps1
├── RunParallel.sh
├── WorkTypeAnalyzer_README.md
└── config.ini (criado pelo usuário)
</code></pre>

<h2>Principais Componentes</h2>

<h3>OSSanaliser.py</h3>
<p>Responsável por coletar dados públicos do GitHub, incluindo repositórios, commits, issues e pull requests.</p>

<h3>ScoringSys.py</h3>
<p>Implementa o sistema de pontuação agregada integrando múltiplas métricas analíticas.</p>

<h3>SentimentalAnaliser.py</h3>
<p>Executa análise de sentimento em comentários e interações públicas.</p>

<h3>StatusAnaliser.py</h3>
<p>Analisa regularidade e padrões temporais de commits.</p>

<h3>WorkTypeAnalyzer.py</h3>
<p>Classifica o tipo de trabalho predominante (backend, frontend, data science, etc.) com base em padrões de importação e estrutura de código.</p>

<h2>Funcionalidades</h2>

<ul>
  <li>Extração de dados via GitHub API.</li>
  <li>Análise de frequência e consistência de commits.</li>
  <li>Análise de sentimento em interações públicas.</li>
  <li>Classificação técnica por área de atuação.</li>
  <li>Geração de score técnico consolidado.</li>
  <li>Execução paralela para múltiplos usuários.</li>
</ul>

<h2>Requisitos</h2>

<ul>
  <li>Python 3.8 ou superior</li>
  <li>Biblioteca <code>requests</code> e demais dependências internas</li>
  <li>Token de acesso GitHub</li>
</ul>

<h2>Instalação</h2>

<h3>Clonar repositório</h3>

<pre><code>git clone https://github.com/Bytsuki0/SourceSkillsMiner.git
cd SourceSkillsMiner
</code></pre>

<h3>Criar ambiente virtual</h3>

<pre><code>python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
</code></pre>

<h3>Instalar dependências</h3>

<pre><code>pip install -r requirements.txt
</code></pre>

<p>
Caso o arquivo <code>requirements.txt</code> não esteja disponível, instale manualmente as bibliotecas listadas nos imports dos scripts.
</p>

<h2>Configuração</h2>

<p>Criar arquivo <code>config.ini</code> na raiz do projeto:</p>

<pre><code>[github]
username = seu_usuario
token = seu_token_github
</code></pre>

<p>
Recomenda-se utilizar um Personal Access Token para evitar limitações de requisições.
</p>

<h2>Execução</h2>

<h3>Análise de usuário</h3>

<pre><code>python OSSanaliser.py --username usuario --token token
</code></pre>

ou

<pre><code>python ScoringSys.py --username usuario --token token
</code></pre>

<p>Resultados podem ser exportados para o diretório <code>json/</code>.</p>

<h3>Execução paralela</h3>

<pre><code>RunParallel.ps1    # Windows
RunParallel.sh     # Linux/macOS
</code></pre>

<h2>Boas Práticas</h2>

<ul>
  <li>Não versionar tokens ou credenciais.</li>
  <li>Utilizar ambiente virtual isolado.</li>
  <li>Respeitar limites de requisição da API do GitHub.</li>
  <li>Documentar alterações no sistema de scoring.</li>
</ul>

<h2>Contribuição</h2>

<ol>
  <li>Abrir uma issue descrevendo a proposta.</li>
  <li>Criar branch específica.</li>
  <li>Submeter Pull Request com descrição técnica detalhada.</li>
</ol>

<h2>Licença</h2>

<p>
Adicionar um arquivo <code>LICENSE</code> especificando os termos de uso (ex.: MIT ou Apache 2.0).
</p>

<hr>

<p align="center">
Documentação técnica mantida para uso acadêmico e analítico.
</p>
