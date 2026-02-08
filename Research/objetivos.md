# Modelo de teste para quantificação de Soft skills

## GitHub

### Tabela de entrada
O objetivo das condições é realizar uma filtragem estatística dos dados, evitando usuários com participação quase nula no GitHub. Isso evita inflação na média dos dados, garantindo que os usuários que passaram pela filtragem inicial tenham uma média mais significativa entre seus avaliados.

#### Critérios de seleção
- [X] Realizou um commit em um repositório que não era dono [Pontuação: 1]
- [X] Abriu um pull-request ou um issue em um repositório  [Pontuação: 1]
- [X] Teve um merge ou resolveu um issue em um repositório  [Pontuação: 1]
- [X] Criou mais de X linhas de codigo em seu histórico [Pontuação: 1]

- [X] Função de aprovação, caso participante obtenha 3 ou mais pontos ele é aprovado para ser pontuado pela proxima parte do teste

### Tabela de pontuação
Baseado na revisão bibliográfica, serão definidos pontos (atividades mensuráveis) que indicam Soft Skills, como ensinar novatos, mostrar compromisso, participação no GitHub e resolução de problemas.

#### Resolução de problemas
- [x] Resolveu X+ issues dentro de um repositório [Pontuação: 30 a 50: 1 issue resolvido = 30 pontos até 50 se 3 ou mais issues resolvido]
- [x] Abriu issues ou pull em repositórios [Pontuação: 30 a 50: 5 merge ou issues abertos = 30 pontos até 50 se 10 ou mais merges]
- [x] Teve X+ merge em Pull-request(s) em repositório [Pontuação: 30 a 50: 5 merge resolvido = 30 pontos até 50 se 10 ou mais merges]
- [x] Teve X+ commits em repositórios [Pontuação: 30 a 50: 100 commits = 30 pontos até 50 se 1000 ou mais commits]
- [ ] Foi marcado para resolver um problema de um repositório [Pontuação: 50: caso marcado para 1 ou mais issues]

#### Capacidade e vontade de ensinar participantes
- [x] Análise de personalidade e agressão (feito por um modelo R ou AI?)
- [ ] Contribuiu com pull-requests de novos participantes
- [ ] Deu suporte ou comentou em um issue
- [ ] Comentou no código em pull-requests
- [ ] Contribuiu com read-me e/ou arquivos para comunidade (Markdown ou txt)

#### Participação e compromisso
- [x] Alterou X números de linhas em um repositório
- [x] Participou de um repositório ao menos 1 vez ao mês por X meses
- [x] Participou de 2+ repositórios ao menos 1 vez ao mês por X meses
- [x] Possui sequência de contribuições de X+ vezes na semana no ano
- [x] Tem direitos de edição em um repositório 
- [x] Tem X% dos commits em um repositório

## StackOverflow   (requisitos e todos os dados para serem alterados depois)

### Tabela de entrada
Tabela de entrada similar ao GitHub, feita para filtragem de dados scrapados do StackOverflow. O modelo de entrada realiza filtragem inicial para aceitação de um perfil na pesquisa.

#### Critérios de seleção
- [ ] Participou em X+ tópicos
- [ ] Reputação no site de X+
- [ ] Possui X+ votos positivos em tópicos

### Tabela de pontuação
Baseado na revisão bibliográfica, serão analisados traços de personalidade que indicam Soft Skills, como persistência em tópicos, análise de personalidade, participação na comunidade e recepção por outros usuários.

#### Persistência na resolução de problemas
- [ ] Teve 2+ comentários em um único tópico
- [ ] Teve X% das participações de um tópico
- [ ] Possui X% dos votos positivos de um tópico

#### Capacidade de passar conhecimento
- [ ] Análise de personalidade e agressão
- [ ] Possui alta aprovação em diferentes tópicos
- [ ] Participou de X tópicos com problemas distintos

#### Participação e compromisso
- [ ] Participou de tópicos uma vez por semana em X meses
- [ ] Teve participação de X% em um mesmo tópico uma vez em X meses
- [ ] Possui sequência de contribuições de X+ no ano
- [ ] Participou de tópicos distintos X+ vezes em um ano

## Possíveis ferramentas para integração nas Avaliações de Personalidade (em Python)
- [VADER](https://github.com/cjhutto/vaderSentiment)
- [Empath](https://github.com/Ejhfast/empath-client)
- [Modelos GPT-J](https://github.com/EleutherAI/gpt-neo)
- [AFINN](https://github.com/fnielsen/afinn)

## Objetivo
Desenvolvimento e fine-tuning de um modelo ARIMA que crie correlação entre dados coletados pelas ferramentas (GitBlame, GitLog e Stack Exchange API).

O objetivo deste projeto é desenvolver e realizar o fine-tuning de um modelo ARIMA que seja capaz de estabelecer correlações entre dados coletados por ferramentas como GitBlame, GitLog e a API do Stack Exchange. O modelo será inspirado na abordagem apresentada no estudo *"Towards Mining OSS Skills from GitHub Activity"*, com o intuito de implementar um algoritmo de pontuação que atribua notas em diferentes tópicos, variando de 0 a 5.

Além disso, o projeto incluirá uma etapa de avaliação cruzada, na qual um questionário será aplicado. Nesse questionário, um participante (pessoa 1) julgará outro (pessoa 2) nos mesmos tópicos analisados pelo algoritmo. O objetivo principal dessa etapa é medir a precisão e a acurácia do algoritmo ao comparar suas pontuações com os resultados da pesquisa manual.

Os tópicos analisados estão abertos à definição, permitindo ajustes que priorizem áreas com maior impacto na precisão geral do estudo. Assim, o modelo poderá focar nos aspectos que apresentam maior correlação com as avaliações subjetivas, garantindo uma análise mais robusta e confiável.
