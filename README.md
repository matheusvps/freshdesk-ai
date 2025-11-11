# Freshdesk Analytics

Sistema completo de análise de dados de atendimento de clientes (Freshdesk) com processamento local, incluindo coleta, processamento, armazenamento, análise de sentimentos, cálculo de métricas e geração de insights estratégicos.

## 🎯 Características

- ✅ **100% Local**: Todo processamento ocorre localmente, sem APIs externas
- 📊 **Análise de Sentimentos**: Modelos de NLP locais (scikit-learn + TF-IDF)
- 🤖 **Machine Learning**: Modelos para predição de churn e satisfação
- 🔍 **Explainable AI**: SHAP e LIME para interpretabilidade
- 📈 **Métricas e KPIs**: CSAT, NPS, CES, SLA e mais
- 📱 **Dashboard Interativo**: Visualização com Streamlit

## 📁 Estrutura do Projeto

```
freshdesk-ai/
│
├── data/                # Dados brutos e tratados
│   ├── raw/            # Dados coletados da API
│   └── processed/      # Dados processados
│
├── notebooks/           # Experimentos e protótipos (Jupyter)
│
├── models/              # Modelos treinados (.pkl, .joblib)
│
├── src/
│   ├── api/            # Coleta de dados via API REST do Freshdesk
│   ├── etl/           # ETL, limpeza e enriquecimento de dados
│   ├── nlp/           # Análise de sentimentos e classificação textual
│   ├── ml/            # Modelos de churn, satisfação e XAI
│   ├── metrics/       # Cálculo de KPIs (CSAT, NPS, CES, SLA)
│   ├── dashboard/     # Visualização (Streamlit)
│   └── utils/         # Funções auxiliares
│
├── main.py            # Script principal do pipeline
├── run_dashboard.py   # Script para executar o dashboard
├── example_usage.py   # Exemplos de uso
├── requirements.txt   # Dependências Python
├── setup.py          # Configuração do pacote
└── env.example       # Exemplo de configuração (.env)
```

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd freshdesk-ai
```

### 2. Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Copie o arquivo `env.example` para `.env` e preencha com suas credenciais:

```bash
cp env.example .env
```

Edite o `.env`:

```env
FRESHDESK_API_PATH=https://suaempresa.freshdesk.com
FRESHDESK_AUTH=seu_token_aqui
```

### 5. Baixe modelos de NLP (opcional)

Para melhor performance em análise de sentimentos em português:

```bash
python -m spacy download pt_core_news_sm
```

## 📖 Uso

### Pipeline Completo

Execute o pipeline completo de análise:

```bash
python main.py
```

Este script:
1. Coleta dados do Freshdesk via API
2. Enriquece tickets com conversations e requester (requer 2 requisições adicionais por ticket)
3. Processa e limpa os dados (ETL)
4. Analisa sentimentos dos tickets
5. Treina modelos de ML (churn e satisfação)
6. Calcula métricas e KPIs
7. Salva tudo no banco de dados SQLite

**Nota sobre enriquecimento de tickets**: Por padrão, o sistema enriquece automaticamente os tickets coletados com:
- **Conversations**: Todas as conversas do ticket (até 10 por requisição)
- **Requester**: Informações do solicitante (email, nome, telefone)
- **Description**: Conteúdo completo do ticket inserido pelo usuário

Isso requer 2 requisições adicionais à API para cada ticket, então pode levar algum tempo dependendo do número de tickets.

### Enriquecer Tickets Já Coletados

Se você já coletou tickets e quer enriquecê-los posteriormente:

```bash
python enrich_tickets.py
```

Este script:
- Carrega tickets de `data/raw/tickets.json`
- Enriquece com conversations e requester
- Cria backup automático antes de sobrescrever
- Salva os tickets enriquecidos

### Dashboard Interativo

Visualize os resultados no dashboard:

```bash
python run_dashboard.py
```

Ou diretamente com Streamlit:

```bash
streamlit run src/dashboard/app.py
```

### Uso Programático

```python
from src.api.collector import FreshdeskCollector
from src.etl.pipeline import ETLPipeline
from src.etl.database import DatabaseManager
from src.nlp.sentiment import SentimentAnalyzer
from src.metrics.calculator import MetricsCalculator

# Coleta dados
collector = FreshdeskCollector()
data = collector.collect_all()

# Processa dados
db = DatabaseManager("data/freshdesk.db")
etl = ETLPipeline(db)
etl.run(tickets_df=data['tickets'])

# Analisa sentimentos
sentiment_analyzer = SentimentAnalyzer()
tickets_df = db.get_tickets()
predictions = sentiment_analyzer.predict_batch(tickets_df['cleaned_text'].tolist())

# Calcula métricas
metrics_calc = MetricsCalculator(db)
csat = metrics_calc.calculate_csat(tickets_df)
nps = metrics_calc.calculate_nps(tickets_df)
```

## 📊 Métricas Calculadas

### CSAT (Customer Satisfaction Score)
Média das notas de satisfação dos clientes (escala 1-5).

### NPS (Net Promoter Score)
Percentual de promotores menos percentual de detratores.

### CES (Customer Effort Score)
Medida do esforço do cliente baseada em reaberturas e interações.

### SLA (Service Level Agreement)
- Tempo médio de resposta
- Taxa de conformidade com SLA
- Tickets dentro do prazo

### Taxa de Reclamação
Número de tickets por produto/serviço.

## 🤖 Modelos de Machine Learning

### Predição de Churn
Modelo que identifica clientes com risco de cancelamento baseado em:
- Histórico de tickets
- Tempo de resposta
- Sentimento dos tickets
- Reaberturas

### Predição de Satisfação
Modelo que prevê satisfação do cliente baseado em:
- Histórico de interações
- Sentimento médio
- Tempo de resposta
- Prioridade dos tickets

## 🔍 Explainable AI (XAI)

O projeto inclui suporte para:
- **SHAP**: Explicação de importância de features
- **LIME**: Explicação local de previsões

## 🛠️ Tecnologias

- **Python 3.8+**
- **Pandas**: Manipulação de dados
- **scikit-learn**: Machine Learning
- **SQLAlchemy**: ORM e banco de dados
- **Streamlit**: Dashboard interativo
- **Plotly**: Visualizações
- **SHAP/LIME**: Explainable AI

## 📝 Notas

- Todos os modelos são treinados localmente
- Dados são armazenados em SQLite por padrão
- Suporta autenticação Basic Auth do Freshdesk
- Remove automaticamente PII (dados pessoais) dos textos

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto é de código aberto e está disponível sob a licença MIT.


