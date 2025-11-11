"""Exemplo de uso do sistema Freshdesk Analytics."""

import os
import sys
from pathlib import Path

# Adiciona o diretório ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.collector import FreshdeskCollector
from src.etl.pipeline import ETLPipeline
from src.etl.database import DatabaseManager
from src.nlp.sentiment import SentimentAnalyzer
from src.ml.churn_predictor import ChurnPredictor
from src.ml.satisfaction_predictor import SatisfactionPredictor
from src.metrics.calculator import MetricsCalculator


def exemplo_coleta():
    """Exemplo de coleta de dados."""
    print("=== Exemplo: Coleta de Dados ===")
    
    collector = FreshdeskCollector()
    
    # Coleta todos os dados
    data = collector.collect_all(output_dir="data/raw")
    
    print(f"Tickets coletados: {len(data.get('tickets', []))}")
    print(f"Contatos coletados: {len(data.get('contacts', []))}")
    print(f"Agentes coletados: {len(data.get('agents', []))}")


def exemplo_etl():
    """Exemplo de processamento ETL."""
    print("\n=== Exemplo: Processamento ETL ===")
    
    db = DatabaseManager("data/freshdesk.db")
    etl = ETLPipeline(db)
    
    # Carrega dados brutos
    import pandas as pd
    tickets_df = pd.read_json("data/raw/tickets.json")
    
    # Processa
    etl.run(tickets_df=tickets_df)
    print("✓ Dados processados e salvos no banco")


def exemplo_sentimentos():
    """Exemplo de análise de sentimentos."""
    print("\n=== Exemplo: Análise de Sentimentos ===")
    
    db = DatabaseManager("data/freshdesk.db")
    tickets_df = db.get_tickets()
    
    # Cria analisador
    sentiment_analyzer = SentimentAnalyzer()
    
    # Tenta carregar modelo existente
    try:
        sentiment_analyzer.load_model()
        print("✓ Modelo carregado")
    except:
        print("Treinando novo modelo...")
        texts, labels = sentiment_analyzer.create_training_data_from_tickets(tickets_df)
        if len(texts) > 0:
            sentiment_analyzer.train(texts, labels)
    
    # Analisa alguns textos
    textos_exemplo = [
        "Estou muito satisfeito com o atendimento!",
        "O serviço foi péssimo, não recomendo.",
        "Tudo funcionou corretamente."
    ]
    
    for texto in textos_exemplo:
        label, score = sentiment_analyzer.predict(texto)
        print(f"  '{texto}' -> {label} ({score:.2f})")


def exemplo_metricas():
    """Exemplo de cálculo de métricas."""
    print("\n=== Exemplo: Cálculo de Métricas ===")
    
    db = DatabaseManager("data/freshdesk.db")
    tickets_df = db.get_tickets()
    
    metrics_calc = MetricsCalculator(db)
    
    # CSAT
    csat = metrics_calc.calculate_csat(tickets_df)
    print(f"CSAT: {csat:.2f}" if csat else "CSAT: N/A")
    
    # NPS
    nps_data = metrics_calc.calculate_nps(tickets_df)
    if nps_data:
        print(f"NPS: {nps_data['nps']:.1f}")
        print(f"  Promotores: {nps_data['promoter_pct']:.1f}%")
        print(f"  Detratores: {nps_data['detractor_pct']:.1f}%")
    
    # SLA
    sla_data = metrics_calc.calculate_sla(tickets_df)
    if sla_data:
        print(f"SLA Compliance: {sla_data['sla_compliance_pct']:.1f}%")
        print(f"Tempo médio de resposta: {sla_data['avg_response_time_hours']:.1f}h")


def exemplo_ml():
    """Exemplo de modelos de ML."""
    print("\n=== Exemplo: Modelos de ML ===")
    
    db = DatabaseManager("data/freshdesk.db")
    tickets_df = db.get_tickets()
    
    # Churn
    print("Modelo de Churn:")
    churn_predictor = ChurnPredictor()
    try:
        churn_predictor.load_model()
        print("  ✓ Modelo carregado")
    except:
        print("  Treinando modelo...")
        churn_predictor.train(tickets_df)
    
    # Predições
    churn_predictions = churn_predictor.predict(tickets_df)
    high_risk = churn_predictions[churn_predictions['churn_probability'] > 0.7]
    print(f"  Clientes com alto risco de churn: {len(high_risk)}")
    
    # Satisfação
    print("\nModelo de Satisfação:")
    satisfaction_predictor = SatisfactionPredictor()
    try:
        satisfaction_predictor.load_model()
        print("  ✓ Modelo carregado")
    except:
        print("  Treinando modelo...")
        satisfaction_predictor.train(tickets_df)
    
    satisfaction_predictions = satisfaction_predictor.predict(tickets_df)
    print(f"  Satisfação média prevista: {satisfaction_predictions['satisfaction_score'].mean():.2f}")


if __name__ == "__main__":
    print("Exemplos de uso do Freshdesk Analytics")
    print("=" * 60)
    
    # Descomente as funções que deseja executar
    # exemplo_coleta()
    # exemplo_etl()
    # exemplo_sentimentos()
    # exemplo_metricas()
    # exemplo_ml()
    
    print("\nDescomente as funções em example_usage.py para executar os exemplos.")


