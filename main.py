"""Script principal para executar o pipeline completo de análise."""

import os
import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.collector import FreshdeskCollector
from src.etl.pipeline import ETLPipeline
from src.etl.database import DatabaseManager
from src.nlp.sentiment import SentimentAnalyzer
from src.ml.churn_predictor import ChurnPredictor
from src.ml.satisfaction_predictor import SatisfactionPredictor
from src.metrics.calculator import MetricsCalculator
from src.utils.helpers import setup_logging

# Configura logging
setup_logging()


def main():
    """Executa o pipeline completo de análise."""
    print("=" * 60)
    print("Freshdesk Analytics - Pipeline Completo")
    print("=" * 60)
    
    # 1. Coleta de dados
    print("\n[1/6] Coletando dados do Freshdesk...")
    try:
        collector = FreshdeskCollector()
        data = collector.collect_all(output_dir="data/raw")
        print(f"✓ Dados coletados: {len(data.get('tickets', []))} tickets")
    except Exception as e:
        print(f"✗ Erro na coleta: {e}")
        return
    
    # 2. ETL
    print("\n[2/6] Processando dados (ETL)...")
    try:
        db = DatabaseManager("data/freshdesk.db")
        etl = ETLPipeline(db)
        etl.run(
            tickets_df=data.get('tickets'),
            contacts_df=data.get('contacts'),
            agents_df=data.get('agents')
        )
    except Exception as e:
        print(f"✗ Erro no ETL: {e}")
        return
    
    # 3. Análise de sentimentos
    print("\n[3/6] Analisando sentimentos...")
    try:
        tickets_df = db.get_tickets()
        
        # Treina modelo se necessário
        sentiment_analyzer = SentimentAnalyzer()
        
        # Tenta carregar modelo existente
        try:
            sentiment_analyzer.load_model()
            print("✓ Modelo de sentimentos carregado")
        except:
            print("Treinando novo modelo de sentimentos...")
            texts, labels = sentiment_analyzer.create_training_data_from_tickets(tickets_df)
            
            if len(texts) > 0:
                sentiment_analyzer.train(texts, labels)
            else:
                print("⚠ Poucos dados para treinar. Usando modelo padrão.")
                # Cria modelo básico com dados sintéticos
                sentiment_analyzer._create_pipeline()
        
        # Prediz sentimentos
        if 'cleaned_text' in tickets_df.columns:
            predictions = sentiment_analyzer.predict_batch(tickets_df['cleaned_text'].tolist())
            tickets_df['sentiment_label'] = predictions['sentiment_label'].values
            tickets_df['sentiment_score'] = predictions['sentiment_score'].values
            
            # Atualiza banco
            db.insert_tickets(tickets_df, if_exists='replace')
            print(f"✓ Sentimentos calculados para {len(tickets_df)} tickets")
    except Exception as e:
        print(f"✗ Erro na análise de sentimentos: {e}")
    
    # 4. Modelos de ML
    print("\n[4/6] Treinando modelos de ML...")
    try:
        tickets_df = db.get_tickets()
        
        # Churn
        print("  - Modelo de Churn...")
        churn_predictor = ChurnPredictor()
        try:
            churn_predictor.load_model()
            print("  ✓ Modelo de churn carregado")
        except:
            churn_predictor.train(tickets_df)
            print("  ✓ Modelo de churn treinado")
        
        # Satisfação
        print("  - Modelo de Satisfação...")
        satisfaction_predictor = SatisfactionPredictor()
        try:
            satisfaction_predictor.load_model()
            print("  ✓ Modelo de satisfação carregado")
        except:
            satisfaction_predictor.train(tickets_df)
            print("  ✓ Modelo de satisfação treinado")
    except Exception as e:
        print(f"✗ Erro no treinamento de ML: {e}")
    
    # 5. Métricas
    print("\n[5/6] Calculando métricas e KPIs...")
    try:
        metrics_calc = MetricsCalculator(db)
        metrics = metrics_calc.calculate_all_metrics()
        print(f"✓ Métricas calculadas: {list(metrics.keys())}")
    except Exception as e:
        print(f"✗ Erro no cálculo de métricas: {e}")
    
    # 6. Resumo
    print("\n[6/6] Resumo final...")
    tickets_df = db.get_tickets()
    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")
    print(f"Total de tickets: {len(tickets_df)}")
    
    if 'sentiment_label' in tickets_df.columns:
        sentiment_dist = tickets_df['sentiment_label'].value_counts()
        print(f"\nSentimentos:")
        for label, count in sentiment_dist.items():
            print(f"  {label}: {count}")
    
    metrics_calc = MetricsCalculator(db)
    csat = metrics_calc.calculate_csat(tickets_df)
    if csat:
        print(f"\nCSAT: {csat:.2f}")
    
    nps_data = metrics_calc.calculate_nps(tickets_df)
    if nps_data:
        print(f"NPS: {nps_data['nps']:.1f}")
    
    print(f"\n{'='*60}")
    print("Pipeline concluído com sucesso!")
    print(f"{'='*60}")
    print("\nPara visualizar os resultados, execute:")
    print("  python run_dashboard.py")
    print("  ou")
    print("  streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()

