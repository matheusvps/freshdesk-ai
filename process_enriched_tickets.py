"""Script para processar tickets já enriquecidos e executar o pipeline completo."""

import os
import sys
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
    """Processa tickets enriquecidos e executa o pipeline completo."""
    print("=" * 60)
    print("Processamento de Tickets Enriquecidos - Pipeline Completo")
    print("=" * 60)
    
    # Caminhos dos arquivos
    tickets_path = "data/raw/tickets_enriched.json"
    ratings_path = "data/raw/satisfaction_ratings.json"
    
    # Verifica se o arquivo de tickets existe
    if not os.path.exists(tickets_path):
        print(f"\n[ERRO] Arquivo não encontrado: {tickets_path}")
        print("Execute primeiro o script collect_enriched_tickets.py")
        return
    
    # 1. Carrega tickets enriquecidos
    print("\n[1/6] Carregando tickets enriquecidos...")
    try:
        tickets_df = pd.read_json(tickets_path)
        print(f"[OK] {len(tickets_df)} tickets carregados")
        
        # Verifica se tem as colunas de enriquecimento
        enrichment_cols = ['conversations', 'requester_email']
        has_enrichment = all(col in tickets_df.columns for col in enrichment_cols)
        if has_enrichment:
            enriched_count = tickets_df[enrichment_cols].notna().any(axis=1).sum()
            print(f"  {enriched_count} tickets estão enriquecidos")
        else:
            print("  [AVISO] Aviso: Algumas colunas de enriquecimento não foram encontradas")
    except Exception as e:
        print(f"[ERRO] Erro ao carregar tickets: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 1.5. Carrega satisfaction ratings se disponível
    ratings_df = None
    if os.path.exists(ratings_path):
        print("\nCarregando satisfaction ratings...")
        try:
            ratings_df = pd.read_json(ratings_path)
            print(f"[OK] {len(ratings_df)} satisfaction ratings carregados")
        except Exception as e:
            print(f"[AVISO] Erro ao carregar satisfaction ratings: {e}")
            ratings_df = None
    else:
        print("\n[AVISO] Arquivo de satisfaction ratings não encontrado. CSAT/NPS podem não estar disponíveis.")
        print("  Para coletar satisfaction ratings, execute:")
        print("    collector = FreshdeskCollector()")
        print("    ratings = collector.get_satisfaction_ratings(save_path='data/raw/satisfaction_ratings.json')")
    
    # 2. ETL - Processa e armazena no banco
    print("\n[2/6] Processando dados (ETL)...")
    try:
        db = DatabaseManager("data/freshdesk.db")
        etl = ETLPipeline(db)
        etl.run(tickets_df=tickets_df, satisfaction_ratings_df=ratings_df)
        print("[OK] ETL concluído")
    except Exception as e:
        print(f"[ERRO] Erro no ETL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Análise de sentimentos
    print("\n[3/6] Analisando sentimentos...")
    try:
        tickets_df = db.get_tickets()
        
        # Verifica se já tem sentimentos calculados
        if 'sentiment_label' in tickets_df.columns and tickets_df['sentiment_label'].notna().any():
            print("  [AVISO] Sentimentos já calculados. Pulando análise...")
        else:
            # Treina modelo se necessário
            sentiment_analyzer = SentimentAnalyzer()
            
            # Tenta carregar modelo existente
            try:
                sentiment_analyzer.load_model()
                print("  [OK] Modelo de sentimentos carregado")
            except:
                print("  Treinando novo modelo de sentimentos...")
                try:
                    texts, labels = sentiment_analyzer.create_training_data_from_tickets(tickets_df)
                    
                    if len(texts) > 0:
                        sentiment_analyzer.train(texts, labels)
                        print("  [OK] Modelo treinado")
                    else:
                        print("  [AVISO] Poucos dados para treinar. Usando modelo padrão.")
                        sentiment_analyzer._create_pipeline()
                except (KeyError, ValueError) as e:
                    print(f"  [AVISO] Não foi possível criar dados de treinamento: {e}")
                    print("  [AVISO] Usando modelo padrão sem treinamento.")
                    sentiment_analyzer._create_pipeline()
            
            # Prediz sentimentos
            if 'cleaned_text' in tickets_df.columns and tickets_df['cleaned_text'].notna().any():
                print("  Calculando sentimentos para todos os tickets...")
                predictions = sentiment_analyzer.predict_batch(tickets_df['cleaned_text'].tolist())
                tickets_df['sentiment_label'] = predictions['sentiment_label'].values
                tickets_df['sentiment_score'] = predictions['sentiment_score'].values
                
                # Atualiza banco
                db.insert_tickets(tickets_df, if_exists='replace')
                print(f"  [OK] Sentimentos calculados para {len(tickets_df)} tickets")
            else:
                print("  [AVISO] Nenhum texto limpo encontrado para análise de sentimentos")
    except Exception as e:
        print(f"[ERRO] Erro na análise de sentimentos: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Modelos de ML
    print("\n[4/6] Treinando modelos de ML...")
    try:
        tickets_df = db.get_tickets()
        
        # Churn
        print("  - Modelo de Churn...")
        churn_predictor = ChurnPredictor()
        try:
            churn_predictor.load_model()
            print("  [OK] Modelo de churn carregado")
        except:
            try:
                churn_predictor.train(tickets_df)
                print("  [OK] Modelo de churn treinado")
            except (ValueError, KeyError) as e:
                print(f"  [AVISO] Não foi possível treinar modelo de churn: {e}")
        
        # Satisfação
        print("  - Modelo de Satisfação...")
        satisfaction_predictor = SatisfactionPredictor()
        try:
            satisfaction_predictor.load_model()
            print("  [OK] Modelo de satisfação carregado")
        except:
            try:
                satisfaction_predictor.train(tickets_df)
                print("  [OK] Modelo de satisfação treinado")
            except (ValueError, KeyError) as e:
                print(f"  [AVISO] Não foi possível treinar modelo de satisfação: {e}")
    except Exception as e:
        print(f"[ERRO] Erro no treinamento de ML: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Métricas
    print("\n[5/6] Calculando métricas e KPIs...")
    try:
        metrics_calc = MetricsCalculator(db)
        metrics = metrics_calc.calculate_all_metrics()
        print(f"[OK] Métricas calculadas: {list(metrics.keys())}")
        
        # Mostra algumas métricas principais
        if 'csat' in metrics:
            print(f"  CSAT: {metrics['csat']:.2f}")
        else:
            print("  CSAT: N/A (nenhum dado de satisfação encontrado)")
        
        if 'nps' in metrics:
            print(f"  NPS: {metrics['nps']:.1f}")
        else:
            print("  NPS: N/A (nenhum dado de satisfação encontrado)")
        
        if 'ces' in metrics:
            print(f"  CES: {metrics['ces']:.2f}")
    except Exception as e:
        print(f"[ERRO] Erro no cálculo de métricas: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Resumo final
    print("\n[6/6] Resumo final...")
    try:
        tickets_df = db.get_tickets()
        print(f"\n{'='*60}")
        print("RESUMO")
        print(f"{'='*60}")
        print(f"Total de tickets processados: {len(tickets_df)}")
        
        # Distribuição de Status
        if 'status_name' in tickets_df.columns:
            status_dist = tickets_df['status_name'].value_counts()
            print(f"\nDistribuição de Status:")
            for status, count in status_dist.items():
                pct = (count / len(tickets_df)) * 100
                print(f"  {status}: {count} ({pct:.1f}%)")
        elif 'status' in tickets_df.columns:
            from src.utils.status_mapper import get_status_name
            status_dist = tickets_df['status'].value_counts()
            print(f"\nDistribuição de Status:")
            for status_val, count in status_dist.items():
                pct = (count / len(tickets_df)) * 100
                status_name = get_status_name(status_val)
                print(f"  {status_name}: {count} ({pct:.1f}%)")
        
        if 'sentiment_label' in tickets_df.columns:
            sentiment_dist = tickets_df['sentiment_label'].value_counts()
            print(f"\nDistribuição de Sentimentos:")
            for label, count in sentiment_dist.items():
                pct = (count / len(tickets_df)) * 100
                print(f"  {label}: {count} ({pct:.1f}%)")
        
        metrics_calc = MetricsCalculator(db)
        
        csat = metrics_calc.calculate_csat(tickets_df)
        if csat is not None:
            print(f"\nCSAT: {csat:.2f}")
        else:
            print("\nCSAT: N/A (nenhum dado de satisfação encontrado)")
            if 'satisfaction_rating' in tickets_df.columns:
                ratings_count = tickets_df['satisfaction_rating'].notna().sum()
                print(f"  Total de tickets com rating: {ratings_count}/{len(tickets_df)}")
        
        nps_data = metrics_calc.calculate_nps(tickets_df)
        if nps_data:
            print(f"\nNPS: {nps_data['nps']:.1f}")
            print(f"  Promotores: {nps_data.get('promoters', 0)} ({nps_data.get('promoter_pct', 0):.1f}%)")
            print(f"  Neutros: {nps_data.get('passives', 0)} ({nps_data.get('passive_pct', 0):.1f}%)")
            print(f"  Detratores: {nps_data.get('detractors', 0)} ({nps_data.get('detractor_pct', 0):.1f}%)")
            print(f"  Total de respostas: {nps_data.get('total_responses', 0)}")
        else:
            print("\nNPS: N/A (nenhum dado de satisfação encontrado)")
            if 'satisfaction_rating' in tickets_df.columns:
                ratings_count = tickets_df['satisfaction_rating'].notna().sum()
                print(f"  Total de tickets com rating: {ratings_count}/{len(tickets_df)}")
        
        ces = metrics_calc.calculate_ces(tickets_df)
        if ces:
            print(f"CES: {ces:.2f}")
        
        print(f"\n{'='*60}")
        print("Pipeline concluído com sucesso! ")
        print(f"{'='*60}")
        print("\nPróximos passos:")
        print("  1. Visualize os resultados no dashboard:")
        print("     python run_dashboard.py")
        print("     ou")
        print("     streamlit run src/dashboard/app.py")
        print("\n  2. Explore os dados no banco:")
        print(f"     Arquivo: data/freshdesk.db")
        print("\n  3. Use os modelos treinados para predições:")
        print("     Veja example_usage.py para exemplos")
        
    except Exception as e:
        print(f"[ERRO] Erro ao gerar resumo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

