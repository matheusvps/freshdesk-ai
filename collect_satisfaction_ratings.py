"""Script para coletar Satisfaction Ratings do Freshdesk."""

import os
import sys
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.collector import FreshdeskCollector
from src.utils.helpers import setup_logging

# Configura logging
setup_logging()


def main():
    """Coleta satisfaction ratings do Freshdesk."""
    print("=" * 60)
    print("Coleta de Satisfaction Ratings")
    print("=" * 60)
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    ratings_path = os.path.join(output_dir, 'satisfaction_ratings.json')
    
    try:
        # Cria coletor
        print("\nConectando à API do Freshdesk...")
        collector = FreshdeskCollector()
        print("Conectado com sucesso")
        
        # Coleta satisfaction ratings
        print("\nColetando satisfaction ratings...")
        print("Isso pode levar algum tempo dependendo do número de ratings.")
        
        ratings_df = collector.get_satisfaction_ratings(save_path=ratings_path)
        
        if len(ratings_df) > 0:
            print(f"\n" + "=" * 60)
            print("RESUMO")
            print("=" * 60)
            print(f"Total de satisfaction ratings coletados: {len(ratings_df)}")
            print(f"Arquivo salvo em: {ratings_path}")
            
            # Estatísticas
            if 'rating_label' in ratings_df.columns:
                print("\nDistribuição de Ratings:")
                rating_dist = ratings_df['rating_label'].value_counts()
                for label, count in rating_dist.items():
                    pct = (count / len(ratings_df)) * 100
                    print(f"  {label}: {count} ({pct:.1f}%)")
            
            if 'ticket_id' in ratings_df.columns:
                unique_tickets = ratings_df['ticket_id'].nunique()
                print(f"\nTickets únicos com rating: {unique_tickets}")
            
            print("\n" + "=" * 60)
            print("Coleta concluída com sucesso!")
            print("=" * 60)
            print("\nPróximos passos:")
            print("  1. Execute process_enriched_tickets.py para processar os ratings")
            print("  2. Os ratings serão mesclados com os tickets automaticamente")
            print("  3. CSAT e NPS serão calculados com base nos ratings coletados")
        else:
            print("\nNenhum satisfaction rating encontrado.")
            print("Isso pode significar que:")
            print("  - Não há ratings de satisfação no Freshdesk")
            print("  - Os tickets não têm surveys de satisfação configurados")
            print("  - É necessário configurar surveys no Freshdesk primeiro")
        
    except KeyboardInterrupt:
        print("\n\nInterrupção detectada pelo usuário.")
        if os.path.exists(ratings_path):
            print(f"Progresso parcial salvo em: {ratings_path}")
        print("Você pode rodar o script novamente para continuar.")
        return
    except Exception as e:
        print(f"\nErro na coleta: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

