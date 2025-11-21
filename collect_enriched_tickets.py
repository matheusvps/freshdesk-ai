"""Script para coletar apenas tickets enriquecidos do Freshdesk."""

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
    """Coleta apenas tickets enriquecidos do Freshdesk."""
    print("=" * 60)
    print("Coleta de Tickets Enriquecidos")
    print("=" * 60)
    
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    tickets_path = os.path.join(output_dir, 'tickets_enriched.json')
    
    try:
        # Cria coletor
        print("\nConectando à API do Freshdesk...")
        collector = FreshdeskCollector()
        print("Conectado com sucesso")
        
        # Verifica se já existe arquivo de tickets básicos
        tickets_basic_path = os.path.join(output_dir, 'tickets_basic.json')
        tickets_df = None
        
        if os.path.exists(tickets_basic_path):
            print(f"\nArquivo de tickets básicos encontrado: {tickets_basic_path}")
            print("Carregando tickets básicos...")
            try:
                tickets_df = pd.read_json(tickets_basic_path)
                print(f"  {len(tickets_df)} tickets básicos carregados")
            except Exception as e:
                print(f"  Erro ao carregar arquivo básico: {e}")
                tickets_df = None
        
        # Se não tem tickets básicos carregados, coleta do zero
        if tickets_df is None or len(tickets_df) == 0:
            print("\nColetando tickets básicos da API...")
            print("Isso pode levar algum tempo dependendo do número de tickets.")
            tickets_df = collector.get_tickets()
            print(f"\n{len(tickets_df)} tickets básicos coletados")
            
            # Salva tickets básicos
            tickets_df.to_json(tickets_basic_path, orient='records', date_format='iso', indent=2)
            print(f"Tickets básicos salvos em {tickets_basic_path}")
        
        # Verifica se já existe arquivo parcial de enriquecidos
        enriched_partial_df = None
        if os.path.exists(tickets_path):
            print(f"\nArquivo parcial de enriquecidos encontrado: {tickets_path}")
            try:
                enriched_partial_df = pd.read_json(tickets_path)
                print(f"  {len(enriched_partial_df)} tickets encontrados no arquivo parcial")
                
                # Verifica quantos já estão enriquecidos
                enrichment_columns = ['conversations', 'requester_email']
                if all(col in enriched_partial_df.columns for col in enrichment_columns):
                    has_enrichment = enriched_partial_df[enrichment_columns].notna().any(axis=1)
                    enriched_count = has_enrichment.sum()
                    print(f"  {enriched_count} tickets já estão enriquecidos")
                    print(f"  {len(enriched_partial_df) - enriched_count} tickets ainda precisam ser enriquecidos")
            except Exception as e:
                print(f"  Erro ao carregar arquivo parcial: {e}")
                enriched_partial_df = None
        
        # Enriquece os tickets com salvamento progressivo
        if len(tickets_df) > 0:
            print("\n" + "=" * 60)
            print("Iniciando enriquecimento de tickets...")
            print("=" * 60)
            print("Rate limit: 50 GETs por minuto para enriquecimento")
            print("Cada ticket requer 1 requisição adicional para enriquecimento.")
            print("O progresso será salvo automaticamente a cada 100 tickets.")
            print("Se houver interrupção, você pode rodar o script novamente")
            print("e ele continuará de onde parou (pulando tickets já enriquecidos).\n")
            
            # Se tem arquivo parcial, usa ele como base e adiciona tickets básicos faltantes
            if enriched_partial_df is not None and len(enriched_partial_df) > 0:
                # Garante que temos todos os tickets básicos
                if 'id' in tickets_df.columns and 'id' in enriched_partial_df.columns:
                    # Encontra tickets básicos que não estão no arquivo parcial
                    partial_ids = set(enriched_partial_df['id'].astype(str))
                    basic_ids = set(tickets_df['id'].astype(str))
                    missing_ids = basic_ids - partial_ids
                    
                    if missing_ids:
                        missing_tickets = tickets_df[tickets_df['id'].astype(str).isin(missing_ids)]
                        # Combina: arquivo parcial (já tem alguns enriquecidos) + tickets básicos faltantes
                        tickets_df = pd.concat([enriched_partial_df, missing_tickets], ignore_index=True)
                        print(f"  {len(missing_tickets)} tickets básicos adicionados ao arquivo parcial")
                    else:
                        # Todos os tickets básicos já estão no arquivo parcial
                        tickets_df = enriched_partial_df
                        print("  Todos os tickets básicos já estão no arquivo parcial")
                else:
                    tickets_df = enriched_partial_df
            else:
                print("  Nenhum arquivo parcial encontrado. Começando do zero.")
            
            # O método enrich_tickets já tem skip_enriched=True, então vai pular automaticamente
            # os tickets que já têm conversations ou requester_email preenchidos
            enriched_df = collector.enrich_tickets(
                tickets_df,
                includes=['conversations', 'requester'],
                progress=True,
                save_path=tickets_path,
                save_interval=100,
                skip_enriched=True  # Pula tickets já enriquecidos automaticamente
            )
            
            print(f"\n" + "=" * 60)
            print("RESUMO")
            print("=" * 60)
            print(f"Total de tickets enriquecidos: {len(enriched_df)}")
            print(f"Arquivo salvo em: {tickets_path}")
            print("\nColunas adicionadas:")
            print("  - conversations: Lista de conversas do ticket")
            print("  - requester_email: Email do solicitante")
            print("  - requester_name: Nome do solicitante")
            print("  - requester_mobile: Mobile do solicitante")
            print("  - requester_phone: Telefone do solicitante")
            print("  - description_text: Texto da descrição do ticket")
            print("  - description: HTML da descrição do ticket")
        else:
            print("\nNenhum ticket encontrado para enriquecer.")
        
        print("\n" + "=" * 60)
        print("Coleta concluida com sucesso!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupcao detectada pelo usuario.")
        if os.path.exists(tickets_path):
            print(f"Progresso parcial salvo em: {tickets_path}")
        print("Voce pode rodar o script novamente para continuar de onde parou.")
        return
    except Exception as e:
        print(f"\nErro na coleta: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(tickets_path):
            print(f"\nProgresso parcial salvo em: {tickets_path}")
            print("Voce pode rodar o script novamente para continuar de onde parou.")
        return


if __name__ == "__main__":
    main()

