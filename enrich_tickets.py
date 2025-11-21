"""Script para enriquecer tickets já coletados com conversations e requester."""

import os
import sys
import pandas as pd
from pathlib import Path

# Adiciona o diretório raiz ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.api.collector import FreshdeskCollector


def main():
    """Enriquece tickets já coletados."""
    print("=" * 60)
    print("Enriquecimento de Tickets")
    print("=" * 60)
    
    # Caminho do arquivo de tickets
    tickets_path = "data/raw/tickets.json"
    
    if not os.path.exists(tickets_path):
        print(f"Erro: Arquivo {tickets_path} não encontrado.")
        print("Execute primeiro: python main.py")
        return
    
    # Carrega tickets
    print(f"\nCarregando tickets de {tickets_path}...")
    tickets_df = pd.read_json(tickets_path)
    print(f"✓ {len(tickets_df)} tickets carregados")
    
    # Verifica se já estão enriquecidos
    if 'conversations' in tickets_df.columns and 'requester_email' in tickets_df.columns:
        print("\n⚠ Tickets já parecem estar enriquecidos.")
        resposta = input("Deseja enriquecer novamente? (s/N): ")
        if resposta.lower() != 's':
            print("Operação cancelada.")
            return
    
    # Cria coletor
    print("\nConectando à API do Freshdesk...")
    try:
        collector = FreshdeskCollector()
    except Exception as e:
        print(f"Erro ao conectar: {e}")
        print("Verifique suas credenciais no arquivo .env")
        return
    
    # Enriquece tickets
    print("\nIniciando enriquecimento...")
    print("Usando processamento paralelo com controle automático de rate limit.")
    print("Cada ticket requer 1 requisição adicional à API.\n")
    
    # Permite configurar número de workers via variável de ambiente
    # Se não especificado, calcula automaticamente baseado no rate limit
    max_workers_env = os.getenv('ENRICH_MAX_WORKERS')
    max_workers = int(max_workers_env) if max_workers_env else None
    
    enriched_df = collector.enrich_tickets(
        tickets_df,
        includes=['conversations', 'requester'],
        progress=True,
        max_workers=max_workers
    )
    
    # Salva tickets enriquecidos
    backup_path = tickets_path.replace('.json', '_backup.json')
    print(f"\nCriando backup em {backup_path}...")
    tickets_df.to_json(backup_path, orient='records', date_format='iso', indent=2)
    
    print(f"Salvando tickets enriquecidos em {tickets_path}...")
    enriched_df.to_json(tickets_path, orient='records', date_format='iso', indent=2)
    
    print("\n" + "=" * 60)
    print("Enriquecimento concluído!")
    print("=" * 60)
    print(f"\nTickets enriquecidos: {len(enriched_df)}")
    print(f"Backup salvo em: {backup_path}")
    print(f"\nColunas adicionadas:")
    print("  - conversations: Lista de conversas do ticket")
    print("  - requester_email: Email do solicitante")
    print("  - requester_name: Nome do solicitante")
    print("  - requester_mobile: Mobile do solicitante")
    print("  - requester_phone: Telefone do solicitante")
    print("  - description_text: Texto da descrição do ticket")
    print("  - description: HTML da descrição do ticket")


if __name__ == "__main__":
    main()

