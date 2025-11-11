"""Coletor de dados do Freshdesk via API REST."""

import os
import json
import requests
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()


class FreshdeskCollector:
    """Classe para coletar dados do Freshdesk via API REST."""
    
    def __init__(self, api_path: Optional[str] = None, auth: Optional[str] = None):
        """
        Inicializa o coletor.
        
        Args:
            api_path: URL base da API do Freshdesk. Se None, lê de FRESHDESK_API_PATH
            auth: Token de autenticação (Basic Auth). Se None, lê de FRESHDESK_AUTH
        """
        self.api_path = api_path or os.getenv('FRESHDESK_API_PATH')
        self.auth = auth or os.getenv('FRESHDESK_AUTH')
        
        if not self.api_path:
            raise ValueError(
                "API path não fornecida. Configure FRESHDESK_API_PATH no .env ou passe api_path no construtor."
            )
        
        if not self.auth:
            raise ValueError(
                "Autenticação não fornecida. Configure FRESHDESK_AUTH no .env ou passe auth no construtor."
            )
        
        # Remove barra final se houver
        self.api_path = self.api_path.rstrip('/')
        
        # Se não terminar com /api/v2, adiciona
        if not self.api_path.endswith('/api/v2'):
            self.base_url = f"{self.api_path}/api/v2"
        else:
            self.base_url = self.api_path
        
        # Configura autenticação básica
        self.session = requests.Session()
        # Para Basic Auth, o formato é "username:password" ou apenas o token
        # Se for token, usa como username e 'X' como password (padrão Freshdesk)
        if ':' in self.auth:
            username, password = self.auth.split(':', 1)
            self.session.auth = (username, password)
        else:
            # Assume que é apenas o token
            self.session.auth = (self.auth, 'X')
        
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None):
        """
        Faz requisição à API com tratamento de erros.
        
        Returns:
            Resposta JSON (pode ser Dict ou List dependendo do endpoint)
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição {url}: {e}")
            if hasattr(e.response, 'text'):
                print(f"Resposta: {e.response.text}")
            raise
    
    def get_tickets(
        self, 
        per_page: int = 100,
        updated_since: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Coleta todos os tickets com paginação.
        
        Args:
            per_page: Número de tickets por página
            updated_since: Data no formato YYYY-MM-DD para filtrar tickets atualizados
            save_path: Caminho para salvar JSON (opcional)
        
        Returns:
            DataFrame com todos os tickets
        """
        all_tickets = []
        page = 1
        
        params = {'per_page': per_page}
        if updated_since:
            params['updated_since'] = updated_since
        
        print("Coletando tickets...")
        
        while True:
            params['page'] = page
            try:
                tickets = self._make_request('tickets', params=params)
                
                if not tickets:
                    break
                
                all_tickets.extend(tickets)
                print(f"Página {page}: {len(tickets)} tickets coletados")
                
                if len(tickets) < per_page:
                    break
                
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Erro ao coletar página {page}: {e}")
                break
        
        df = pd.DataFrame(all_tickets)
        
        if save_path:
            df.to_json(save_path, orient='records', date_format='iso', indent=2)
            print(f"Tickets salvos em {save_path}")
        
        return df
    
    def get_contacts(
        self,
        per_page: int = 100,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Coleta todos os contatos (clientes).
        
        Args:
            per_page: Número de contatos por página
            save_path: Caminho para salvar JSON (opcional)
        
        Returns:
            DataFrame com todos os contatos
        """
        all_contacts = []
        page = 1
        
        print("Coletando contatos...")
        
        while True:
            params = {'per_page': per_page, 'page': page}
            try:
                contacts = self._make_request('contacts', params=params)
                
                if not contacts:
                    break
                
                all_contacts.extend(contacts)
                print(f"Página {page}: {len(contacts)} contatos coletados")
                
                if len(contacts) < per_page:
                    break
                
                page += 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Erro ao coletar página {page}: {e}")
                break
        
        df = pd.DataFrame(all_contacts)
        
        if save_path:
            df.to_json(save_path, orient='records', date_format='iso', indent=2)
            print(f"Contatos salvos em {save_path}")
        
        return df
    
    def get_agents(
        self,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Coleta todos os agentes.
        
        Args:
            save_path: Caminho para salvar JSON (opcional)
        
        Returns:
            DataFrame com todos os agentes
        """
        print("Coletando agentes...")
        
        try:
            agents = self._make_request('agents')
            df = pd.DataFrame(agents)
            
            if save_path:
                df.to_json(save_path, orient='records', date_format='iso', indent=2)
                print(f"Agentes salvos em {save_path}")
            
            return df
        except Exception as e:
            print(f"Erro ao coletar agentes: {e}")
            return pd.DataFrame()
    
    def get_ticket_details(
        self, 
        ticket_id: int, 
        includes: Optional[List[str]] = None
    ) -> Dict:
        """
        Coleta detalhes de um ticket específico com includes.
        
        Args:
            ticket_id: ID do ticket
            includes: Lista de includes (ex: ['conversations', 'requester'])
        
        Returns:
            Dicionário com detalhes do ticket
        """
        if includes is None:
            includes = ['conversations', 'requester']
        
        params = {'include': ','.join(includes)}
        
        try:
            ticket = self._make_request(f'tickets/{ticket_id}', params=params)
            return ticket
        except Exception as e:
            print(f"Erro ao coletar detalhes do ticket {ticket_id}: {e}")
            return {}
    
    def get_ticket_conversations(self, ticket_id: int) -> List[Dict]:
        """Coleta conversas de um ticket específico."""
        try:
            conversations = self._make_request(f'tickets/{ticket_id}/conversations')
            return conversations
        except Exception as e:
            print(f"Erro ao coletar conversas do ticket {ticket_id}: {e}")
            return []
    
    def enrich_tickets(
        self,
        tickets_df: pd.DataFrame,
        includes: Optional[List[str]] = None,
        progress: bool = True
    ) -> pd.DataFrame:
        """
        Enriquece tickets com informações adicionais (conversations, requester).
        
        Args:
            tickets_df: DataFrame com tickets básicos
            includes: Lista de includes (padrão: ['conversations', 'requester'])
            progress: Se True, mostra barra de progresso
        
        Returns:
            DataFrame com tickets enriquecidos
        """
        if includes is None:
            includes = ['conversations', 'requester']
        
        if len(tickets_df) == 0:
            return tickets_df
        
        print(f"\nEnriquecendo {len(tickets_df)} tickets com {', '.join(includes)}...")
        
        enriched_tickets = []
        
        # Itera sobre os tickets
        iterator = tickets_df.iterrows()
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=len(tickets_df), desc="Enriquecendo tickets")
        
        for idx, ticket in iterator:
            ticket_dict = ticket.to_dict()
            ticket_id = ticket_dict.get('id')
            
            if not ticket_id:
                enriched_tickets.append(ticket_dict)
                continue
            
            try:
                # Obtém detalhes do ticket
                details = self.get_ticket_details(ticket_id, includes=includes)
                
                # Mescla os detalhes no ticket original
                if details:
                    # Adiciona conversations se existir
                    if 'conversations' in includes and 'conversations' in details:
                        ticket_dict['conversations'] = details['conversations']
                    
                    # Adiciona requester se existir
                    if 'requester' in includes and 'requester' in details:
                        requester = details['requester']
                        ticket_dict['requester_email'] = requester.get('email')
                        ticket_dict['requester_name'] = requester.get('name')
                        ticket_dict['requester_mobile'] = requester.get('mobile')
                        ticket_dict['requester_phone'] = requester.get('phone')
                    
                    # Adiciona description_text se existir (conteúdo do ticket)
                    if 'description_text' in details:
                        ticket_dict['description_text'] = details['description_text']
                    
                    # Adiciona description (HTML) se existir
                    if 'description' in details:
                        ticket_dict['description'] = details['description']
                
                enriched_tickets.append(ticket_dict)
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"\nErro ao enriquecer ticket {ticket_id}: {e}")
                # Mantém o ticket original mesmo se houver erro
                enriched_tickets.append(ticket_dict)
        
        enriched_df = pd.DataFrame(enriched_tickets)
        print(f"✓ {len(enriched_df)} tickets enriquecidos")
        
        return enriched_df
    
    def collect_all(
        self,
        output_dir: str = "data/raw",
        updated_since: Optional[str] = None,
        enrich_tickets: bool = True,
        includes: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Coleta todos os dados disponíveis.
        
        Args:
            output_dir: Diretório para salvar os dados
            updated_since: Data para filtrar tickets atualizados
            enrich_tickets: Se True, enriquece tickets com conversations e requester
            includes: Lista de includes para enriquecimento (padrão: ['conversations', 'requester'])
        
        Returns:
            Dicionário com DataFrames de tickets, contatos e agentes
        """
        os.makedirs(output_dir, exist_ok=True)
        
        data = {}
        
        # Coleta tickets básicos
        tickets_path = os.path.join(output_dir, 'tickets.json')
        tickets_df = self.get_tickets(
            updated_since=updated_since,
            save_path=None  # Não salva ainda, vai salvar depois do enriquecimento
        )
        
        # Enriquece tickets se solicitado
        if enrich_tickets and len(tickets_df) > 0:
            tickets_df = self.enrich_tickets(tickets_df, includes=includes)
        
        # Salva tickets (enriquecidos ou não)
        if len(tickets_df) > 0:
            tickets_df.to_json(tickets_path, orient='records', date_format='iso', indent=2)
            print(f"Tickets salvos em {tickets_path}")
        
        data['tickets'] = tickets_df
        
        # Coleta contatos
        contacts_path = os.path.join(output_dir, 'contacts.json')
        data['contacts'] = self.get_contacts(save_path=contacts_path)
        
        # Coleta agentes
        agents_path = os.path.join(output_dir, 'agents.json')
        data['agents'] = self.get_agents(save_path=agents_path)
        
        return data

