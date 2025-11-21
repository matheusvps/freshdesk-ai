"""Coletor de dados do Freshdesk via API REST."""

import os
import json
import requests
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, local
from dotenv import load_dotenv

from .rate_limiter import RateLimiter, get_request_type, RequestType

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
        
        # Thread-local storage para sessões thread-safe
        self._thread_local = local()
        
        # Rate limiter para controlar requisições
        self.rate_limiter = RateLimiter()
    
    def _get_thread_session(self):
        """Obtém ou cria uma sessão HTTP thread-local para thread safety."""
        if not hasattr(self._thread_local, 'session'):
            session = requests.Session()
            # Configura autenticação básica
            if ':' in self.auth:
                username, password = self.auth.split(':', 1)
                session.auth = (username, password)
            else:
                session.auth = (self.auth, 'X')
            session.headers.update({'Content-Type': 'application/json'})
            self._thread_local.session = session
        return self._thread_local.session
    
    def _make_request(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None, 
        method: str = 'GET',
        use_thread_local: bool = False
    ):
        """
        Faz requisição à API com tratamento de erros e rate limiting.
        
        Args:
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            method: Método HTTP (GET, POST, PUT, etc.)
            use_thread_local: Se True, usa sessão thread-local (para uso em threads)
        
        Returns:
            Resposta JSON (pode ser Dict ou List dependendo do endpoint)
        """
        # Determina tipo de requisição para rate limiting
        req_type = get_request_type(endpoint, method)
        
        # Espera se necessário (rate limiting)
        self.rate_limiter.wait_if_needed(req_type)
        
        url = f"{self.base_url}/{endpoint}"
        
        # Usa sessão thread-local se solicitado (para thread safety)
        session = self._get_thread_session() if use_thread_local else self.session
        
        try:
            # Faz a requisição
            if method.upper() == 'GET':
                response = session.get(url, params=params)
            elif method.upper() == 'POST':
                response = session.post(url, json=params)
            elif method.upper() in ('PUT', 'PATCH'):
                response = session.request(method.upper(), url, json=params)
            else:
                response = session.request(method.upper(), url, params=params)
            
            response.raise_for_status()
            
            # Atualiza rate limiter com headers da resposta
            response_headers = {k: v for k, v in response.headers.items()}
            self.rate_limiter.record_request(req_type, response_headers)
            
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Se for erro 429 (Too Many Requests), espera e tenta novamente
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                retry_after_header = e.response.headers.get('Retry-After', '60')
                
                try:
                    retry_after_value = int(retry_after_header)
                    # Se o valor for muito grande (provavelmente um timestamp Unix), calcula diferença
                    current_time = int(time.time())
                    if retry_after_value > current_time:
                        # É um timestamp Unix, calcula diferença
                        retry_after = retry_after_value - current_time
                    else:
                        # É segundos direto
                        retry_after = retry_after_value
                except (ValueError, TypeError):
                    retry_after = 60
                
                # Limita a espera máxima a 60 segundos (rate limit é por minuto)
                retry_after = min(retry_after, 60)
                
                # Se ainda assim for muito grande, usa 60 segundos como padrão seguro
                if retry_after > 60 or retry_after < 0:
                    retry_after = 60
                
                print(f"[AVISO] Rate limit excedido. Aguardando {retry_after} segundos...")
                time.sleep(retry_after)
                # Tenta novamente uma vez
                return self._make_request(endpoint, params, method, use_thread_local)
            
            print(f"Erro na requisição {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                if hasattr(e.response, 'text'):
                    print(f"Resposta: {e.response.text}")
            raise
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
                # Rate limiting é feito automaticamente em _make_request
                
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
                # Rate limiting é feito automaticamente em _make_request
                
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
        includes: Optional[List[str]] = None,
        use_thread_local: bool = False
    ) -> Dict:
        """
        Coleta detalhes de um ticket específico com includes.
        
        Args:
            ticket_id: ID do ticket
            includes: Lista de includes (ex: ['conversations', 'requester'])
            use_thread_local: Se True, usa sessão thread-local (para uso em threads)
        
        Returns:
            Dicionário com detalhes do ticket
        """
        if includes is None:
            includes = ['conversations', 'requester']
        
        params = {'include': ','.join(includes)}
        
        try:
            ticket = self._make_request(f'tickets/{ticket_id}', params=params, use_thread_local=use_thread_local)
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
    
    def _enrich_single_ticket(
        self,
        ticket_dict: Dict,
        includes: List[str],
        lock: Lock
    ) -> Dict:
        """
        Enriquece um único ticket (função auxiliar para processamento paralelo).
        
        Args:
            ticket_dict: Dicionário com dados do ticket
            includes: Lista de includes
            lock: Lock para thread safety (não usado atualmente, mas mantido para compatibilidade)
        
        Returns:
            Dicionário com ticket enriquecido
        """
        ticket_id = ticket_dict.get('id')
        
        if not ticket_id:
            return ticket_dict
        
        try:
            # Obtém detalhes do ticket (usa thread-local para thread safety)
            details = self.get_ticket_details(ticket_id, includes=includes, use_thread_local=True)
            
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
            
            return ticket_dict
            
        except Exception as e:
            # Mantém o ticket original mesmo se houver erro
            return ticket_dict
    
    def enrich_tickets(
        self,
        tickets_df: pd.DataFrame,
        includes: Optional[List[str]] = None,
        progress: bool = True,
        max_workers: Optional[int] = None,
        save_path: Optional[str] = None,
        save_interval: int = 100,
        skip_enriched: bool = True
    ) -> pd.DataFrame:
        """
        Enriquece tickets com informações adicionais (conversations, requester).
        Usa processamento paralelo para acelerar o processo.
        Salva progressivamente para evitar perda de dados.
        
        Args:
            tickets_df: DataFrame com tickets básicos
            includes: Lista de includes (padrão: ['conversations', 'requester'])
            progress: Se True, mostra barra de progresso
            max_workers: Número máximo de threads paralelas (None = calcula automaticamente baseado no rate limit)
            save_path: Caminho para salvar progressivamente (opcional)
            save_interval: Intervalo de tickets processados antes de salvar (padrão: 100)
            skip_enriched: Se True, pula tickets já enriquecidos (verifica se têm conversations ou requester_email)
        
        Returns:
            DataFrame com tickets enriquecidos
        """
        if includes is None:
            includes = ['conversations', 'requester']
        
        if len(tickets_df) == 0:
            return tickets_df
        
        # Verifica quais tickets já estão enriquecidos
        tickets_to_enrich = tickets_df.copy()
        enriched_count = 0
        has_enrichment = None
        
        if skip_enriched:
            # Verifica se já tem as colunas de enriquecimento
            # Um ticket está enriquecido se tem conversations OU requester_email preenchido
            enrichment_columns = ['conversations', 'requester_email']
            
            # Verifica quais colunas existem no DataFrame
            existing_columns = [col for col in enrichment_columns if col in tickets_df.columns]
            
            if existing_columns:
                # Verifica se pelo menos uma das colunas de enriquecimento está preenchida
                # Para conversations, verifica se não é None e se é uma lista não vazia
                # Para requester_email, verifica se não é None/NaN
                has_enrichment = pd.Series([False] * len(tickets_df))
                
                if 'conversations' in tickets_df.columns:
                    # Verifica se conversations não é None/NaN e não é lista vazia
                    conv_check = tickets_df['conversations'].notna()
                    # Para valores que são listas, verifica se não estão vazias
                    conv_check = conv_check & tickets_df['conversations'].apply(
                        lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else x is not None
                    )
                    has_enrichment = has_enrichment | conv_check
                
                if 'requester_email' in tickets_df.columns:
                    # Verifica se requester_email não é None/NaN
                    email_check = tickets_df['requester_email'].notna()
                    has_enrichment = has_enrichment | email_check
            else:
                # Nenhuma coluna de enriquecimento existe, então nenhum ticket está enriquecido
                has_enrichment = pd.Series([False] * len(tickets_df))
            
            if has_enrichment.any():
                enriched_count = has_enrichment.sum()
                tickets_to_enrich = tickets_df[~has_enrichment].copy()
                print(f"\n{enriched_count} tickets ja estao enriquecidos. Pulando...")
                print(f"  Restam {len(tickets_to_enrich)} tickets para enriquecer")
        
        if len(tickets_to_enrich) == 0:
            print("[OK] Todos os tickets já estão enriquecidos!")
            return tickets_df
        
        # Calcula número de workers baseado no rate limit se não especificado
        # Com limite de 50/min, usa no máximo 5 workers (10 req/worker/min com margem)
        if max_workers is None:
            max_workers = self.rate_limiter.get_max_workers_for_type(
                RequestType.TICKET_GET, 
                default=5  # Reduzido para respeitar limite de 50/min
            )
            # Garante que não ultrapasse 5 workers mesmo se o cálculo retornar mais
            max_workers = min(max_workers, 5)
        
        print(f"\nEnriquecendo {len(tickets_to_enrich)} tickets com {', '.join(includes)}...")
        print(f"Usando {max_workers} threads paralelas (ajustado ao rate limit da API).")
        
        # Mostra status do rate limiter
        status = self.rate_limiter.get_status()
        if status['api_remaining'] is not None:
            print(f"Rate limit da API: {status['api_remaining']}/{status['api_total']} requisições restantes")
        
        if save_path:
            print(f"Salvando progresso a cada {save_interval} tickets em: {save_path}")
        
        # Converte DataFrame para lista de dicionários mantendo a ordem
        tickets_list = [ticket.to_dict() for _, ticket in tickets_to_enrich.iterrows()]
        
        # Cria um lock para thread safety e salvamento progressivo
        lock = Lock()
        
        enriched_tickets = []
        errors = []
        processed_count = 0
        
        try:
            # Processa em paralelo
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submete todas as tarefas
                future_to_ticket = {
                    executor.submit(self._enrich_single_ticket, ticket_dict, includes, lock): idx
                    for idx, ticket_dict in enumerate(tickets_list)
                }
                
                # Processa resultados conforme completam
                if progress:
                    with tqdm(total=len(tickets_list), desc="Enriquecendo tickets") as pbar:
                        for future in as_completed(future_to_ticket):
                            idx = future_to_ticket[future]
                            try:
                                enriched_ticket = future.result()
                                enriched_tickets.append((idx, enriched_ticket))
                                processed_count += 1
                                
                                # Salva progressivamente
                                if save_path and processed_count % save_interval == 0:
                                    self._save_enriched_progress(
                                        enriched_tickets, 
                                        tickets_df if skip_enriched else None,
                                        enriched_count if skip_enriched else 0,
                                        save_path,
                                        lock
                                    )
                                    
                            except Exception as e:
                                ticket_id = tickets_list[idx].get('id', 'unknown')
                                errors.append((ticket_id, str(e)))
                                # Mantém o ticket original em caso de erro
                                enriched_tickets.append((idx, tickets_list[idx]))
                                processed_count += 1
                                
                                # Salva mesmo em caso de erro para não perder progresso
                                if save_path and processed_count % save_interval == 0:
                                    self._save_enriched_progress(
                                        enriched_tickets,
                                        tickets_df if skip_enriched else None,
                                        enriched_count if skip_enriched else 0,
                                        save_path,
                                        lock
                                    )
                            finally:
                                pbar.update(1)
                else:
                    for future in as_completed(future_to_ticket):
                        idx = future_to_ticket[future]
                        try:
                            enriched_ticket = future.result()
                            enriched_tickets.append((idx, enriched_ticket))
                            processed_count += 1
                            
                            # Salva progressivamente
                            if save_path and processed_count % save_interval == 0:
                                self._save_enriched_progress(
                                    enriched_tickets,
                                    tickets_df if skip_enriched else None,
                                    enriched_count if skip_enriched else 0,
                                    save_path,
                                    lock
                                )
                                
                        except Exception as e:
                            ticket_id = tickets_list[idx].get('id', 'unknown')
                            errors.append((ticket_id, str(e)))
                            # Mantém o ticket original em caso de erro
                            enriched_tickets.append((idx, tickets_list[idx]))
                            processed_count += 1
                            
                            # Salva mesmo em caso de erro
                            if save_path and processed_count % save_interval == 0:
                                self._save_enriched_progress(
                                    enriched_tickets,
                                    tickets_df if skip_enriched else None,
                                    enriched_count if skip_enriched else 0,
                                    save_path,
                                    lock
                                )
        except KeyboardInterrupt:
            print("\n\n[AVISO] Interrupção detectada! Salvando progresso...")
            if save_path:
                self._save_enriched_progress(
                    enriched_tickets,
                    tickets_df if skip_enriched else None,
                    enriched_count if skip_enriched else 0,
                    save_path,
                    lock
                )
            raise
        except Exception as e:
            print(f"\n[AVISO] Erro durante o enriquecimento: {e}")
            print("Salvando progresso parcial...")
            if save_path:
                self._save_enriched_progress(
                    enriched_tickets,
                    tickets_df if skip_enriched else None,
                    enriched_count if skip_enriched else 0,
                    save_path,
                    lock
                )
            raise
        
        # Ordena pelos índices originais para manter a ordem
        enriched_tickets.sort(key=lambda x: x[0])
        enriched_tickets = [ticket for _, ticket in enriched_tickets]
        
        # Combina com tickets já enriquecidos se necessário
        if skip_enriched and enriched_count > 0 and has_enrichment is not None:
            enriched_df_new = pd.DataFrame(enriched_tickets)
            enriched_df_old = tickets_df[has_enrichment].copy()
            enriched_df = pd.concat([enriched_df_old, enriched_df_new], ignore_index=True)
        else:
            enriched_df = pd.DataFrame(enriched_tickets)
        
        # Salva resultado final
        if save_path:
            self._save_enriched_progress(
                enriched_tickets,
                tickets_df if skip_enriched else None,
                enriched_count if skip_enriched else 0,
                save_path,
                lock,
                is_final=True
            )
        
        # Mostra erros se houver
        if errors:
            print(f"\n[AVISO] {len(errors)} tickets tiveram erros durante o enriquecimento:")
            for ticket_id, error in errors[:5]:  # Mostra apenas os primeiros 5
                print(f"  Ticket {ticket_id}: {error}")
            if len(errors) > 5:
                print(f"  ... e mais {len(errors) - 5} erros")
        
        print(f"[OK] {len(enriched_df)} tickets enriquecidos (total)")
        
        return enriched_df
    
    def _save_enriched_progress(
        self,
        enriched_tickets: List[tuple],
        original_tickets_df: Optional[pd.DataFrame],
        enriched_count: int,
        save_path: str,
        lock: Lock,
        is_final: bool = False
    ):
        """
        Salva progresso do enriquecimento de forma thread-safe.
        
        Args:
            enriched_tickets: Lista de tuplas (idx, ticket_dict) com tickets enriquecidos
            original_tickets_df: DataFrame original com tickets já enriquecidos (se skip_enriched=True)
            enriched_count: Número de tickets já enriquecidos anteriormente
            save_path: Caminho para salvar
            lock: Lock para thread safety
            is_final: Se True, é o salvamento final
        """
        with lock:
            try:
                # Ordena pelos índices
                sorted_tickets = sorted(enriched_tickets, key=lambda x: x[0])
                enriched_list = [ticket for _, ticket in sorted_tickets]
                
                # Cria DataFrame com tickets enriquecidos
                enriched_df_new = pd.DataFrame(enriched_list)
                
                # Carrega arquivo existente se houver (para combinar com progresso anterior)
                existing_df = None
                if os.path.exists(save_path):
                    try:
                        existing_df = pd.read_json(save_path)
                    except:
                        pass
                
                # Combina com tickets já enriquecidos
                if existing_df is not None and len(existing_df) > 0:
                    # Remove duplicatas baseado no ID do ticket
                    if 'id' in enriched_df_new.columns and 'id' in existing_df.columns:
                        # Remove tickets que já existem no arquivo salvo
                        existing_ids = set(existing_df['id'].astype(str))
                        new_ids = set(enriched_df_new['id'].astype(str))
                        new_unique_ids = new_ids - existing_ids
                        
                        if new_unique_ids:
                            enriched_df_new_unique = enriched_df_new[enriched_df_new['id'].astype(str).isin(new_unique_ids)]
                            enriched_df = pd.concat([existing_df, enriched_df_new_unique], ignore_index=True)
                        else:
                            enriched_df = existing_df
                    else:
                        # Se não tem ID, combina tudo (pode ter duplicatas)
                        enriched_df = pd.concat([existing_df, enriched_df_new], ignore_index=True)
                elif original_tickets_df is not None and enriched_count > 0:
                    # Se não tem arquivo salvo, usa o DataFrame original
                    enrichment_columns = ['conversations', 'requester_email']
                    if all(col in original_tickets_df.columns for col in enrichment_columns):
                        has_enrichment = original_tickets_df[enrichment_columns].notna().any(axis=1)
                        enriched_df_old = original_tickets_df[has_enrichment].copy()
                        enriched_df = pd.concat([enriched_df_old, enriched_df_new], ignore_index=True)
                    else:
                        enriched_df = enriched_df_new
                else:
                    enriched_df = enriched_df_new
                
                # Salva em arquivo temporário primeiro, depois renomeia (atomicidade)
                temp_path = save_path + '.tmp'
                enriched_df.to_json(temp_path, orient='records', date_format='iso', indent=2)
                
                # Renomeia para o arquivo final (operacao atomica)
                if os.path.exists(save_path):
                    os.replace(temp_path, save_path)
                else:
                    os.rename(temp_path, save_path)
                
                if is_final:
                    print(f"\nProgresso salvo em {save_path} ({len(enriched_df)} tickets)")
                else:
                    print(f"\nProgresso salvo: {len(enriched_df)} tickets ({len(enriched_list)} novos processados)")
                    
            except Exception as e:
                print(f"\nErro ao salvar progresso: {e}")
                import traceback
                traceback.print_exc()
                # Não levanta exceção para não interromper o processamento
    
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

