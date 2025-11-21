"""Sistema de rate limiting para API do Freshdesk."""

import time
from threading import Lock, RLock
from typing import Dict, Optional
from collections import deque
from enum import Enum


class RequestType(Enum):
    """Tipos de requisição para controle de rate limit."""
    TICKET_LIST = "tickets_list"
    TICKET_GET = "ticket_get"
    TICKET_CREATE = "ticket_create"
    TICKET_UPDATE = "ticket_update"
    CONTACTS_LIST = "contacts_list"
    CONTACT_GET = "contact_get"
    AGENTS_LIST = "agents_list"
    SATISFACTION_RATINGS = "satisfaction_ratings"
    OTHER = "other"


class RateLimiter:
    """
    Controla rate limiting baseado nos limites da API Freshdesk.
    
    Limites Pro:
    - Ticket Create: 160/min
    - Ticket Update: 160/min
    - Tickets List: 100/min
    - Contacts List: 100/min
    """
    
    # Limites por minuto por tipo de requisição
    LIMITS = {
        RequestType.TICKET_CREATE: 160,
        RequestType.TICKET_UPDATE: 160,
        RequestType.TICKET_LIST: 100,
        RequestType.CONTACTS_LIST: 100,
        RequestType.TICKET_GET: 50,  # Limite de 50/min para enriquecimento de tickets
        RequestType.CONTACT_GET: 100,  # Assumindo mesmo limite de list
        RequestType.AGENTS_LIST: 100,  # Assumindo mesmo limite de contacts
        RequestType.SATISFACTION_RATINGS: 100,  # Limite conservador, similar a outros endpoints de listagem
        RequestType.OTHER: 100,  # Limite conservador para outros endpoints
    }
    
    def __init__(self):
        """Inicializa o rate limiter."""
        # Histórico de requisições por tipo (timestamps)
        self._request_history: Dict[RequestType, deque] = {
            req_type: deque() for req_type in RequestType
        }
        
        # Lock para thread safety
        self._lock = RLock()
        
        # Estado atual dos rate limits da API (atualizado pelos headers)
        self._api_remaining: Optional[int] = None
        self._api_total: Optional[int] = None
        self._last_header_update = 0
        
        # Janela de tempo em segundos (60 segundos = 1 minuto)
        self._window_seconds = 60
    
    def _clean_old_requests(self, req_type: RequestType, now: float):
        """Remove requisições antigas fora da janela de tempo."""
        history = self._request_history[req_type]
        cutoff = now - self._window_seconds
        
        while history and history[0] < cutoff:
            history.popleft()
    
    def _can_make_request(self, req_type: RequestType, now: float) -> bool:
        """
        Verifica se pode fazer uma requisição do tipo especificado.
        
        Returns:
            True se pode fazer a requisição, False caso contrário
        """
        self._clean_old_requests(req_type, now)
        
        limit = self.LIMITS[req_type]
        history = self._request_history[req_type]
        
        # Usa apenas 80% do limite para ter margem de segurança
        safe_limit = int(limit * 0.8)
        
        # Verifica limite local (com margem de segurança)
        if len(history) >= safe_limit:
            return False
        
        # Se temos informação dos headers da API, verifica também
        if self._api_remaining is not None:
            # Reserva um buffer de segurança maior (20% do limite ou mínimo de 20 requisições)
            if self._api_total:
                buffer = max(20, int(self._api_total * 0.2))
            else:
                buffer = 20
            
            if self._api_remaining <= buffer:
                return False
        
        return True
    
    def _get_wait_time(self, req_type: RequestType, now: float) -> float:
        """
        Calcula quanto tempo esperar antes da próxima requisição.
        
        Returns:
            Tempo em segundos para esperar (0 se não precisa esperar)
        """
        self._clean_old_requests(req_type, now)
        
        limit = self.LIMITS[req_type]
        history = self._request_history[req_type]
        
        if len(history) < limit:
            return 0.0
        
        # Se atingiu o limite, espera até a requisição mais antiga sair da janela
        oldest_request = history[0]
        wait_until = oldest_request + self._window_seconds
        wait_time = max(0.0, wait_until - now)
        
        return wait_time
    
    def wait_if_needed(self, req_type: RequestType):
        """
        Espera se necessário antes de fazer uma requisição.
        NOTA: Não registra a requisição aqui - isso deve ser feito em record_request após sucesso.
        
        Args:
            req_type: Tipo da requisição
        """
        with self._lock:
            now = time.time()
            
            # Verifica se precisa esperar
            if not self._can_make_request(req_type, now):
                wait_time = self._get_wait_time(req_type, now)
                if wait_time > 0:
                    time.sleep(wait_time)
    
    def record_request(self, req_type: RequestType, response_headers: Optional[Dict] = None):
        """
        Registra uma requisição feita e atualiza estado baseado nos headers.
        
        Args:
            req_type: Tipo da requisição
            response_headers: Headers da resposta HTTP (opcional)
        """
        with self._lock:
            now = time.time()
            
            # Atualiza estado baseado nos headers da API
            if response_headers:
                self._update_from_headers(response_headers, now)
            
            # Limpa histórico antigo
            self._clean_old_requests(req_type, now)
            
            # Registra a requisição
            self._request_history[req_type].append(now)
    
    def _update_from_headers(self, headers: Dict, now: float):
        """
        Atualiza estado interno baseado nos headers de rate limit da API.
        
        Args:
            headers: Headers da resposta HTTP (case-insensitive)
            now: Timestamp atual
        """
        # Headers esperados:
        # X-Ratelimit-Remaining: requisições restantes
        # X-Ratelimit-Total: total de requisições permitidas
        # X-Ratelimit-Used-CurrentRequest: requisições usadas nesta requisição
        
        # Headers HTTP são case-insensitive, então normalizamos para lowercase
        headers_lower = {k.lower(): v for k, v in headers.items()}
        
        remaining = headers_lower.get('x-ratelimit-remaining')
        total = headers_lower.get('x-ratelimit-total')
        
        if remaining is not None:
            try:
                self._api_remaining = int(remaining)
                self._last_header_update = now
            except (ValueError, TypeError):
                pass
        
        if total is not None:
            try:
                self._api_total = int(total)
            except (ValueError, TypeError):
                pass
    
    def get_status(self) -> Dict:
        """
        Retorna status atual do rate limiter.
        
        Returns:
            Dicionário com informações de status
        """
        with self._lock:
            now = time.time()
            status = {
                'api_remaining': self._api_remaining,
                'api_total': self._api_total,
                'requests_by_type': {}
            }
            
            for req_type in RequestType:
                self._clean_old_requests(req_type, now)
                history = self._request_history[req_type]
                limit = self.LIMITS[req_type]
                status['requests_by_type'][req_type.value] = {
                    'count': len(history),
                    'limit': limit,
                    'remaining': max(0, limit - len(history))
                }
            
            return status
    
    def get_max_workers_for_type(self, req_type: RequestType, default: int = 10) -> int:
        """
        Calcula número máximo de workers recomendado baseado no rate limit.
        
        Args:
            req_type: Tipo de requisição
            default: Valor padrão se não conseguir calcular
        
        Returns:
            Número máximo de workers recomendado
        """
        with self._lock:
            limit = self.LIMITS[req_type]
            
            # Se temos informação da API, usa ela
            if self._api_remaining is not None and self._api_total is not None:
                # Usa apenas 60% do limite restante para ter margem de segurança maior
                # E garante que sempre tenha pelo menos 20 requisições de buffer
                buffer = max(20, int(self._api_total * 0.2))
                safe_remaining = max(0, self._api_remaining - buffer)
                safe_limit = int(safe_remaining * 0.6)
                return max(1, min(safe_limit, default))
            
            # Caso contrário, usa 60% do limite conhecido (mais conservador)
            safe_limit = int(limit * 0.6)
            return max(1, min(safe_limit, default))


def get_request_type(endpoint: str, method: str = 'GET') -> RequestType:
    """
    Determina o tipo de requisição baseado no endpoint e método HTTP.
    
    Args:
        endpoint: Endpoint da API (ex: 'tickets', 'tickets/123', 'contacts')
        method: Método HTTP (GET, POST, PUT, etc.)
    
    Returns:
        Tipo de requisição
    """
    endpoint_lower = endpoint.lower()
    
    # Tickets
    if endpoint_lower.startswith('tickets'):
        if method.upper() == 'POST':
            return RequestType.TICKET_CREATE
        elif method.upper() in ('PUT', 'PATCH'):
            return RequestType.TICKET_UPDATE
        elif '/' in endpoint_lower and endpoint_lower != 'tickets':
            # tickets/123 ou tickets/123/conversations
            return RequestType.TICKET_GET
        else:
            return RequestType.TICKET_LIST
    
    # Contacts
    elif endpoint_lower.startswith('contacts'):
        if '/' in endpoint_lower and endpoint_lower != 'contacts':
            return RequestType.CONTACT_GET
        else:
            return RequestType.CONTACTS_LIST
    
    # Agents
    elif endpoint_lower.startswith('agents'):
        return RequestType.AGENTS_LIST
    
    # Satisfaction Ratings
    elif endpoint_lower.startswith('surveys/satisfaction_ratings') or endpoint_lower.startswith('surveys'):
        return RequestType.SATISFACTION_RATINGS
    
    # Outros
    else:
        return RequestType.OTHER

