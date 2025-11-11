"""Calculadora de métricas e KPIs."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from ..etl.database import DatabaseManager


class MetricsCalculator:
    """Calculadora de métricas de atendimento."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Inicializa a calculadora de métricas.
        
        Args:
            db_manager: Instância do DatabaseManager
        """
        self.db = db_manager
    
    def calculate_csat(self, tickets_df: pd.DataFrame) -> float:
        """
        Calcula CSAT (Customer Satisfaction Score).
        
        CSAT = média das notas de satisfação (escala 1-5)
        
        Args:
            tickets_df: DataFrame com tickets
        
        Returns:
            CSAT score (0-5)
        """
        if 'satisfaction_rating' not in tickets_df.columns:
            return None
        
        # Mapeia ratings para números
        def rating_to_number(rating):
            if pd.isna(rating):
                return None
            
            rating_str = str(rating).lower()
            if 'good' in rating_str or 'satisfied' in rating_str:
                return 5
            elif 'bad' in rating_str or 'dissatisfied' in rating_str:
                return 1
            elif rating_str.isdigit():
                return int(rating_str)
            else:
                return 3  # Neutral
        
        ratings = tickets_df['satisfaction_rating'].apply(rating_to_number).dropna()
        
        if len(ratings) == 0:
            return None
        
        return float(ratings.mean())
    
    def calculate_nps(self, tickets_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula NPS (Net Promoter Score).
        
        NPS = % Promotores - % Detratores
        
        Args:
            tickets_df: DataFrame com tickets
        
        Returns:
            Dicionário com NPS e componentes
        """
        if 'satisfaction_rating' not in tickets_df.columns:
            return None
        
        # Mapeia ratings para categorias NPS
        def rating_to_nps_category(rating):
            if pd.isna(rating):
                return None
            
            rating_str = str(rating).lower()
            if 'good' in rating_str or 'satisfied' in rating_str:
                return 'promoter'
            elif 'bad' in rating_str or 'dissatisfied' in rating_str:
                return 'detractor'
            else:
                return 'passive'
        
        categories = tickets_df['satisfaction_rating'].apply(rating_to_nps_category).dropna()
        
        if len(categories) == 0:
            return None
        
        total = len(categories)
        promoters = (categories == 'promoter').sum()
        detractors = (categories == 'detractor').sum()
        passives = (categories == 'passive').sum()
        
        promoter_pct = (promoters / total) * 100
        detractor_pct = (detractors / total) * 100
        passive_pct = (passives / total) * 100
        
        nps = promoter_pct - detractor_pct
        
        return {
            'nps': nps,
            'promoter_pct': promoter_pct,
            'detractor_pct': detractor_pct,
            'passive_pct': passive_pct,
            'total_responses': total
        }
    
    def calculate_ces(self, tickets_df: pd.DataFrame) -> float:
        """
        Calcula CES (Customer Effort Score).
        
        CES = função do número de reaberturas e interações
        
        Args:
            tickets_df: DataFrame com tickets
        
        Returns:
            CES score (quanto maior, mais esforço)
        """
        if 'requester_id' not in tickets_df.columns:
            return None
        
        # Conta reaberturas por cliente
        if 'status' in tickets_df.columns:
            reopened = tickets_df[tickets_df['status'] == 5].groupby('requester_id').size()
        else:
            reopened = pd.Series()
        
        # Conta número de tickets por cliente
        ticket_counts = tickets_df.groupby('requester_id').size()
        
        # CES = média de (reaberturas + número de tickets) / número de tickets
        # Quanto maior, mais esforço
        if len(reopened) > 0:
            effort = (reopened + ticket_counts) / ticket_counts
            ces = effort.mean()
        else:
            # Se não há reaberturas, CES baseado apenas no número de tickets
            ces = ticket_counts.mean() / 10  # Normaliza
        
        return float(ces)
    
    def calculate_sla(
        self,
        tickets_df: pd.DataFrame,
        sla_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Calcula métricas de SLA (Service Level Agreement).
        
        Args:
            tickets_df: DataFrame com tickets
            sla_hours: Horas do SLA
        
        Returns:
            Dicionário com métricas de SLA
        """
        if 'created_at' not in tickets_df.columns or 'updated_at' not in tickets_df.columns:
            return None
        
        # Calcula tempo de resposta
        tickets_df = tickets_df.copy()
        tickets_df['created_at'] = pd.to_datetime(tickets_df['created_at'])
        tickets_df['updated_at'] = pd.to_datetime(tickets_df['updated_at'])
        
        tickets_df['response_time_hours'] = (
            tickets_df['updated_at'] - tickets_df['created_at']
        ).dt.total_seconds() / 3600
        
        # Filtra apenas tickets resolvidos
        if 'status' in tickets_df.columns:
            resolved = tickets_df[tickets_df['status'] == 4]  # Resolved
        else:
            resolved = tickets_df
        
        if len(resolved) == 0:
            return None
        
        # Métricas
        avg_response_time = resolved['response_time_hours'].mean()
        median_response_time = resolved['response_time_hours'].median()
        
        # Taxa de conformidade com SLA
        within_sla = (resolved['response_time_hours'] <= sla_hours).sum()
        sla_compliance = (within_sla / len(resolved)) * 100
        
        return {
            'avg_response_time_hours': float(avg_response_time),
            'median_response_time_hours': float(median_response_time),
            'sla_compliance_pct': float(sla_compliance),
            'sla_hours': sla_hours,
            'tickets_within_sla': int(within_sla),
            'total_resolved_tickets': len(resolved)
        }
    
    def calculate_complaint_rate(
        self,
        tickets_df: pd.DataFrame,
        product_sales: Optional[Dict[str, int]] = None
    ) -> Dict[str, float]:
        """
        Calcula taxa de reclamação por produto.
        
        Args:
            tickets_df: DataFrame com tickets
            product_sales: Dicionário com vendas por produto (opcional)
        
        Returns:
            Dicionário com taxas de reclamação
        """
        # Tenta identificar produtos via tags ou custom_fields
        if 'tags' in tickets_df.columns:
            # Assumindo que tags podem conter nomes de produtos
            all_tags = tickets_df['tags'].dropna().str.split(',').explode().str.strip()
            product_counts = all_tags.value_counts()
        else:
            product_counts = pd.Series()
        
        if product_sales:
            # Calcula taxa = tickets / vendas
            complaint_rates = {}
            for product, sales in product_sales.items():
                tickets_count = product_counts.get(product, 0)
                rate = (tickets_count / sales) * 100 if sales > 0 else 0
                complaint_rates[product] = rate
            return complaint_rates
        else:
            # Retorna apenas contagem de tickets por produto
            return product_counts.to_dict()
    
    def calculate_all_metrics(
        self,
        tickets_df: Optional[pd.DataFrame] = None,
        date: Optional[datetime] = None
    ) -> Dict:
        """
        Calcula todas as métricas e salva no banco.
        
        Args:
            tickets_df: DataFrame com tickets (se None, busca do banco)
            date: Data das métricas (padrão: hoje)
        
        Returns:
            Dicionário com todas as métricas
        """
        if tickets_df is None:
            tickets_df = self.db.get_tickets()
        
        if date is None:
            date = datetime.now()
        
        metrics = {}
        
        # CSAT
        csat = self.calculate_csat(tickets_df)
        if csat is not None:
            metrics['csat'] = csat
        
        # NPS
        nps = self.calculate_nps(tickets_df)
        if nps:
            metrics['nps'] = nps['nps']
            metrics['nps_promoter_pct'] = nps['promoter_pct']
            metrics['nps_detractor_pct'] = nps['detractor_pct']
        
        # CES
        ces = self.calculate_ces(tickets_df)
        if ces is not None:
            metrics['ces'] = ces
        
        # SLA
        sla = self.calculate_sla(tickets_df)
        if sla:
            metrics['sla_compliance'] = sla['sla_compliance_pct']
            metrics['avg_response_time'] = sla['avg_response_time_hours']
        
        # Salva no banco
        metrics_df = pd.DataFrame([{
            'date': date,
            'metric_name': name,
            'metric_value': value,
            'category': 'kpi'
        } for name, value in metrics.items()])
        
        if len(metrics_df) > 0:
            self.db.insert_metrics(metrics_df)
        
        return metrics

