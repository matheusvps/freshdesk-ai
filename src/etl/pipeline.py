"""Pipeline ETL para processamento de dados."""

import pandas as pd
import re
import json
from typing import Dict, Optional
from datetime import datetime
import unicodedata

from .database import DatabaseManager
from ..utils.status_mapper import add_status_name_column


class ETLPipeline:
    """Pipeline ETL para processar dados do Freshdesk."""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Inicializa o pipeline ETL.
        
        Args:
            db_manager: Instância do DatabaseManager
        """
        self.db = db_manager
    
    def remove_pii(self, text: str) -> str:
        """
        Remove informações pessoais identificáveis (PII).
        
        Args:
            text: Texto a ser processado
        
        Returns:
            Texto sem PII
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove telefones (formato brasileiro e internacional)
        text = re.sub(r'(\+55\s?)?(\(?\d{2}\)?\s?)?(\d{4,5}[-.\s]?\d{4})', '[PHONE]', text)
        
        # Remove CPF/CNPJ
        text = re.sub(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}', '[CPF]', text)
        text = re.sub(r'\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}', '[CNPJ]', text)
        
        # Remove cartões de crédito
        text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza texto.
        
        Args:
            text: Texto a ser limpo
        
        Returns:
            Texto limpo
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove PII primeiro
        text = self.remove_pii(text)
        
        # Normaliza unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Remove caracteres especiais excessivos
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}\-]', ' ', text)
        
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Remove espaços no início e fim
        text = text.strip()
        
        return text
    
    def normalize_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Normaliza string de data para datetime."""
        if pd.isna(dt_str) or not dt_str:
            return None
        
        try:
            # Formato ISO do Freshdesk
            if isinstance(dt_str, str):
                dt_str = dt_str.replace('Z', '+00:00')
            return pd.to_datetime(dt_str)
        except:
            return None
    
    def process_tickets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa DataFrame de tickets.
        
        Args:
            df: DataFrame com tickets brutos
        
        Returns:
            DataFrame processado
        """
        df = df.copy()
        
        # Limpa textos
        if 'description' in df.columns:
            df['cleaned_text'] = df['description'].apply(self.clean_text)
        elif 'subject' in df.columns:
            df['cleaned_text'] = df['subject'].apply(self.clean_text)
        else:
            df['cleaned_text'] = ""
        
        # Normaliza datas
        date_columns = ['created_at', 'updated_at', 'due_by', 'fr_due_by']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.normalize_datetime)
        
        # Processa custom_fields (JSON)
        if 'custom_fields' in df.columns:
            df['custom_fields'] = df['custom_fields'].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
        
        # Processa tags
        if 'tags' in df.columns:
            df['tags'] = df['tags'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
        
        # Garante tipos corretos
        if 'id' in df.columns:
            df['id'] = df['id'].astype(int)
        
        # Adiciona coluna com nome do status
        if 'status' in df.columns:
            df = add_status_name_column(df, 'status', 'status_name')
        
        # Seleciona apenas colunas relevantes para o banco
        ticket_columns = [
            'id', 'subject', 'description', 'status', 'status_name', 'priority', 'type',
            'created_at', 'updated_at', 'due_by', 'fr_due_by',
            'requester_id', 'responder_id', 'group_id', 'tags',
            'satisfaction_rating', 'custom_fields', 'cleaned_text'
        ]
        
        available_columns = [col for col in ticket_columns if col in df.columns]
        df_processed = df[available_columns].copy()
        
        return df_processed
    
    def process_contacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa DataFrame de contatos."""
        df = df.copy()
        
        # Normaliza datas
        date_columns = ['created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.normalize_datetime)
        
        # Processa tags
        if 'tags' in df.columns:
            df['tags'] = df['tags'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
        
        # Processa custom_fields
        if 'custom_fields' in df.columns:
            df['custom_fields'] = df['custom_fields'].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
            )
        
        return df
    
    def process_agents(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processa DataFrame de agentes."""
        df = df.copy()
        
        # Normaliza datas
        date_columns = ['created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = df[col].apply(self.normalize_datetime)
        
        return df
    
    def merge_satisfaction_ratings(
        self,
        tickets_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Mescla satisfaction ratings com tickets.
        
        Args:
            tickets_df: DataFrame de tickets
            ratings_df: DataFrame de satisfaction ratings
        
        Returns:
            DataFrame de tickets com satisfaction_rating preenchido
        """
        if ratings_df is None or len(ratings_df) == 0:
            return tickets_df
        
        # Garante que ticket_id existe e é do tipo correto
        if 'ticket_id' not in ratings_df.columns:
            print("[AVISO] Coluna ticket_id não encontrada em satisfaction ratings")
            return tickets_df
        
        # Converte ratings para formato compatível com CSAT/NPS
        def convert_rating_for_csat(rating_value):
            """Converte rating do Freshdesk para escala 1-5 para CSAT."""
            if pd.isna(rating_value):
                return None
            
            rating = int(rating_value)
            
            # New Survey Ratings -> escala 1-5
            if rating == 103:  # Extremely Happy
                return 5
            elif rating == 102:  # Very Happy
                return 5
            elif rating == 101:  # Happy
                return 4
            elif rating == 100:  # Neutral
                return 3
            elif rating == -101:  # Unhappy
                return 2
            elif rating == -102:  # Very Unhappy
                return 1
            elif rating == -103:  # Extremely Unhappy
                return 1
            # Classic Survey Ratings
            elif rating == 1:  # Happy
                return 4
            elif rating == 2:  # Neutral
                return 3
            elif rating == 3:  # Unhappy
                return 2
            else:
                return None
        
        # Prepara DataFrame de ratings para merge
        ratings_merge = ratings_df[['ticket_id', 'rating_value', 'rating_label', 'feedback', 'created_at']].copy()
        ratings_merge['satisfaction_rating'] = ratings_merge['rating_value'].apply(convert_rating_for_csat)
        
        # Remove duplicatas mantendo o mais recente (baseado em created_at)
        if 'created_at' in ratings_merge.columns:
            ratings_merge['created_at'] = pd.to_datetime(ratings_merge['created_at'], errors='coerce')
            ratings_merge = ratings_merge.sort_values('created_at', ascending=False)
            ratings_merge = ratings_merge.drop_duplicates(subset=['ticket_id'], keep='first')
        
        # Faz merge com tickets
        tickets_merged = tickets_df.merge(
            ratings_merge[['ticket_id', 'satisfaction_rating', 'rating_label', 'feedback']],
            left_on='id',
            right_on='ticket_id',
            how='left'
        )
        
        # Remove coluna ticket_id temporária se foi criada
        if 'ticket_id' in tickets_merged.columns:
            tickets_merged = tickets_merged.drop(columns=['ticket_id'])
        
        # Mantém satisfaction_rating como numérico (já está convertido para escala 1-5)
        # Não converte para string, mantém como float/int para facilitar cálculos
        if 'satisfaction_rating' in tickets_merged.columns:
            # Converte None/NaN para None explicitamente
            tickets_merged['satisfaction_rating'] = tickets_merged['satisfaction_rating'].where(
                tickets_merged['satisfaction_rating'].notna(), None
            )
        
        matched_count = tickets_merged['satisfaction_rating'].notna().sum()
        if matched_count > 0:
            print(f"[OK] {matched_count} tickets com satisfaction rating associado")
        
        return tickets_merged
    
    def run(
        self,
        tickets_df: Optional[pd.DataFrame] = None,
        contacts_df: Optional[pd.DataFrame] = None,
        agents_df: Optional[pd.DataFrame] = None,
        satisfaction_ratings_df: Optional[pd.DataFrame] = None
    ):
        """
        Executa o pipeline ETL completo.
        
        Args:
            tickets_df: DataFrame de tickets
            contacts_df: DataFrame de contatos
            agents_df: DataFrame de agentes
            satisfaction_ratings_df: DataFrame de satisfaction ratings
        """
        if tickets_df is not None:
            print("Processando tickets...")
            tickets_processed = self.process_tickets(tickets_df)
            
            # Mescla satisfaction ratings se disponível
            if satisfaction_ratings_df is not None and len(satisfaction_ratings_df) > 0:
                print("Mesclando satisfaction ratings com tickets...")
                tickets_processed = self.merge_satisfaction_ratings(tickets_processed, satisfaction_ratings_df)
            
            self.db.insert_tickets(tickets_processed)
            print(f"[OK] {len(tickets_processed)} tickets processados e inseridos")
        
        if contacts_df is not None:
            print("Processando contatos...")
            contacts_processed = self.process_contacts(contacts_df)
            self.db.insert_contacts(contacts_processed)
            print(f"[OK] {len(contacts_processed)} contatos processados e inseridos")
        
        if agents_df is not None:
            print("Processando agentes...")
            agents_processed = self.process_agents(agents_df)
            self.db.insert_agents(agents_processed)
            print(f"[OK] {len(agents_processed)} agentes processados e inseridos")
        
        print("Pipeline ETL concluído!")

