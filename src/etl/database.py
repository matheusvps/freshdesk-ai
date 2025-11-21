"""Gerenciador de banco de dados para armazenamento local."""

import os
import sqlite3
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import Optional
import pandas as pd

Base = declarative_base()


class Tickets(Base):
    """Tabela de tickets."""
    __tablename__ = 'tickets'
    
    id = Column(Integer, primary_key=True)
    subject = Column(String(500))
    description = Column(Text)
    status = Column(Integer)
    status_name = Column(String(50))
    priority = Column(Integer)
    type = Column(String(50))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    due_by = Column(DateTime)
    fr_due_by = Column(DateTime)
    requester_id = Column(Integer)
    responder_id = Column(Integer)
    group_id = Column(Integer)
    tags = Column(String(500))
    satisfaction_rating = Column(String(50))
    custom_fields = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    cleaned_text = Column(Text)


class Contacts(Base):
    """Tabela de contatos."""
    __tablename__ = 'contacts'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    email = Column(String(200))
    phone = Column(String(50))
    company_id = Column(Integer)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    tags = Column(String(500))
    custom_fields = Column(Text)


class Agents(Base):
    """Tabela de agentes."""
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(200))
    name = Column(String(200))
    available = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class Metrics(Base):
    """Tabela de métricas agregadas."""
    __tablename__ = 'metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    category = Column(String(50))


class DatabaseManager:
    """Gerenciador de banco de dados SQLite."""
    
    def __init__(self, db_path: str = "data/freshdesk.db"):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            db_path: Caminho do arquivo SQLite
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.Session = sessionmaker(bind=self.engine)
        
        # Cria todas as tabelas
        Base.metadata.create_all(self.engine)
    
    def insert_tickets(self, df: pd.DataFrame, if_exists: str = 'replace'):
        """Insere tickets no banco de dados."""
        df.to_sql('tickets', self.engine, if_exists=if_exists, index=False)
    
    def insert_contacts(self, df: pd.DataFrame, if_exists: str = 'replace'):
        """Insere contatos no banco de dados."""
        df.to_sql('contacts', self.engine, if_exists=if_exists, index=False)
    
    def insert_agents(self, df: pd.DataFrame, if_exists: str = 'replace'):
        """Insere agentes no banco de dados."""
        df.to_sql('agents', self.engine, if_exists=if_exists, index=False)
    
    def insert_metrics(self, df: pd.DataFrame, if_exists: str = 'append'):
        """Insere métricas no banco de dados."""
        df.to_sql('metrics', self.engine, if_exists=if_exists, index=False)
    
    def query(self, sql: str) -> pd.DataFrame:
        """Executa query SQL e retorna DataFrame."""
        return pd.read_sql(sql, self.engine)
    
    def get_tickets(self) -> pd.DataFrame:
        """Retorna todos os tickets."""
        return self.query("SELECT * FROM tickets")
    
    def get_contacts(self) -> pd.DataFrame:
        """Retorna todos os contatos."""
        return self.query("SELECT * FROM contacts")
    
    def get_agents(self) -> pd.DataFrame:
        """Retorna todos os agentes."""
        return self.query("SELECT * FROM agents")
    
    def get_metrics(self) -> pd.DataFrame:
        """Retorna todas as métricas."""
        return self.query("SELECT * FROM metrics")

