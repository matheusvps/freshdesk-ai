"""Módulo ETL para processamento e armazenamento de dados."""

from .pipeline import ETLPipeline
from .database import DatabaseManager

__all__ = ['ETLPipeline', 'DatabaseManager']

