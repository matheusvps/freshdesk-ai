"""Funções auxiliares para o projeto."""

import os
import logging
import yaml
from typing import Dict, Optional
from pathlib import Path


def setup_logging(level: str = 'INFO', log_file: Optional[str] = None):
    """
    Configura logging para o projeto.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Caminho do arquivo de log (opcional)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Formato do log
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Carrega configuração de arquivo YAML.
    
    Args:
        config_path: Caminho do arquivo de configuração
    
    Returns:
        Dicionário com configurações
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config.yaml'
        )
    
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str):
    """Garante que um diretório existe."""
    os.makedirs(path, exist_ok=True)

