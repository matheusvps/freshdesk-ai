"""Utilitário para mapear status de tickets do Freshdesk."""

# Mapeamento de status do Freshdesk
STATUS_MAP = {
    2: "Open",
    3: "Pending",
    4: "Resolved",
    5: "Closed",
    6: "Waiting on Customer",
    7: "Waiting on Third Party"
}

# Mapeamento reverso (nome -> valor)
STATUS_REVERSE_MAP = {v: k for k, v in STATUS_MAP.items()}


def get_status_name(status_value: int) -> str:
    """
    Retorna o nome do status baseado no valor inteiro.
    
    Args:
        status_value: Valor inteiro do status
        
    Returns:
        Nome do status ou "Unknown" se não encontrado
    """
    return STATUS_MAP.get(status_value, f"Unknown ({status_value})")


def get_status_value(status_name: str) -> int:
    """
    Retorna o valor inteiro do status baseado no nome.
    
    Args:
        status_name: Nome do status
        
    Returns:
        Valor inteiro do status ou None se não encontrado
    """
    return STATUS_REVERSE_MAP.get(status_name)


def add_status_name_column(df, status_column: str = 'status', status_name_column: str = 'status_name'):
    """
    Adiciona coluna com nomes de status ao DataFrame.
    
    Args:
        df: DataFrame com tickets
        status_column: Nome da coluna com valores de status
        status_name_column: Nome da nova coluna a ser criada
        
    Returns:
        DataFrame com coluna status_name adicionada
    """
    df = df.copy()
    if status_column in df.columns:
        df[status_name_column] = df[status_column].apply(get_status_name)
    return df

