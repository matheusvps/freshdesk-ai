"""Módulo de análise de sentimentos e processamento de linguagem natural."""

# Import lazy para evitar problemas com dependências opcionais
def _lazy_import():
    try:
        from .sentiment import SentimentAnalyzer
        return SentimentAnalyzer
    except Exception as e:
        # Se houver erro no import, ainda permite usar outras partes do módulo
        import warnings
        warnings.warn(f"Erro ao importar SentimentAnalyzer: {e}")
        return None

# Tenta importar, mas não falha se houver problema
try:
    from .sentiment import SentimentAnalyzer
    __all__ = ['SentimentAnalyzer']
except Exception:
    # Se falhar, cria um placeholder
    __all__ = []

