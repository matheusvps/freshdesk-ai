"""Explainable AI usando SHAP e LIME."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP não disponível. Instale com: pip install shap")

try:
    from lime import lime_tabular
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME não disponível. Instale com: pip install lime")


class XAIExplainer:
    """Classe para explicar previsões usando SHAP e LIME."""
    
    def __init__(self):
        """Inicializa o explicador."""
        if not SHAP_AVAILABLE and not LIME_AVAILABLE:
            raise ImportError("SHAP ou LIME devem estar instalados.")
    
    def explain_shap(
        self,
        model,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Dict:
        """
        Explica previsões usando SHAP.
        
        Args:
            model: Modelo treinado
            X: DataFrame com features
            feature_names: Nomes das features
            max_samples: Número máximo de amostras para explicar
        
        Returns:
            Dicionário com valores SHAP e explicações
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP não está instalado.")
        
        # Limita amostras para performance
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=42)
        else:
            X_sample = X
        
        # Cria explicador baseado no tipo de modelo
        try:
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model)
            
            shap_values = explainer.shap_values(X_sample.values)
            
            # Se for array multidimensional, pega primeira classe
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Classe positiva
            
            # Calcula importância média
            feature_importance = pd.DataFrame({
                'feature': feature_names or X.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            return {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'explainer': explainer
            }
        except Exception as e:
            print(f"Erro ao calcular SHAP: {e}")
            return {}
    
    def explain_lime_tabular(
        self,
        model,
        X_train: pd.DataFrame,
        X_explain: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explica previsões tabulares usando LIME.
        
        Args:
            model: Modelo treinado
            X_train: Dados de treino
            X_explain: Dados a explicar
            feature_names: Nomes das features
            class_names: Nomes das classes
        
        Returns:
            Dicionário com explicações LIME
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME não está instalado.")
        
        explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_names or list(X_train.columns),
            class_names=class_names or ['Class 0', 'Class 1'],
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )
        
        explanations = []
        for idx, row in X_explain.iterrows():
            exp = explainer.explain_instance(
                row.values,
                model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                num_features=min(10, len(X_explain.columns))
            )
            explanations.append(exp)
        
        return {
            'explanations': explanations,
            'explainer': explainer
        }
    
    def explain_lime_text(
        self,
        model,
        texts: List[str],
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Explica previsões de texto usando LIME.
        
        Args:
            model: Modelo treinado (deve ter predict_proba)
            texts: Lista de textos a explicar
            class_names: Nomes das classes
        
        Returns:
            Dicionário com explicações LIME
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME não está instalado.")
        
        explainer = LimeTextExplainer(class_names=class_names or ['negative', 'positive'])
        
        explanations = []
        for text in texts:
            exp = explainer.explain_instance(
                text,
                model.predict_proba,
                num_features=10
            )
            explanations.append(exp)
        
        return {
            'explanations': explanations,
            'explainer': explainer
        }
    
    def get_feature_importance_summary(
        self,
        shap_results: Dict,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Retorna resumo de importância de features.
        
        Args:
            shap_results: Resultados do SHAP
            top_n: Número de features top
        
        Returns:
            DataFrame com importância
        """
        if 'feature_importance' in shap_results:
            return shap_results['feature_importance'].head(top_n)
        return pd.DataFrame()

