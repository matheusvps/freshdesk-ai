"""Modelo de predição de churn."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost não disponível. Usando RandomForest.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM não disponível. Usando RandomForest.")


class ChurnPredictor:
    """Preditor de risco de churn de clientes."""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'random_forest'):
        """
        Inicializa o preditor de churn.
        
        Args:
            model_path: Caminho para modelo pré-treinado
            model_type: Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')
        """
        self.model_path = model_path or "models/churn_model.pkl"
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.pipeline = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _create_model(self):
        """Cria o modelo baseado no tipo escolhido."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(random_state=42, verbose=-1)
        else:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features para predição de churn.
        
        Args:
            df: DataFrame com dados de tickets
        
        Returns:
            DataFrame com features
        """
        features = pd.DataFrame()
        
        # Features básicas
        if 'requester_id' in df.columns:
            # Número de tickets por cliente
            ticket_counts = df.groupby('requester_id').size().reset_index(name='ticket_count')
            features = features.merge(ticket_counts, left_index=True, right_on='requester_id', how='left')
        
        # Tempo médio de resposta
        if 'created_at' in df.columns and 'updated_at' in df.columns:
            df['response_time'] = (pd.to_datetime(df['updated_at']) - 
                                  pd.to_datetime(df['created_at'])).dt.total_seconds() / 3600
            if 'requester_id' in df.columns:
                avg_response = df.groupby('requester_id')['response_time'].mean().reset_index(name='avg_response_time')
                features = features.merge(avg_response, on='requester_id', how='left')
        
        # Taxa de tickets negativos
        if 'sentiment_label' in df.columns and 'requester_id' in df.columns:
            negative_tickets = df[df['sentiment_label'] == 'negative'].groupby('requester_id').size().reset_index(name='negative_count')
            total_tickets = df.groupby('requester_id').size().reset_index(name='total_count')
            sentiment_merge = negative_tickets.merge(total_tickets, on='requester_id', how='outer').fillna(0)
            sentiment_merge['negative_rate'] = sentiment_merge['negative_count'] / sentiment_merge['total_count']
            features = features.merge(sentiment_merge[['requester_id', 'negative_rate']], on='requester_id', how='left')
        
        # Reaberturas
        if 'status' in df.columns and 'requester_id' in df.columns:
            reopened = df[df['status'] == 5].groupby('requester_id').size().reset_index(name='reopened_count')
            features = features.merge(reopened, on='requester_id', how='left').fillna(0)
        
        # Prioridade média
        if 'priority' in df.columns and 'requester_id' in df.columns:
            avg_priority = df.groupby('requester_id')['priority'].mean().reset_index(name='avg_priority')
            features = features.merge(avg_priority, on='requester_id', how='left')
        
        return features
    
    def _create_target(self, df: pd.DataFrame, churn_threshold_days: int = 90) -> pd.Series:
        """
        Cria variável target de churn.
        
        Args:
            df: DataFrame com dados
            churn_threshold_days: Dias sem tickets para considerar churn
        
        Returns:
            Series com labels de churn
        """
        if 'requester_id' not in df.columns or 'updated_at' not in df.columns:
            return pd.Series()
        
        # Último ticket por cliente
        last_ticket = df.groupby('requester_id')['updated_at'].max().reset_index()
        last_ticket['days_since_last'] = (
            pd.Timestamp.now() - pd.to_datetime(last_ticket['updated_at'])
        ).dt.days
        
        # Churn = sem tickets há mais de X dias
        last_ticket['churn'] = (last_ticket['days_since_last'] > churn_threshold_days).astype(int)
        
        return last_ticket.set_index('requester_id')['churn']
    
    def train(
        self,
        tickets_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Treina o modelo de churn.
        
        Args:
            tickets_df: DataFrame com tickets
            test_size: Proporção do conjunto de teste
            random_state: Seed
        
        Returns:
            Dicionário com métricas
        """
        print("Criando features...")
        features = self._create_features(tickets_df)
        target = self._create_target(tickets_df)
        
        # Merge
        data = features.merge(target, left_on='requester_id', right_index=True, how='inner')
        data = data.dropna()
        
        if len(data) == 0:
            raise ValueError("Nenhum dado válido para treinamento")
        
        # Separa features e target
        feature_cols = [col for col in data.columns if col not in ['requester_id', 'churn']]
        X = data[feature_cols].fillna(0)
        y = data['churn']
        
        self.feature_names = feature_cols
        
        # Divide treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Cria e treina modelo
        print("Treinando modelo...")
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Avalia
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Acurácia: {accuracy:.3f}")
        print(f"AUC: {auc:.3f}")
        print(f"\nRelatório:\n{classification_report(y_test, y_pred)}")
        
        # Validação cruzada
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"CV AUC Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Salva modelo
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'classification_report': report
        }
    
    def predict(self, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz risco de churn para clientes.
        
        Args:
            tickets_df: DataFrame com tickets
        
        Returns:
            DataFrame com predições
        """
        if not self.model:
            raise ValueError("Modelo não treinado. Chame train() primeiro.")
        
        features = self._create_features(tickets_df)
        feature_cols = [col for col in features.columns if col != 'requester_id']
        
        # Prepara dados
        X = features[feature_cols].fillna(0)
        
        # Predições
        churn_proba = self.model.predict_proba(X)[:, 1]
        churn_pred = self.model.predict(X)
        
        results = pd.DataFrame({
            'requester_id': features['requester_id'].values,
            'churn_probability': churn_proba,
            'churn_prediction': churn_pred
        })
        
        return results
    
    def save_model(self, path: Optional[str] = None):
        """Salva o modelo."""
        if not self.model:
            raise ValueError("Nenhum modelo para salvar.")
        
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, save_path)
        print(f"Modelo salvo em {save_path}")
    
    def load_model(self, path: Optional[str] = None):
        """Carrega modelo pré-treinado."""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Modelo não encontrado em {load_path}")
        
        model_data = joblib.load(load_path)
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names')
        self.model_type = model_data.get('model_type', 'random_forest')
        
        print(f"Modelo carregado de {load_path}")

