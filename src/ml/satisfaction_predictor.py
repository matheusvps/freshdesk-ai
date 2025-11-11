"""Modelo de predição de satisfação."""

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class SatisfactionPredictor:
    """Preditor de satisfação (CSAT/NPS) baseado em histórico."""
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'random_forest'):
        """
        Inicializa o preditor de satisfação.
        
        Args:
            model_path: Caminho para modelo pré-treinado
            model_type: Tipo de modelo ('random_forest', 'xgboost', 'lightgbm')
        """
        self.model_path = model_path or "models/satisfaction_model.pkl"
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _create_model(self):
        """Cria o modelo baseado no tipo escolhido."""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(random_state=42)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(random_state=42, verbose=-1)
        else:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features para predição de satisfação.
        
        Args:
            df: DataFrame com dados de tickets
        
        Returns:
            DataFrame com features
        """
        features = pd.DataFrame()
        
        if 'requester_id' not in df.columns:
            return features
        
        # Features por cliente
        client_features = []
        
        # Número de tickets
        ticket_counts = df.groupby('requester_id').size().reset_index(name='ticket_count')
        client_features.append(ticket_counts)
        
        # Tempo médio de resposta
        if 'created_at' in df.columns and 'updated_at' in df.columns:
            df['response_time'] = (pd.to_datetime(df['updated_at']) - 
                                  pd.to_datetime(df['created_at'])).dt.total_seconds() / 3600
            avg_response = df.groupby('requester_id')['response_time'].mean().reset_index(name='avg_response_time')
            client_features.append(avg_response)
        
        # Sentimento médio
        if 'sentiment_score' in df.columns:
            avg_sentiment = df.groupby('requester_id')['sentiment_score'].mean().reset_index(name='avg_sentiment')
            client_features.append(avg_sentiment)
        
        # Taxa de tickets negativos
        if 'sentiment_label' in df.columns:
            negative_rate = df.groupby('requester_id')['sentiment_label'].apply(
                lambda x: (x == 'negative').sum() / len(x)
            ).reset_index(name='negative_rate')
            client_features.append(negative_rate)
        
        # Prioridade média
        if 'priority' in df.columns:
            avg_priority = df.groupby('requester_id')['priority'].mean().reset_index(name='avg_priority')
            client_features.append(avg_priority)
        
        # Merge todas as features
        if client_features:
            features = client_features[0]
            for feat in client_features[1:]:
                features = features.merge(feat, on='requester_id', how='outer')
        
        return features.fillna(0)
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Cria variável target de satisfação.
        
        Args:
            df: DataFrame com dados
        
        Returns:
            Series com scores de satisfação
        """
        if 'requester_id' not in df.columns:
            return pd.Series()
        
        # Usa sentiment_score como proxy de satisfação
        if 'sentiment_score' in df.columns:
            satisfaction = df.groupby('requester_id')['sentiment_score'].mean()
            # Normaliza para escala 0-10
            satisfaction = (satisfaction - satisfaction.min()) / (satisfaction.max() - satisfaction.min() + 1e-6) * 10
            return satisfaction
        
        return pd.Series()
    
    def train(
        self,
        tickets_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Treina o modelo de satisfação.
        
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
        feature_cols = [col for col in data.columns if col not in ['requester_id', 'satisfaction']]
        X = data[feature_cols].fillna(0)
        y = data['satisfaction']
        
        self.feature_names = feature_cols
        
        # Divide treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Normaliza features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Cria e treina modelo
        print("Treinando modelo...")
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Avalia
        y_pred = self.model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R²: {r2:.3f}")
        
        # Validação cruzada
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"CV R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Salva modelo
        self.save_model()
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    def predict(self, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz satisfação para clientes.
        
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
        X_scaled = self.scaler.transform(X)
        
        # Predições
        satisfaction_pred = self.model.predict(X_scaled)
        
        results = pd.DataFrame({
            'requester_id': features['requester_id'].values,
            'satisfaction_score': satisfaction_pred
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
            'scaler': self.scaler,
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
        self.scaler = model_data['scaler']
        self.feature_names = model_data.get('feature_names')
        self.model_type = model_data.get('model_type', 'random_forest')
        
        print(f"Modelo carregado de {load_path}")

