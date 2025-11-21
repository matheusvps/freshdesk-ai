"""Análise de sentimentos usando modelos locais."""

import os
import pickle
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Tenta importar NLTK para stop words em português
try:
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# spaCy é opcional - não importa no nível do módulo para evitar problemas
# O import será feito apenas quando necessário
SPACY_AVAILABLE = False


class SentimentAnalyzer:
    """Analisador de sentimentos usando modelos locais."""
    
    def __init__(self, model_path: Optional[str] = None, language: str = 'pt'):
        """
        Inicializa o analisador de sentimentos.
        
        Args:
            model_path: Caminho para modelo pré-treinado (opcional)
            language: Idioma ('pt' ou 'en')
        """
        self.model_path = model_path or "models/sentiment_model.pkl"
        self.language = language
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        
        # Carrega modelo se existir
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def _get_stop_words(self):
        """Obtém lista de stop words baseado no idioma."""
        if self.language == 'pt':
            # Para português, tenta usar NLTK, senão usa lista básica
            if NLTK_AVAILABLE:
                try:
                    return stopwords.words('portuguese')
                except:
                    pass
            # Lista básica de stop words em português
            return ['a', 'o', 'e', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'suas', 'numa', 'pelos', 'pelas', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'pelas', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm']
        else:
            # Para inglês, usa stop words padrão do scikit-learn
            return 'english'
    
    def _create_pipeline(self):
        """Cria pipeline de processamento."""
        # Obtém stop words
        stop_words = self._get_stop_words()
        
        # Usa TF-IDF com n-grams
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            lowercase=True,
            stop_words=stop_words
        )
        
        # Classificador
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    
    def _create_default_model(self):
        """Cria um modelo padrão treinado com dados sintéticos."""
        self._create_pipeline()
        
        # Dados sintéticos básicos para treinar o modelo
        synthetic_texts = [
            # Positivos
            "muito bom excelente ótimo satisfeito feliz agradecido",
            "perfeito maravilhoso adorado incrível fantástico",
            "gostei muito obrigado pela ajuda excelente serviço",
            "resolvido rapidamente muito eficiente parabéns",
            "atendimento perfeito muito satisfeito recomendo",
            # Negativos
            "ruim péssimo insatisfeito decepcionado frustrado",
            "horrível terrível não funciona problema erro",
            "lento demorado não resolveu nada inútil",
            "péssimo atendimento muito ruim não recomendo",
            "problema não resolvido insatisfeito reclamar",
            # Neutros
            "ok tudo bem normal regular sem problemas",
            "funcionou como esperado sem novidades",
            "resolvido sem problemas tudo certo",
            "atendimento normal sem reclamações",
            "tudo bem funcionando normalmente"
        ]
        
        synthetic_labels = [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral'
        ]
        
        # Treina o modelo com dados sintéticos
        print("  Treinando modelo padrão com dados sintéticos...")
        try:
            self.pipeline.fit(synthetic_texts, synthetic_labels)
            print("  [OK] Modelo padrão criado e treinado")
        except Exception as e:
            print(f"  [AVISO] Erro ao treinar modelo padrão: {e}")
            # Se falhar, pelo menos cria o pipeline (mas não treinado)
            self._create_pipeline()
    
    def train(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> dict:
        """
        Treina o modelo de sentimentos.
        
        Args:
            texts: Lista de textos
            labels: Lista de labels ('positive', 'negative', 'neutral')
            test_size: Proporção do conjunto de teste
            random_state: Seed para reprodutibilidade
        
        Returns:
            Dicionário com métricas de avaliação
        """
        if not self.pipeline:
            self._create_pipeline()
        
        # Remove textos vazios
        data = pd.DataFrame({'text': texts, 'label': labels})
        data = data[data['text'].str.len() > 0].copy()
        
        if len(data) == 0:
            raise ValueError("Nenhum texto válido fornecido para treinamento")
        
        # Divide em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            data['text'].values,
            data['label'].values,
            test_size=test_size,
            random_state=random_state,
            stratify=data['label'].values
        )
        
        # Treina
        print("Treinando modelo de sentimentos...")
        self.pipeline.fit(X_train, y_train)
        
        # Avalia
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Acurácia: {accuracy:.3f}")
        print(f"\nRelatório de classificação:\n{classification_report(y_test, y_pred)}")
        
        # Salva modelo
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Prediz sentimento de um texto.
        
        Args:
            text: Texto a ser analisado
        
        Returns:
            Tupla (label, confidence)
        """
        if not self.pipeline:
            raise ValueError("Modelo não treinado. Chame train() primeiro ou carregue um modelo.")
        
        if not text or len(text.strip()) == 0:
            return 'neutral', 0.5
        
        # Predição
        label = self.pipeline.predict([text])[0]
        
        # Probabilidades
        probabilities = self.pipeline.predict_proba([text])[0]
        confidence = float(np.max(probabilities))
        
        return label, confidence
    
    def predict_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Prediz sentimentos de múltiplos textos.
        
        Args:
            texts: Lista de textos
        
        Returns:
            DataFrame com predições
        """
        if not self.pipeline:
            raise ValueError("Modelo não treinado.")
        
        results = []
        for text in texts:
            label, confidence = self.predict(text)
            results.append({
                'text': text,
                'sentiment_label': label,
                'sentiment_score': confidence
            })
        
        return pd.DataFrame(results)
    
    def save_model(self, path: Optional[str] = None):
        """Salva o modelo treinado."""
        if not self.pipeline:
            raise ValueError("Nenhum modelo para salvar.")
        
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        joblib.dump(self.pipeline, save_path)
        print(f"Modelo salvo em {save_path}")
    
    def load_model(self, path: Optional[str] = None):
        """Carrega modelo pré-treinado."""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Modelo não encontrado em {load_path}")
        
        self.pipeline = joblib.load(load_path)
        print(f"Modelo carregado de {load_path}")
    
    def create_training_data_from_tickets(
        self,
        tickets_df: pd.DataFrame,
        text_column: str = 'cleaned_text',
        rating_column: str = 'satisfaction_rating'
    ) -> Tuple[List[str], List[str]]:
        """
        Cria dados de treinamento a partir de tickets com ratings.
        
        Args:
            tickets_df: DataFrame de tickets
            text_column: Coluna com texto
            rating_column: Coluna com rating de satisfação
        
        Returns:
            Tupla (texts, labels)
        """
        # Filtra tickets com rating
        df = tickets_df[
            tickets_df[rating_column].notna() &
            (tickets_df[text_column].str.len() > 10)
        ].copy()
        
        if len(df) == 0:
            print("Nenhum ticket com rating encontrado.")
            return [], []
        
        # Mapeia ratings para sentimentos
        def rating_to_sentiment(rating):
            if pd.isna(rating):
                return None
            
            rating_str = str(rating).lower()
            if 'good' in rating_str or 'satisfied' in rating_str or rating_str == '5':
                return 'positive'
            elif 'bad' in rating_str or 'dissatisfied' in rating_str or rating_str == '1':
                return 'negative'
            else:
                return 'neutral'
        
        df['sentiment'] = df[rating_column].apply(rating_to_sentiment)
        df = df[df['sentiment'].notna()].copy()
        
        texts = df[text_column].tolist()
        labels = df['sentiment'].tolist()
        
        print(f"Criados {len(texts)} exemplos de treinamento")
        
        return texts, labels

