"""Dashboard Streamlit para visualização de métricas."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Adiciona caminho do projeto
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

# Importa módulos do projeto
from src.etl.database import DatabaseManager
from src.metrics.calculator import MetricsCalculator
from src.nlp.sentiment import SentimentAnalyzer


def create_app():
    """Cria e configura o dashboard Streamlit."""
    
    st.set_page_config(
        page_title="Freshdesk Analytics",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Freshdesk Analytics Dashboard")
    st.markdown("---")
    
    # Inicializa componentes
    db_path = st.sidebar.text_input("Caminho do Banco", value="data/freshdesk.db")
    
    try:
        db = DatabaseManager(db_path)
        metrics_calc = MetricsCalculator(db)
    except Exception as e:
        st.error(f"Erro ao conectar ao banco: {e}")
        st.stop()
    
    # Menu lateral
    page = st.sidebar.selectbox(
        "Página",
        ["Visão Geral", "Métricas", "Sentimentos", "Análise de Tickets"]
    )
    
    if page == "Visão Geral":
        show_overview(db, metrics_calc)
    elif page == "Métricas":
        show_metrics(db, metrics_calc)
    elif page == "Sentimentos":
        show_sentiments(db)
    elif page == "Análise de Tickets":
        show_ticket_analysis(db)


def show_overview(db: DatabaseManager, metrics_calc: MetricsCalculator):
    """Mostra visão geral do dashboard."""
    st.header("Visão Geral")
    
    # Carrega dados
    tickets_df = db.get_tickets()
    
    if len(tickets_df) == 0:
        st.warning("Nenhum ticket encontrado no banco de dados.")
        return
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tickets = len(tickets_df)
        st.metric("Total de Tickets", total_tickets)
    
    with col2:
        if 'status' in tickets_df.columns:
            resolved = (tickets_df['status'] == 4).sum()
            st.metric("Tickets Resolvidos", resolved)
        elif 'status_name' in tickets_df.columns:
            resolved = (tickets_df['status_name'] == 'Resolved').sum()
            st.metric("Tickets Resolvidos", resolved)
        else:
            st.metric("Tickets Resolvidos", "N/A")
    
    with col3:
        csat = metrics_calc.calculate_csat(tickets_df)
        if csat:
            st.metric("CSAT", f"{csat:.2f}")
        else:
            st.metric("CSAT", "N/A")
    
    with col4:
        nps_data = metrics_calc.calculate_nps(tickets_df)
        if nps_data:
            st.metric("NPS", f"{nps_data['nps']:.1f}")
        else:
            st.metric("NPS", "N/A")
    
    # Gráficos
    st.subheader("Distribuição de Tickets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'status_name' in tickets_df.columns:
            status_counts = tickets_df['status_name'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Tickets por Status"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif 'status' in tickets_df.columns:
            # Usa status numérico mas tenta mapear
            from ..utils.status_mapper import get_status_name
            status_counts = tickets_df['status'].value_counts()
            status_names = status_counts.index.map(get_status_name)
            fig = px.pie(
                values=status_counts.values,
                names=status_names,
                title="Tickets por Status"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'priority' in tickets_df.columns:
            priority_counts = tickets_df['priority'].value_counts()
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Tickets por Prioridade"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Timeline de tickets
    if 'created_at' in tickets_df.columns:
        st.subheader("Tickets ao Longo do Tempo")
        tickets_df['created_at'] = pd.to_datetime(tickets_df['created_at'])
        tickets_df['date'] = tickets_df['created_at'].dt.date
        daily_tickets = tickets_df.groupby('date').size().reset_index(name='count')
        
        fig = px.line(
            daily_tickets,
            x='date',
            y='count',
            title="Tickets Criados por Dia"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_metrics(db: DatabaseManager, metrics_calc: MetricsCalculator):
    """Mostra página de métricas detalhadas."""
    st.header("Métricas e KPIs")
    
    tickets_df = db.get_tickets()
    
    if len(tickets_df) == 0:
        st.warning("Nenhum ticket encontrado.")
        return
    
    # Calcula todas as métricas
    metrics = metrics_calc.calculate_all_metrics(tickets_df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CSAT")
        csat = metrics.get('csat')
        if csat:
            st.metric("Customer Satisfaction Score", f"{csat:.2f}/5.0")
        else:
            st.info("CSAT não disponível (sem ratings)")
    
    with col2:
        st.subheader("NPS")
        nps = metrics.get('nps')
        if nps:
            st.metric("Net Promoter Score", f"{nps:.1f}")
            promoter = metrics.get('nps_promoter_pct', 0)
            detractor = metrics.get('nps_detractor_pct', 0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Promotores', 'Detratores', 'Neutros'],
                y=[promoter, detractor, 100 - promoter - detractor],
                marker_color=['green', 'red', 'gray']
            ))
            fig.update_layout(title="Distribuição NPS")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("NPS não disponível")
    
    # SLA
    st.subheader("SLA")
    sla_data = metrics_calc.calculate_sla(tickets_df)
    if sla_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tempo Médio de Resposta", f"{sla_data['avg_response_time_hours']:.1f}h")
        with col2:
            st.metric("Conformidade SLA", f"{sla_data['sla_compliance_pct']:.1f}%")
        with col3:
            st.metric("Tickets no SLA", f"{sla_data['tickets_within_sla']}/{sla_data['total_resolved_tickets']}")


def show_sentiments(db: DatabaseManager):
    """Mostra análise de sentimentos."""
    st.header("Análise de Sentimentos")
    
    tickets_df = db.get_tickets()
    
    if len(tickets_df) == 0:
        st.warning("Nenhum ticket encontrado.")
        return
    
    if 'sentiment_label' not in tickets_df.columns:
        st.warning("Sentimentos não calculados. Execute o pipeline de NLP primeiro.")
        return
    
    # Distribuição de sentimentos
    sentiment_counts = tickets_df['sentiment_label'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Distribuição de Sentimentos"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sentiment_score' in tickets_df.columns:
            fig = px.histogram(
                tickets_df,
                x='sentiment_score',
                nbins=20,
                title="Distribuição de Scores de Sentimento"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentimentos ao longo do tempo
    if 'created_at' in tickets_df.columns:
        tickets_df['created_at'] = pd.to_datetime(tickets_df['created_at'])
        tickets_df['date'] = tickets_df['created_at'].dt.date
        daily_sentiment = tickets_df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        
        fig = px.line(
            daily_sentiment,
            x='date',
            y='count',
            color='sentiment_label',
            title="Sentimentos ao Longo do Tempo"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_ticket_analysis(db: DatabaseManager):
    """Mostra análise detalhada de tickets."""
    st.header("Análise de Tickets")
    
    tickets_df = db.get_tickets()
    
    if len(tickets_df) == 0:
        st.warning("Nenhum ticket encontrado.")
        return
    
    # Filtros
    st.sidebar.subheader("Filtros")
    
    if 'status_name' in tickets_df.columns:
        status_filter = st.sidebar.multiselect(
            "Status",
            options=sorted(tickets_df['status_name'].unique()),
            default=tickets_df['status_name'].unique()
        )
        tickets_df = tickets_df[tickets_df['status_name'].isin(status_filter)]
    elif 'status' in tickets_df.columns:
        from ..utils.status_mapper import get_status_name
        status_options = sorted(tickets_df['status'].unique())
        status_labels = {str(s): get_status_name(s) for s in status_options}
        status_filter = st.sidebar.multiselect(
            "Status",
            options=status_options,
            format_func=lambda x: f"{get_status_name(x)} ({x})",
            default=status_options
        )
        tickets_df = tickets_df[tickets_df['status'].isin(status_filter)]
    
    if 'priority' in tickets_df.columns:
        priority_filter = st.sidebar.multiselect(
            "Prioridade",
            options=tickets_df['priority'].unique(),
            default=tickets_df['priority'].unique()
        )
        tickets_df = tickets_df[tickets_df['priority'].isin(priority_filter)]
    
    # Tabela de tickets
    st.subheader("Tickets")
    
    display_cols = ['id', 'subject', 'status_name', 'status', 'priority', 'created_at']
    if 'sentiment_label' in tickets_df.columns:
        display_cols.append('sentiment_label')
    
    # Remove status_name se não existir, mantém status como fallback
    if 'status_name' not in tickets_df.columns and 'status' in tickets_df.columns:
        display_cols.remove('status_name')
    
    available_cols = [col for col in display_cols if col in tickets_df.columns]
    st.dataframe(tickets_df[available_cols], use_container_width=True)


if __name__ == "__main__":
    create_app()
else:
    # Para uso com streamlit run
    create_app()

