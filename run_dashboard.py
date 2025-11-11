"""Script para executar o dashboard Streamlit."""

import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # Encontra o caminho do dashboard
    project_root = Path(__file__).parent
    dashboard_path = project_root / "src" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Erro: Dashboard não encontrado em {dashboard_path}")
        sys.exit(1)
    
    # Executa o dashboard
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

