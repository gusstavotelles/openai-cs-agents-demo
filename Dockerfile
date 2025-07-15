FROM python:3.12-slim

# Instala dependências do sistema (opcional, pode ser ajustado conforme necessidade)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Copia requirements e instala dependências Python
COPY python-backend/requirements.txt ./python-backend/requirements.txt
RUN pip install --no-cache-dir -r python-backend/requirements.txt

# Copia todo o código do backend
COPY python-backend/ ./python-backend/
COPY data/ ./data/

# Expõe a porta padrão do FastAPI/Uvicorn
EXPOSE 8000

# Comando para rodar o servidor
CMD ["uvicorn", "python-backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
