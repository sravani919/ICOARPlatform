FROM nikolaik/python-nodejs:python3.10-nodejs21-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/citations

RUN npm install --legacy-peer-deps \
    && npm run build

WORKDIR /app/corousel

RUN npm install --legacy-peer-deps \
    && npm run build

WORKDIR /app/header_tab

RUN npm install --legacy-peer-deps \
    && npm run build

WORKDIR /app

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

RUN mkdir -p /app/model

RUN mkdir -p /app/models

RUN pip install gdown

RUN gdown 1D-DK__8fRpIwt1Xqe5xRg2brNrjVIqYx

RUN mkdir -p /app/data/images/image

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
