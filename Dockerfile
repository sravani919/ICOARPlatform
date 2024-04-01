FROM nikolaik/python-nodejs:python3.10-nodejs21-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    wget \
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

WORKDIR /app/model

RUN wget https://drive.usercontent.google.com/download?id=1D-DK__8fRpIwt1Xqe5xRg2brNrjVIqYx&export=download&authuser=0&confirm=t&uuid=fdf98500-6591-4f4a-a2ae-32520a8870b9&at=APZUnTW5_Gr07qEsjOps53gV7AFg%3A1712004489631

WORKDIR /app

RUN pip install gdown

RUN pip install replicate

RUN pip install pyLDAvis

RUN gdown 1D-DK__8fRpIwt1Xqe5xRg2brNrjVIqYx

RUN mkdir -p /app/data/images/image

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
