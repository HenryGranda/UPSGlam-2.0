FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Copiar solo requirements primero (mejor cache)
COPY requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# 2) Ahora sí, copiar TODO el código
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
