FROM python:3.12
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir pandas-ta \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY models/ /app/models/
COPY utils/ /app/utils/  
COPY app.py .           

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]