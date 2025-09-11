FROM python:3.12
WORKDIR /usr/local/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py regression.joblib ./
EXPOSE 5212

RUN useradd app
USER app

CMD ["python", "app.py"]
