FROM python:3.9-slim

COPY requirements.txt /requirements.txt
COPY predict.py /predict.py

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR .

ENTRYPOINT ["python", "predict.py"]