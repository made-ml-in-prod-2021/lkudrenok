FROM python:3.9-slim

COPY requirements.txt /requirements.txt
COPY preprocess.py /preprocess.py

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR .

ENTRYPOINT ["python", "preprocess.py"]