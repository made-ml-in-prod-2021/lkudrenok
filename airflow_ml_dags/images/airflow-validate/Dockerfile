FROM python:3.9-slim

COPY requirements.txt /requirements.txt
COPY validate.py /validate.py

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR .

ENTRYPOINT ["python", "validate.py"]