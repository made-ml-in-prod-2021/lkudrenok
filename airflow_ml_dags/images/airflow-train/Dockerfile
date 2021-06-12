FROM python:3.9-slim

COPY requirements.txt /requirements.txt
COPY train.py /train.py

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR .

ENTRYPOINT ["python", "train.py"]