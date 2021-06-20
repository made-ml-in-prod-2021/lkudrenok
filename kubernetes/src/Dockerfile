FROM python:3.9-slim

COPY models/rfc_model.pkl /model.pkl
COPY src/. /src/.
RUN pip install -r src/requirements.txt

ENV PATH_TO_MODEL="/model.pkl"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
