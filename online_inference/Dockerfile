FROM python:3.9-slim

COPY models/rfc_model.pkl /model.pkl

COPY dist/ml_project_inference-0.1.0.tar.gz /ml_project_inference-0.1.0.tar.gz
RUN pip install /ml_project_inference-0.1.0.tar.gz

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"

CMD ["run_service", "0.0.0.0", "80"]
