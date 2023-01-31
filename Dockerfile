ARG version=2021.7.1-debian-11-r6
FROM bitnami/scikit-learn-intel:$version

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
COPY ai_cloud ai_cloud

# install project deps
RUN pip3 install -r ./ai_cloud/requirements.txt

# run inference
ENTRYPOINT ["python", "ai_cloud/ai_cloud/server.py"]

