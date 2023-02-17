ARG version=2021.7.1-debian-11-r6
ARG DEBIAN_FRONTEND=noninteractive

FROM bitnami/scikit-learn-intel:$version

COPY ai_cloud ai_cloud
RUN pip3 install --user -r ai_cloud/requirements.txt
ENV PATH=.local/bin:$PATH

ENTRYPOINT ["python", "ai_cloud/ai_cloud/server.py"]