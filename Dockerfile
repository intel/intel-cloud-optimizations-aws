# set env variables
ARG version=2021.7.1-debian-11-r6
ARG DEBIAN_FRONTEND=noninteractive

# pull binami scikit intel image
FROM bitnami/scikit-learn-intel:$version

# copy assets over to image
COPY /app /app
RUN pip3 install --user --no-cache-dir -r requirements.txt
ENV PATH=.local/bin:$PATH

# set the working directory
WORKDIR /app

# export port for ELB
EXPOSE 5000

ENTRYPOINT ["python", "server.py"]
