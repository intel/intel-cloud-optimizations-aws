# ===============================================================================
#  Copyright 2021-2022 Intel Corporation.
# 
#  This software and the related documents are Intel copyrighted  materials,  and
#  your use of  them is  governed by the  express license  under which  they were
#  provided to you (License).  Unless the License provides otherwise, you may not
#  use, modify, copy, publish, distribute,  disclose or transmit this software or
#  the related documents without Intel's prior written permission.
# 
#  This software and the related documents  are provided as  is,  with no express
#  or implied  warranties,  other  than those  that are  expressly stated  in the
#  License.
# ===============================================================================


FROM public.ecr.aws/docker/library/python:3.8

# copy requirement file and install python lib
COPY requirements.txt /build/
RUN pip --no-cache-dir install -r /build/requirements.txt

# install programs for proper hosting of our endpoint server
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# We update PATH so that the train and serve programs are found when the container is invoked.
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY xgboost_model_code/train /opt/program/train
COPY xgboost_model_code/serve /opt/program/serve
COPY xgboost_model_code/nginx.conf /opt/program/nginx.conf
COPY xgboost_model_code/predictor.py /opt/program/predictor.py
COPY xgboost_model_code/wsgi.py /opt/program/wsgi.py

#set executable permissions for all scripts
RUN chmod +x /opt/program/train
RUN chmod +x /opt/program/serve
RUN chmod +x /opt/program/nginx.conf
RUN chmod +x /opt/program/predictor.py
RUN chmod +x /opt/program/wsgi.py

# set the working directory
WORKDIR /opt/program
