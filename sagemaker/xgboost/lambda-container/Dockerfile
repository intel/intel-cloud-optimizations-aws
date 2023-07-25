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

# download base image
FROM public.ecr.aws/lambda/python:3.8

# copy our lambda handler script
COPY app.py ${LAMBDA_TASK_ROOT}


# install our dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip3 --no-cache-dir install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# execute function for sagemaker endpoint prediction
RUN echo Utilizing SageMaker Endpoint
CMD [ "app.sagemaker_endpoint" ]
