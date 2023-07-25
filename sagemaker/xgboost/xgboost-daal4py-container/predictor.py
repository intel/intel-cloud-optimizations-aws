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

# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import http.client
import numpy as np
import daal4py as d4p

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            with open(os.path.join(model_path, "xgboost-model"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input, daal_opt=False):
        """Receives an input and conditionally optimizes xgboost model using daal4py conversion.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""

        clf = cls.get_model()
        
        if daal_opt:
            daal_model = d4p.get_gbt_model_from_xgboost(clf.get_booster())
            return d4p.gbt_classification_prediction(nClasses=2, resultsToEvaluate='computeClassProbabilities', fptype='float').compute(input, daal_model).probabilities[:,1]
            
        return clf.predict(input)
        
def process_payload(payload):
    sample = list(payload.split(","))
    processed = np.asarray(sample, dtype=float)
    processed = processed.reshape(1,-1)
    return processed


def encode_predictions_as_json(predictions):
    """Encode the selected predictions based on the JSON output format expected.
    :param predictions: list of predictions.
    :return: encoded content in JSON
        example: b'{"predictions": [{"score": 0.43861907720565796},
        {"score": 0.4533972144126892}, {"score": 0.06351257115602493}]}'
    """
    preds_list_of_dict = []
    for pred in predictions:
        preds_list_of_dict.append({"score": pred})
    return json.dumps({"predictions": preds_list_of_dict})

# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invocations():
    payload = flask.request.data.strip().decode("utf-8")
    if len(payload) == 0:
        return flask.Response(response="no valid data passed", status=http.client.NO_CONTENT)
        
    try:
        model_load_test = ScoringService.get_model()
    except Exception as e:
        return flask.Response(response="Unable to load model", status=http.client.INTERNAL_SERVER_ERROR)
    
    processed_payload = process_payload(payload)

    pred = ScoringService.predict(processed_payload, daal_opt=True)
    
    pred_list = pred.tolist()
    
    encoded_pred = encode_predictions_as_json(pred_list)

    return flask.Response(response=encoded_pred, status=200, mimetype='text/csv')
