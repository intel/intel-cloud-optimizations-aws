from flask import Flask, jsonify

from model import Model

app = Flask(__name__)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model = Model(backend="disk")
    return jsonify({"result": model.infer().tolist()})


def serve(port=8030):
    """start server."""
    app.run(port=port)
