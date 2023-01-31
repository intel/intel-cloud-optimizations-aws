import warnings
from pathlib import Path

import daal4py as d4p
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

from base_model import AbstractModel
from logger import log
from storage import store
from .fairness import get_fairness_parity_report
from .preprocess import get_feature_names

warnings.filterwarnings("ignore")

np.random.seed(42)


class Model(AbstractModel):
    def __init__(
        self, backend="disk", bucket=None, path=None
    ):
        if path is not None:
            path = Path(path).parent.resolve()
        else:
            raise FileNotFoundError
        print(f"path here is :: {path}")
        d_path = path / "data" / "data.csv"
        print(f"path here is :: {d_path}")
        self.data = pd.read_csv(d_path)
        print(f"updated path is here is :: {d_path}")
        self.path = path / "models"
        self.store = store(
            backend=backend, bucket=bucket, path=self.path, model_name="model.joblib"
        )

    def generate_data(self, size=4000000):

        """
        Function to synthetically generate 4M (default) rows
        from loan default data.
        """

        # synthesizing bias variable
        bias_prob = 0.65
        default = self.data["loan_status"] == 1

        default_bias = np.random.choice(
            [0, 1], p=[bias_prob, 1 - bias_prob], size=len(default)
        )
        non_default_bias = np.random.choice(
            [0, 1], p=[1 - bias_prob, bias_prob], size=len(default)
        )

        # bias conditional on label
        self.data["bias_variable"] = np.where(default, default_bias, non_default_bias)

        # number of rows to generate
        if size < self.data.shape[0]:
            pass
        else:
            log.info(f"Generating {size:,} rows of data")

            repeats = size // len(self.data)
            self.data = self.data.loc[np.repeat(self.data.index.values, repeats + 1)]
            self.data = self.data.iloc[:size]

            # perturbing all int/float columns
            person_age = self.data["person_age"].values + np.random.randint(
                -1, 1, size=len(self.data)
            )
            person_income = self.data["person_income"].values + np.random.normal(
                0, 10, size=len(self.data)
            )
            person_emp_length = self.data[
                "person_emp_length"
            ].values + np.random.randint(-1, 1, size=len(self.data))
            loan_amnt = self.data["loan_amnt"].values + np.random.normal(
                0, 5, size=len(self.data)
            )
            loan_int_rate = self.data["loan_int_rate"].values + np.random.normal(
                0, 0.2, size=len(self.data)
            )
            loan_percent_income = self.data["loan_percent_income"].values + (
                np.random.randint(0, 100, size=len(self.data)) / 1000
            )
            cb_person_cred_hist_length = self.data[
                "cb_person_cred_hist_length"
            ].values + np.random.randint(0, 2, size=len(self.data))

            # perturbing all binary columns
            perturb_idx = np.random.rand(len(self.data)) > 0.1
            random_values = np.random.choice(
                self.data["person_home_ownership"].unique(), len(self.data)
            )
            person_home_ownership = np.where(
                perturb_idx, self.data["person_home_ownership"], random_values
            )

            perturb_idx = np.random.rand(len(self.data)) > 0.1
            random_values = np.random.choice(
                self.data["loan_intent"].unique(), len(self.data)
            )
            loan_intent = np.where(perturb_idx, self.data["loan_intent"], random_values)

            perturb_idx = np.random.rand(len(self.data)) > 0.1
            random_values = np.random.choice(
                self.data["loan_grade"].unique(), len(self.data)
            )
            loan_grade = np.where(perturb_idx, self.data["loan_grade"], random_values)

            perturb_idx = np.random.rand(len(self.data)) > 0.1
            random_values = np.random.choice(
                self.data["cb_person_default_on_file"].unique(), len(self.data)
            )
            cb_person_default_on_file = np.where(
                perturb_idx, self.data["cb_person_default_on_file"], random_values
            )

            self.data = pd.DataFrame(
                list(
                    zip(
                        person_age,
                        person_income,
                        person_home_ownership,
                        person_emp_length,
                        loan_intent,
                        loan_grade,
                        loan_amnt,
                        loan_int_rate,
                        self.data["loan_status"].values,
                        loan_percent_income,
                        cb_person_default_on_file,
                        cb_person_cred_hist_length,
                        self.data["bias_variable"].values,
                    )
                ),
                columns=self.data.columns,
            )

            self.data = self.data.drop_duplicates()
            assert len(self.data) == size
            self.data.reset_index(drop=True)

    def preprocess_data(self):

        """
        Function to create and preprocess training and test sets
        with 75% and 25% in each, respectively.

        Input: A pandas dataframe.
        Output: A training matrix, test set, and bias indicator
        to evaluate model fairness.
        """

        # create a hold-out set
        log.info("Creating training and test sets")
        train, test = train_test_split(self.data, test_size=0.25, random_state=0)
        num_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        pow_transformer = PowerTransformer()
        cat_transformer = OneHotEncoder(handle_unknown="ignore")
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    num_imputer,
                    [
                        "loan_int_rate",
                        "person_emp_length",
                        "cb_person_cred_hist_length",
                    ],
                ),
                (
                    "pow",
                    pow_transformer,
                    ["person_age", "person_income", "loan_amnt", "loan_percent_income"],
                ),
                (
                    "cat",
                    cat_transformer,
                    [
                        "person_home_ownership",
                        "loan_intent",
                        "loan_grade",
                        "cb_person_default_on_file",
                    ],
                ),
            ],
            remainder="passthrough",
        )

        # separate pipeline to allow for benchmarking
        preprocess = Pipeline(steps=[("preprocessor", preprocessor)])
        X_train = train.drop(["loan_status", "bias_variable"], axis=1)
        y_train = train["loan_status"]
        X_train_out = preprocess.fit_transform(X_train)
        print(f"preprocess named steps: {preprocess.named_steps['preprocessor']}, {type(preprocess.named_steps['preprocessor'])}")
        fnames = get_feature_names(preprocess.named_steps["preprocessor"])
        # create training matrix for xgboost
        self.dtrain = xgb.DMatrix(X_train_out, y_train.values)#, feature_names=X_train.columns)
        self.bias_indicator = test["bias_variable"].values.astype(int)
        X_test = test.drop(["loan_status", "bias_variable"], axis=1)
        self.y_test = test["loan_status"]
        self.X_test_out = preprocess.transform(X_test)
        self.X_test_out = pd.DataFrame(self.X_test_out)#, columns=fnames)

    def train(self):

        # define model
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "nthread": 4,  # flags.num_cpu,
            "tree_method": "hist",
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 6,
            "n_jobs": 4,  # flags.num_cpu,
            "verbosity": 0,
            "silent": 1,
        }

        log.info("Training XGBoost model")
        self.clf = xgb.train(params, self.dtrain, num_boost_round=500)

    def validate(self):
        """Validate the model performance by computing fairness metrics and classification metrics.

        Returns:
            dict: A dictionary containing the following keys:
                'auc' (float): The area under the receiver operating characteristic curve (AUC).
                'precision' (dict): A dictionary of precision scores, with keys 'Non-Default' and 'Default'.
                'recall' (dict): A dictionary of recall scores, with keys 'Non-Default' and 'Default'.
                'f1-score' (dict): A dictionary of F1 scores, with keys 'Non-Default' and 'Default'.
                'support' (dict): A dictionary of support values, with keys 'Non-Default' and 'Default'.
                'parity' (dict): A dictionary of fairness metrics, such as demographic parity and equal opportunity.
        """
        model = self.store.get()
        parity_values = get_fairness_parity_report(
            model, self.X_test_out, self.y_test, self.bias_indicator
        )
        daal_model = d4p.get_gbt_model_from_xgboost(model)
        y_hat = (
            d4p.gbt_classification_prediction(
                nClasses=2, resultsToEvaluate="computeClassProbabilities"
            )
            .compute(self.X_test_out, daal_model)
            .probabilities[:, 1]
        )
        auc = roc_auc_score(self.y_test, y_hat)
        results = classification_report(
            self.y_test,
            y_hat > 0.5,
            target_names=["Non-Default", "Default"],
            output_dict=True,
        )
        results.update({"auc": auc, "parity": parity_values})
        return results

    def infer(self, data: pd.DataFrame):
        """Returns predictions based on the input `data` using the stored model.

        Parameters:
        - data (pd.DataFrame): A pandas dataframe containing the input data for prediction.

        Returns:
        - numpy.ndarray: A binary array indicating the class predictions (True or False).
        """
        model = self.store.get()
        daal_model = d4p.get_gbt_model_from_xgboost(model)
        daal_predictions = (
            d4p.gbt_classification_prediction(
                nClasses=2, resultsToEvaluate="computeClassProbabilities"
            )
            .compute(data, daal_model)
            .probabilities[:, 1]
        )
        return daal_predictions > 0.5

    def save(self):
        log.info("Saving model")
        self.store.put(self.clf)
