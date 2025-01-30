from plmfit.functions.extract_embeddings import extract_embeddings
from plmfit.functions.fine_tune import fine_tune
from plmfit.functions.onehot import onehot
from plmfit.functions.feature_extraction import feature_extraction
from plmfit.functions.predict import predict
from plmfit.functions.blosum62 import blosum
from plmfit.functions.categorical_train import categorical_train
import os

# Find current file absolute path
os.environ.get("PLMFIT_PATH", os.path.dirname(os.path.abspath(__file__)))
