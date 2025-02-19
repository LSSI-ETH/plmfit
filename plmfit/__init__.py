__version__ = "1.0.0"
__author__ = "Thomas Bikias, Evangelos Stamkopoulos"
__credits__ = "Argonne National Laboratory"

import os

# Find current file absolute path
plmfit_path = os.getenv("PLMFIT_PATH", os.path.dirname(os.path.abspath(__file__)))
os.environ["PLMFIT_PATH"] = plmfit_path
