__version__ = "1.0.0"
__author__ = "Thomas Bikias, Evangelos Stamkopoulos"
__credits__ = "Argonne National Laboratory"

import os

# Find current file absolute path
os.environ.get("PLMFIT_PATH", os.path.dirname(os.path.abspath(__file__)))
