import os

# Find current file absolute path
os.environ.get("PLMFIT_PATH", os.path.dirname(os.path.abspath(__file__)))
