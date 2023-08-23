import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Read CSV file
with open("data/size-Pd_June2023.csv") as fp:
    reader = csv.reader(fp, delimiter=",")