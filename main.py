import numpy as np
import pandas as pd 
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 20

import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile

import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as st

api = KaggleApi()
api.authenticate()

api.competition_download_files("house-prices-advanced-regression-techniques")


zf = ZipFile('house-prices-advanced-regression-techniques.zip')
zf.extractall("data/")
zf.close()


# \o/ Hello Catherine \o/


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print(train)

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
quantitative.remove('Id')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']




"""
api.competition_submit(
    'gender_submission.csv', #name of saved results
    'API Submission',
    'titanic' #name of competition
    )
"""
