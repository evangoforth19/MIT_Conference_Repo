# ===========================
# General External Import Packages
# ===========================

# --- PyBaseball ---
from pybaseball import (
    batting_stats, batting_stats_bref,
    cache, playerid_lookup, playerid_reverse_lookup,
    pitching_stats, pitching_stats_range,
    statcast, statcast_batter, statcast_pitcher,
    standings, team_batting, team_pitching
)

# --- Enable Cache ---
cache.enable()

# --- Standard Library ---
import os
import random
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path

# --- Data Handling ---
import numpy as np
import pandas as pd

# --- Math & Stats ---
from scipy.interpolate import griddata
from scipy.stats import sem
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- Scikit-Learn ---
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, PolynomialFeatures, StandardScaler
)

# --- Torch (Deep Learning) ---
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
"""

# --- Plotting & Visualization ---
import matplotlib.pyplot as plt
#import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# --- Web Scraping ---
import requests
from bs4 import BeautifulSoup

# --- External Utilities ---
import joblib
import mlbgame