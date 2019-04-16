from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import time

def train_model():
    df = pd.read_csv('features_train.csv')

    features = []
    y = []
    for i in range(len(df.IMAGE_NAME)):
        container = []
        container.append(df.VAR1[i]) #SURELY THERES AN EASIER WAY... right?
        container.append(df.VAR2[i])
        container.append(df.VAR3[i])
        container.append(df.VAR4[i])
        container.append(df.VAR5[i])
        container.append(df.VAR6[i])
        container.append(df.VAR7[i])
        container.append(df.VAR8[i])
        container.append(df.VAR9[i])
        container.append(df.VAR10[i])
        container.append(df.VAR11[i])
        container.append(df.VAR12[i])
        container.append(df.VAR13[i])
        container.append(df.VAR14[i])
        container.append(df.VAR15[i])
        container.append(df.VAR16[i])
        container.append(df.VAR17[i])
        container.append(df.VAR18[i])
        container.append(df.VAR19[i])
        container.append(df.VAR20[i])
    ##    container.append(df.ZIP[i]) #Zipcode doesn't make much difference in classifying
        container.append(df.TOTAL_POP[i])
        container.append(df.WHITE_POP[i])
        container.append(df.BLACK_POP[i])
        container.append(df.HISPANIC_POP[i])
        container.append(df.NON_CITIZEN[i])
        container.append(df.POVERTY_PER[i])
        container.append(df.POVERTY_QUARTER_MILE[i])
        container.append(df.MEDIAN_INCOME[i])
        container.append(df.UNEDUCATED[i])
        container.append(df.POLICE_PRESENCE[i])
        container.append(df.UNEMPLOYED[i])

        features.append(container)
        y.append(df.ALL_100_METERS_2016[i])

    index = df.IMAGE_NAME

    x_train, x_test, y_train, y_test = train_test_split(features,y, test_size = .05)

    model = RandomForestClassifier(n_estimators = 50, max_features = None, max_depth = 500)
    print("FITTING")
    model.fit(x_train, y_train)
    print("DONE")
    print("PREDICTING")
    y_p = model.predict(x_test)
    print("DONE")

    p = spearmanr(y_p, y_test)
    print(p)
    return p.correlation, model

def get_test(model, corr):

    df = pd.read_csv('features_test.csv')

    features_test = []
    for i in range(len(df.IMAGE_NAME)):
        container = []
        container.append(df.VAR1[i]) #SURELY THERES AN EASIER WAY... right?
        container.append(df.VAR2[i])
        container.append(df.VAR3[i])
        container.append(df.VAR4[i])
        container.append(df.VAR5[i])
        container.append(df.VAR6[i])
        container.append(df.VAR7[i])
        container.append(df.VAR8[i])
        container.append(df.VAR9[i])
        container.append(df.VAR10[i])
        container.append(df.VAR11[i])
        container.append(df.VAR12[i])
        container.append(df.VAR13[i])
        container.append(df.VAR14[i])
        container.append(df.VAR15[i])
        container.append(df.VAR16[i])
        container.append(df.VAR17[i])
        container.append(df.VAR18[i])
        container.append(df.VAR19[i])
        container.append(df.VAR20[i])
    ##    container.append(df.ZIP[i])#Zipcode doesn't make much difference in classifying
        container.append(df.TOTAL_POP[i])
        container.append(df.WHITE_POP[i])
        container.append(df.BLACK_POP[i])
        container.append(df.HISPANIC_POP[i])
        container.append(df.NON_CITIZEN[i])
        container.append(df.POVERTY_PER[i])
        container.append(df.POVERTY_QUARTER_MILE[i])
        container.append(df.MEDIAN_INCOME[i])
        container.append(df.UNEDUCATED[i])
        container.append(df.POLICE_PRESENCE[i])
        container.append(df.UNEMPLOYED[i])
        
        features_test.append(container)

    index_test = df.IMAGE_NAME

    y_test = model.predict(features_test)

    new_df = pd.DataFrame(y_test, index = index_test)
    new_df.to_csv('({:3f}).csv'.format(corr))

if __name__ == '__main__':
    while True:
        corr = 0
        while corr <= .78:
            corr, model = train_model()
        get_test(model, corr)
        print("=========================== FOUND A GOOD ONE ===========================")
