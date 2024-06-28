import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def standarize_train_set(train_features):
    '''
    Fits standard scaler to train set
    Returns: fitted scaler, train_set_scaled
    '''
    scaler = StandardScaler()
    train_features_sc = scaler.fit_transform(train_features)
    train_features_sc = pd.DataFrame(train_features_sc ,columns=list(train_features.columns),index=train_features.index)
    return scaler,train_features_sc


def standarize_test_set(scaler,features):
    '''Aplies the fitted scaler on a new set of features'''
    features_sc = scaler.transform(features)
    features_sc = pd.DataFrame(features_sc ,columns=list(features.columns),index=features.index)
    return features_sc
 
def training(params,train_features_sc,train_target_0,train_target_1):
    '''
    Trains both models 
        - Inputs: regressor params, features and both targets (E0_truth, E1_truth)
        - Returns: trained regressors
    '''
    
    print('\nJET 0...')
    GBR_0 = GradientBoostingRegressor(**params) 
    GBR_0.fit(train_features_sc, train_target_0)

    print('\nJET 1...')
    GBR_1 = GradientBoostingRegressor(**params)   
    GBR_1.fit(train_features_sc, train_target_1)
    print('\nTraining finished!')
    return GBR_0,GBR_1


def correction_factors(GBR_0,GBR_1,new_features_sc, E0, E1):
    '''Applies both regressors to both jets of a new set of features and determines the correction factors for the energy of both jets
    
        Inputs: 
            - Regressor trained on Jet 0
            - Regressor trained on Jet 1
            - New scaled features
        Returns: 
            -  k0 (correction factor for Jet 0)
            -  k1 (correction factor for Jet 1)
    '''
    E0_predict_test=GBR_0.predict(new_features_sc)
    E1_predict_test=GBR_1.predict(new_features_sc)
    k0=E0_predict_test/E0
    k1=E1_predict_test/E1

    return k0, k1


def gbr_invariant_mass(k0, k1, dataset):
    '''
    Inputs:   correction factor for both jets and dataset
    Returns:  invariant mass with new correction
    '''
    PX_test_0 = dataset['PX0']
    PY_test_0 = dataset['PY0']
    PZ_test_0 = dataset['PZ0']
    PX_test_1 = dataset['PX1']
    PY_test_1 = dataset['PY1']
    PZ_test_1 = dataset['PZ1']

    PX0_c,PY0_c,PZ0_c=PX_test_0*k0,PY_test_0*k0,PZ_test_0*k0
    PX1_c,PY1_c,PZ1_c=PX_test_1*k1,PY_test_1*k1,PZ_test_1*k1

    def s(m0,m1,p0x,p0y,p0z,p1x,p1y,p1z):
        E0 = np.sqrt(m0**2+p0x**2+p0y**2+p0z**2)
        E1 = np.sqrt(m1**2+p1x**2+p1y**2+p1z**2)
        s = 2*(E1*E0-p0x*p1x-p0y*p1y-p0z*p1z)
        return s

    s_c = s(0,0,PX0_c,PY0_c,PZ0_c,PX1_c,PY1_c,PZ1_c)
    mbb_GBR = np.sqrt(s_c)
    return mbb_GBR
