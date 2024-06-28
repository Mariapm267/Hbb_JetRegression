from preprocessing import RunPreprocessing
import regression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


print('**********************************************')
print('************** H to bb regression ************')
print('**********************************************')

print('\n 1) Preprocessing...\n')

  
need_preprocessing = False      # if datasets are not preprocessed yet
if need_preprocessing:
    signal_bb = './Datasets/Dijet_Higgsbb_skim_arrays.pickle'
    bkg_bb    = './Datasets/DiJet_Data.pickle'
  
    RunPreprocessing(file_in = signal_bb, file_out = 'processed_bb_signal')
    RunPreprocessing(file_in = bkg_bb, file_out = 'processed_bb_bkg', save_MC_variables= False)
    print('Done!')

else:
    print('Already done!')
    
    
print('\n2) Regression...\n')

signal_processed = 'ProcessedDatasets/processed_bb_signal'
bkg_processed = 'ProcessedDatasets/processed_bb_bkg'
df_signal = pd.read_pickle(signal_processed)
df_bkg=pd.read_pickle(bkg_processed)


train_set, test_set = train_test_split(df_signal, test_size=0.2, random_state=42)   


# TRAIN FEATURES AND TARGET DEFINITION
variables=['Jet0_PT', 'Jet1_PT', 'Jet0_Eta', 'Jet1_Eta', 'Jet0_CPF', 'Jet1_CPF', 'Jet0_nDaughters', 'Jet1_nDaughters',
           'nMuonTracks','Jet0_JetWidth','Jet1_JetWidth', 'Jet1daughtptmax','Jet1daughtptmin','Jet0daughtptmax',
           'Jet0daughtptmin','Jet0_nmuons','Jet0_muon_PZ','Jet0_muon_PT','Jet1_nmuons','Jet1_muon_PE', 'Jet1_muon_PZ',
           'Jet1_muon_PT','Jet0_Daughters_CaloNeutralEcal_max','Jet0_Daughters_CaloNeutralEcal_min','Jet1_Daughters_CaloNeutralEcal_max',
           'Jet1_Daughters_CaloNeutralEcal_min', 'nTracks','nPVs','E_0_DeltaR_1','E_0_DeltaR_2','E_0_DeltaR_3','E_0_DeltaR_4','E_0_DeltaR_5', 
           'E_1_DeltaR_1', 'E_1_DeltaR_2', 'E_1_DeltaR_3','E_1_DeltaR_4','E_1_DeltaR_5']


# features
train_features=train_set[variables]
test_features=test_set[variables]
bkg_features=df_bkg[variables]


# targets for training
train_target_0=train_set['Jet0_MC_Jet_E']  
train_target_1=train_set['Jet1_MC_Jet_E']  


# scaling 
scaler,train_features_sc = regression.standarize_train_set(train_features)
test_features_sc = regression.standarize_test_set(scaler, test_features)
bkg_features_sc = regression.standarize_test_set(scaler, bkg_features)


print('Training...')
params = {'n_estimators': 850, 'min_samples_split': 20, 'min_samples_leaf': 15, 'max_depth': 5, 'loss': 'huber', 'learning_rate': 0.05}
GBR_0,GBR_1 = regression.training(params, train_features_sc, train_target_0, train_target_1)

print('Aplying regressor to test set...')
k0_signal,k1_signal = regression.correction_factors(GBR_0,GBR_1,test_features_sc, test_set['Jet0_PE'],test_set['Jet1_PE'])
mbb_signal_GBR = regression.gbr_invariant_mass(k0_signal, k1_signal, dataset = test_set)

print('Aplying regressor to bkg dataset...')
k0_bkg,k1_bkg = regression.correction_factors(GBR_0,GBR_1,bkg_features_sc,df_bkg['Jet0_PE'],df_bkg['Jet1_PE'])
mbb_bkg_GBR = regression.gbr_invariant_mass(k0_bkg, k1_bkg, dataset = df_bkg)


print('Saving results...')
results = pd.DataFrame({'mbb_signal_JEC':test_set['H_10_M'],'mbb_signal_gbr':mbb_signal_GBR,'mbb_bkg_JEC':df_bkg['H_10_M'],'mbb_bkg_gbr':mbb_bkg_GBR,'mbb_truth':test_set['HM_MC']})
file_dir = "ProcessedDatasets/results_mbb"
results.to_pickle(file_dir)

