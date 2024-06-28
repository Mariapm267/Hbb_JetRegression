import numpy as np
import pandas as pd


class Preprocessing:
    def __init__(self, file_dir):
        self.file_dir = file_dir
        self.data = pd.read_pickle(file_dir)
        
    def GetJEC(self):
        '''Gets previous Jet Energy Correction'''
        k0=self.data['Jet0_JEC_Cor'].copy()
        k1=self.data['Jet1_JEC_Cor'].copy()   
        return k0, k1
    
    def EliminateJEC(self):
        '''Provides jet cinematics without JEC'''
        k0, k1 = self.GetJEC()
        
        PT0 = self.data['Jet0_PT'].copy() / k0
        E0 = self.data['Jet0_PE'].copy() / k0
        PX0 = self.data['Jet0_PX'].copy() / k0
        PY0 = self.data['Jet0_PY'].copy() / k0
        PZ0 = self.data['Jet0_PZ'].copy() / k0

        PT1 = self.data['Jet1_PT'].copy() / k1
        E1 = self.data['Jet1_PE'].copy() / k1
        PX1 = self.data['Jet1_PX'].copy() / k1
        PY1 = self.data['Jet1_PY'].copy() / k1
        PZ1 = self.data['Jet1_PZ'].copy() / k1
        
        return PT0, E0, PX0, PY0, PZ0, PT1, E1, PX1, PY1, PZ1
	
    def DeltaR(self, JetEta, JetPhi, JetDaughtersEta, JetDaughtersPhi):
        '''Calculates DeltaR for all particles within a jet'''
        DeltaR = []
        for i in range(len(JetEta)):         
            DeltaR_i = []
            for j in range(len(JetDaughtersEta[i])):  
                Delta_Eta = JetEta[i] - JetDaughtersEta[i][j]
                Delta_Phi = JetPhi[i] - JetDaughtersPhi[i][j]
                DeltaR_ij = np.sqrt(Delta_Eta**2 + Delta_Phi**2)
                DeltaR_i.append(DeltaR_ij)
            DeltaR.append(DeltaR_i)
        return np.array(DeltaR, dtype=object)
    
    
    def GetEnergyRings(self, DeltaR, energy, ranges):
        energy_rings = []
        for r in ranges:
            ring_energy = np.array([sum([e for d, e in zip(DeltaR[i], energy[i]) if r[0] <= d < r[1]]) for i in range(len(DeltaR))])
            energy_rings.append(ring_energy)
        return energy_rings

    def MaxAndMinEnergyDepositedEcal(self, Jet_Daughters_CaloNeutralEcal):
        """Calculates the maximum and minimum energy deposited in each jet"""
        Jet_Daughters_CaloNeutralEcal_max = []
        Jet_Daughters_CaloNeutralEcal_min = []
        for Jet_Daughters_CaloNeutral_i in Jet_Daughters_CaloNeutralEcal:
            non_default_values = [e for e in Jet_Daughters_CaloNeutral_i if e != -1000]
            if non_default_values:
                Jet_Daughters_CaloNeutralEcal_min.append(min(non_default_values))
                Jet_Daughters_CaloNeutralEcal_max.append(max(non_default_values))
            else:       
                Jet_Daughters_CaloNeutralEcal_min.append(0)      
                Jet_Daughters_CaloNeutralEcal_max.append(0)  
        return np.array(Jet_Daughters_CaloNeutralEcal_max), np.array(Jet_Daughters_CaloNeutralEcal_min)



    
    def ProcessData(self, eliminate_JEC=True):
        '''Main function that processes all data'''
        
        # cinematics
        
        if eliminate_JEC:
            self.PT0, self.E0, self.PX0, self.PY0, self.PZ0, self.PT1, self.E1, self.PX1, self.PY1, self.PZ1 = self.EliminateJEC()
        else:
            self.PT0 = self.data['Jet0_PT'].copy()
            self.E0 = self.data['Jet0_PE'].copy()
            self.PX0 = self.data['Jet0_PX'].copy()
            self.PY0 = self.data['Jet0_PY'].copy()
            self.PZ0 = self.data['Jet0_PZ'].copy()
            self.PT1 = self.data['Jet1_PT'].copy()
            self.E1 = self.data['Jet1_PE'].copy()
            self.PX1 = self.data['Jet1_PX'].copy()
            self.PY1 = self.data['Jet1_PY'].copy()
            self.PZ1 = self.data['Jet1_PZ'].copy()
            
            
        # for energy rings
        self.DeltaR_0 = self.DeltaR(self.data['Jet0_Eta'], self.data['Jet0_Phi'], self.data['Jet0_Daughters_Eta'], self.data['Jet0_Daughters_Phi'])
        self.DeltaR_1 = self.DeltaR(self.data['Jet1_Eta'], self.data['Jet1_Phi'], self.data['Jet1_Daughters_Eta'], self.data['Jet1_Daughters_Phi'])
    
        ranges = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
        
        E_0_rings = self.GetEnergyRings(self.DeltaR_0, self.data['Jet0_Daughters_E'].copy(), ranges)
        E_1_rings = self.GetEnergyRings(self.DeltaR_1, self.data['Jet1_Daughters_E'].copy(), ranges)
        self.E_0_DeltaR_1, self.E_0_DeltaR_2, self.E_0_DeltaR_3, self.E_0_DeltaR_4, self.E_0_DeltaR_5 = E_0_rings
        self.E_1_DeltaR_1, self.E_1_DeltaR_2, self.E_1_DeltaR_3, self.E_1_DeltaR_4, self.E_1_DeltaR_5 = E_1_rings
    
        
        # daugthers pt
        self.Jet1daughtpt=self.data['Jet1_Daughters_pT'].copy()
        self.Jet0daughtpt=self.data['Jet0_Daughters_pT'].copy()
        self.Jet1daughtptmax=np.array([max(self.Jet1daughtpt[i]) for i in range(len(self.Jet1daughtpt))])
        self.Jet1daughtptmin=np.array([min(self.Jet1daughtpt[i]) for i in range(len(self.Jet1daughtpt))])
        self.Jet0daughtptmax=np.array([max(self.Jet0daughtpt[i]) for i in range(len(self.Jet0daughtpt))])
        self.Jet0daughtptmin=np.array([min(self.Jet0daughtpt[i]) for i in range(len(self.Jet0daughtpt))])
        
        
        # caloneutral_ecal
        self.Jet0_Daughters_CaloNeutralEcal_max, self.Jet0_Daughters_CaloNeutralEcal_min = self.MaxAndMinEnergyDepositedEcal(self.data['Jet0_Daughters_CaloNeutralEcal'])
        self.Jet1_Daughters_CaloNeutralEcal_max, self.Jet1_Daughters_CaloNeutralEcal_min = self.MaxAndMinEnergyDepositedEcal(self.data['Jet1_Daughters_CaloNeutralEcal'])
        
    def SaveDataFrame(self, filename = 'processed_dijet_bb.pickle', save_MC_variables = True):
        df=pd.DataFrame({'PX0':self.PX0,'PY0':self.PY0,'PZ0':self.PZ0,'PX1':self.PX1,'PY1':self.PY1,'PZ1':self.PZ1,
                         'Jet0_PT':self.PT0,'Jet1_PT':self.PT1,'Jet0_Eta':self.data['Jet0_Eta'],
                 'Jet1_Eta':self.data['Jet1_Eta'],'Jet0_CPF':self.data['Jet0_CPF'],
                 'Jet1_CPF':self.data['Jet1_CPF'],'Jet0_nDaughters':self.data['Jet0_nDaughters'],'Jet1_nDaughters':self.data['Jet1_nDaughters'],'nMuonTracks':self.data['nMuonTracks'],
                 'Jet0_JetWidth':self.data['Jet0_JetWidth'],'Jet1_JetWidth':self.data['Jet1_JetWidth'],'Jet1daughtptmax':self.Jet1daughtptmax,
                 'Jet1daughtptmin':self.Jet1daughtptmin,'Jet0daughtptmax':self.Jet0daughtptmax,'Jet0daughtptmin':self.Jet0daughtptmin,
                 'H_10_M':self.data['H_10_M'],'Jet0_nmuons':self.data['Jet0_nmuons'],
                 'Jet0_muon_PE':self.data['Jet0_muon_PE'],'Jet0_muon_PX':self.data['Jet0_muon_PX'],'Jet0_muon_PY':self.data['Jet0_muon_PY'],
                 'Jet0_muon_PZ':self.data['Jet0_muon_PZ'],'Jet0_muon_PT':self.data['Jet0_muon_PT'],
                 'Jet1_nmuons':self.data['Jet1_nmuons'],
                 'Jet1_muon_PE':self.data['Jet1_muon_PE'],'Jet1_muon_PX':self.data['Jet1_muon_PX'],'Jet1_muon_PY':self.data['Jet1_muon_PY'],
                 'Jet1_muon_PZ':self.data['Jet1_muon_PZ'],'Jet1_muon_PT':self.data['Jet1_muon_PT'],
                 'Jet0_Daughters_CaloNeutralEcal_max':self.Jet0_Daughters_CaloNeutralEcal_max,
                'Jet0_Daughters_CaloNeutralEcal_min':self.Jet0_Daughters_CaloNeutralEcal_min,
                'Jet1_Daughters_CaloNeutralEcal_max':self.Jet1_Daughters_CaloNeutralEcal_max,
                'Jet1_Daughters_CaloNeutralEcal_min':self.Jet1_Daughters_CaloNeutralEcal_min,
                'nTracks':self.data['nTracks'],'nPVs':self.data['nPVs'],'Jet0_PE':self.E0,'Jet1_PE':self.E1,
                'E_0_DeltaR_1':self.E_0_DeltaR_1,'E_0_DeltaR_2':self.E_0_DeltaR_2,'E_0_DeltaR_3':self.E_0_DeltaR_3,
                'E_0_DeltaR_4':self.E_0_DeltaR_4,'E_0_DeltaR_5':self.E_0_DeltaR_5
                ,'E_1_DeltaR_1':self.E_1_DeltaR_1,'E_1_DeltaR_2':self.E_1_DeltaR_2,'E_1_DeltaR_3':self.E_1_DeltaR_3,
                'E_1_DeltaR_4':self.E_1_DeltaR_4,'E_1_DeltaR_5':self.E_1_DeltaR_5})

        if save_MC_variables:    # needed for the training
            def s(E0,E1,p0x,p0y,p0z,p1x,p1y,p1z):
                return 2*(E1*E0-p0x*p1x-p0y*p1y-p0z*p1z)

            sMC=s(self.data['Jet0_MC_Jet_E'],self.data['Jet1_MC_Jet_E'],self.data['Jet0_MC_Jet_PX'],self.data['Jet0_MC_Jet_PY'],self.data['Jet0_MC_Jet_PZ'],self.data['Jet1_MC_Jet_PX'],self.data['Jet1_MC_Jet_PY'],self.data['Jet1_MC_Jet_PZ'])
            HM_MC=np.sqrt(sMC)
            df.insert(0,'HM_MC',HM_MC)  
            df.insert(0, 'Jet0_MC_Jet_E', self.data['Jet0_MC_Jet_E'])
            df.insert(0, 'Jet1_MC_Jet_E', self.data['Jet1_MC_Jet_E'])
        
        df.to_pickle(filename)
            

def RunPreprocessing(file_in, file_out, save_MC_variables = True):
    preprocessing = Preprocessing(file_in)
    preprocessing.ProcessData()
    preprocessing.SaveDataFrame(f'ProcessedDatasets/{file_out}', save_MC_variables)

