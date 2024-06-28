import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import crystalball


'''This script takes the regression file with the invariant masses (true, JEC and GBR) and fits them into crystal balls to estimate the sensitivity improvement factor'''

file_dir="ProcessedDatasets/results_mbb"
inv_masses = pd.read_pickle(file_dir)

mbb_signal_JEC=inv_masses['mbb_signal_JEC']/1000
mbb_signal_gbr=inv_masses['mbb_signal_gbr']/1000
mbb_bkg_JEC=inv_masses['mbb_bkg_JEC']/1000
mbb_bkg_gbr=inv_masses['mbb_bkg_gbr']/1000
mbb_truth=inv_masses['mbb_truth']/1000


def crystal_ball(x, beta, m, loc, scale):
    return crystalball.pdf(x, beta, m, loc, scale)

def crystall_ball_fitting(x,bins):
    import numpy as np
    y,bin_edges=np.histogram(x,bins,density=True)
    midbins=(bin_edges[:-1] + bin_edges[1:]) / 2
    param, cov=curve_fit(crystal_ball,midbins,y,p0=[0.5,10, bin_edges[np.argmax(y)], np.std(x)])
    perror=np.sqrt(np.diag(cov))
    return (param, perror)

bins = np.linspace(0,200,40).tolist()
p_GBR, error_GBR = crystall_ball_fitting(mbb_signal_gbr,bins)
p_JEC, error_JEC = crystall_ball_fitting(mbb_signal_JEC,bins)
p_truth, error_truth = crystall_ball_fitting(mbb_truth,bins)


print('Estimating sensitivity improvement...')

_,_,mean_gbr,sigma_gbr=p_GBR
_,_,mean_JEC,sigma_JEC=p_JEC

#integration limits at  mu ± sigma
lim_inf_gbr,lim_sup_gbr=mean_gbr - sigma_gbr, mean_gbr + sigma_gbr
lim_inf_JEC,lim_sup_JEC=mean_JEC - sigma_JEC,mean_JEC + sigma_JEC



fig=plt.figure(figsize=(10,8))
d=plt.hist(mbb_bkg_JEC, bins, color='black', histtype='step', density=True, label='Bkg with JEC' )
m=plt.hist(mbb_bkg_gbr, bins, color='grey', histtype='step', density=True, label='Bkg with GBR ')

plt.xlabel(r'$m_{b\bar{b}}$  (GeV)')
plt.ylabel('Normalizado a 1')
plt.legend(fontsize=12)
plt.xlim(25,200)
plt.savefig('figs/mbb_bkg.jpg', dpi=500, bbox_inches='tight')

B_regresion=0
for i in range(len(mbb_bkg_gbr)):
    if mbb_bkg_gbr[i]<lim_sup_gbr and mbb_bkg_gbr[i]>lim_inf_gbr:
        B_regresion+=1

B_data=0
for i in range(len(mbb_bkg_JEC)):
    if mbb_bkg_JEC[i]<lim_sup_JEC and mbb_bkg_JEC[i]>lim_inf_JEC:
        B_data+=1
        
factor=np.sqrt(B_data/B_regresion)
print('Sensitivity improvement:',factor)

fig=plt.figure(figsize=(10,8))

#histograms
plt.hist(mbb_signal_gbr, bins, histtype='step', color='red', density=True)
plt.hist(mbb_signal_JEC, bins, histtype='step', color='blue', density=True)
plt.hist(mbb_truth, bins, histtype='step', color='green', density=True)


#fitting
t=np.linspace(0,200,200)
plt.plot(t,crystal_ball(t,*p_GBR),color='red',label=r'$m_{b \bar{b}}$ GBR')
plt.plot(t,crystal_ball(t,*p_JEC),color='blue',label=r'$m_{b \bar{b}}$ JEC')
plt.plot(t,crystal_ball(t,*p_truth),color='green',label=r'$m_{b \bar{b}}$ true')

#text with sigmas 
plt.text(180, 0.016, 'JEC: $\sigma$=(%.2f $\pm$ %.2f) GeV\n \nGBR: $\sigma$=(%.2f $\pm$ %.2f) GeV\n \nTruth: $\sigma$=(%.2f $\pm$ %.2f) GeV' 
        % (p_JEC[3],error_JEC[3],p_GBR[3],error_GBR[3],p_truth[3],error_truth[3]), horizontalalignment='center',fontsize = 12)

plt.legend(loc='upper right',fontsize=14)
plt.xlabel(r'$m_{b\bar{b}}$  (GeV)',fontsize=14)
plt.ylabel('Normalized to Unity',fontsize=14)
plt.text(23, 0.025, s = 'LHCb  simulation preliminary', fontsize=14)
plt.xlim(15,220)
plt.ylim(0,0.028)

plt.savefig('figs/mbb_corrected.jpg', dpi=500, bbox_inches='tight')