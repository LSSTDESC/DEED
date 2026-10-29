"""
Adapting the chain plotter from Y3 extensions for use in Y6.
It's basically just a wrapper for getdist plotting, with some added
functionality for reading in cosmosis chains, adding derived parameters,
computing summary statistics.

To use, import this script into your plotting script or notebook as a module.
If you want to do this for a script that's in a directory other
than where this one is, you can do this:
import sys
chainplotdir = os.environ['Y6METHODSDIR']+"/cosmosis/plot_chains"
sys.path.insert(0,chainplotdir)
cpu = __import__('getdist_chainplot_utils')

Contact jessie muir with questions

April 15 2024 - moving some functions focused more on summary stats and
model comparison over to stat_utils.py. This should remove the dependence of this
script on anesthetic.
"""
import os, sys
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import math, copy

import getdist as gd
import getdist.plots as gdplots
from getdist.mcsamples import MCSamples

#from anesthetic import NestedSamples
from scipy.stats import chi2
from scipy.special import erfinv, erfcinv
from scipy.interpolate import interp1d

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
from matplotlib import rc
rc('text',usetex=True)
import colorsys
from matplotlib.ticker import FormatStrFormatter

default_callfrom = os.environ['Y6METHODSDIR']

# light blue, dark blue, l green, d green, peach, red
DEFAULT_COLORS = ['#a6cee3',\
                  '#1f78b4',\
                  '#b2df8a',\
                  '#33a02c',\
                  '#fb9a99',\
                  '#e31a1c',\
                  '#fdbf6f',\
                  '#ff7f00',\
                  '#cab2d6',\
                  '#6a3d9a',\
                  '#ffff99',\
                  '#b15928']
# ^ just grabbed the 12 class qualitative color ist from colorbrewer2.org

# colors used in Y3ext paper
lightblue='#a6cee3'
midblue = '#1f78b4'
darkblue = '#03538A'
lightgreen = '#b2df8a'
midgreen = '#33a02c'
pink='#fb9a99'
red='#e31a1c'
lightred='#f6baba'#'#ee7576' # for filled planck results
darkred='#710d0e'
lightorange='#fdbf6f'
orange='#ff7f00'
lightpurple='#cab2d6'
purple='#6a3d9a'
darkpurple='#4E049F'
yellow='#ffed6f'
brown='#b15928'
salmon='#fc8d62'
beige1='#f7f5eb' #beige
beige2='#e8e6df'
beige3='#deddd9'

#==================================================================
# Defaults for convenience.
#==================================================================

OMNUH2_TO_MNU_FACTOR = 93.14 # for active neutrinos

# TO DO CHECK AGAINST Y6 SETUP
DEFAULT_FILE_PREFIX = 'chain'
DEFAULT_FILE_SUFFIX = '.txt'

# TO DO, add TATT params
# tex labels
DEFAULT_PLABELS = {\
                   'cosmological_parameters--omega_m':r'$\Omega_{\rm m}$',\
                   'cosmological_parameters--w':r'$w_0$',\
                   'cosmological_parameters--wa':r'$w_a$',\
                   'cosmological_parameters--wp':r'$w_{\rm p}$',\
                   'a_pivot':r'$a_{\rm p}$',\
                   'z_pivot':r'$z_{\rm p}$',\
                   'cosmological_parameters--omega_k':r'$\Omega_k$',\
                   'cosmological_parameters--omega_c':r'$\Omega_{\rm c}$',\
                   'COSMOLOGICAL_PARAMETERS--SIGMA_8':r'$\sigma_8$',\
                   'cosmological_parameters--sigma_8':r'$\sigma_8$',\
                   'S8':r'$S_8$',\
                   'cosmological_parameters--h0':r'$h$',\
                   'cosmological_parameters--omega_b':r'$\Omega_b$',\
                   'cosmological_parameters--n_s':r'$n_s$',\
                   'cosmological_parameters--a_s':r'$A_s$',\
                   'cosmological_parameters--a_s_1e9':r'$10^9A_s$',\
                   'cosmological_parameters--omnuh2':r'$\Omega_{\nu}h^2$',\
                   'cosmological_parameters--ommh2':r'$\Omega_{\rm m}h^2$',\
                   'cosmological_parameters--ombh2':r'$\Omega_{\rm b}h^2$',\
                   'cosmological_parameters--omch2':r'$\Omega_{\rm c}h^2$',\
                   'cosmological_parameters--alens':r'$A_{\rm L}$',\
                   'cosmological_parameters--tau':r'$\tau$',\
                   'summnu':r'$\sum m_{\nu}\,[{\rm eV}]$',\
                   'cosmological_parameters--mnu':r'$\sum m_{\nu}\,[{\rm eV}]$',\
                   'supernova_params--m':r'$M_{\rm SN}$',\
                   'bin_bias--b1':r'$b^{[1]}$',\
                   'bin_bias--b2':r'$b^{[2]}$',\
                   'bin_bias--b3':r'$b^{[3]}$',\
                   'bin_bias--b4':r'$b^{[4]}$',\
                   'bin_bias--b5':r'$b^{[5]}$',\
                   'bin_bias--b6':r'$b^{[6]}$',\
                   'bias_lens--b1':r'$b^{[1]}$',\
                   'bias_lens--b2':r'$b^{[2]}$',\
                   'bias_lens--b3':r'$b^{[3]}$',\
                   'bias_lens--b4':r'$b^{[4]}$',\
                   'bias_lens--b5':r'$b^{[5]}$',\
                   'bias_lens--b6':r'$b^{[6]}$',\
                   'bias_lens--b1wt_bin1':r'$b^{[1]}}$',\
                   'bias_lens--b1wt_bin2':r'$b^{[2]}$',\
                   'bias_lens--b1wt_bin3':r'$b^{[3]}$',\
                   'bias_lens--b1wt_bin4':r'$b^{[4]}$',\
                   'bias_lens--b1wt_bin5':r'$b^{[5]}$',\
                   'bias_lens--b1wt_bin6':r'$b^{[6]}$',\
                   'bias_lens--b1e_bin1':r'$b_1^{[1]}$',\
                   'bias_lens--b1e_bin2':r'$b_1^{[2]}$',\
                   'bias_lens--b1e_bin3':r'$b_1^{[3]}$',\
                   'bias_lens--b1e_bin4':r'$b_1^{[4]}$',\
                   'bias_lens--b1e_bin5':r'$b_1^{[5]}$',\
                   'bias_lens--b1e_bin6':r'$b_1^{[6]}$',\
                   'bias_lens--b2e_bin1':r'$b_2^{[1]}$',\
                   'bias_lens--b2e_bin2':r'$b_2^{[2]}$',\
                   'bias_lens--b2e_bin3':r'$b_2^{[3]}$',\
                   'bias_lens--b2e_bin4':r'$b_2^{[4]}$',\
                   'bias_lens--b2e_bin5':r'$b_2^{[5]}$',\
                   'bias_lens--b2e_bin6':r'$b_2^{[6]}$',\
                   'bias_lens--b1e_sig8_w_bin1':r'$b_1^{[1]}\sigma_8 w^{-0.25}$',\
                   'bias_lens--b1e_sig8_bin1':r'$b_1^{[1]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin2':r'$b_1^{[2]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin3':r'$b_1^{[3]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin4':r'$b_1^{[4]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin5':r'$b_1^{[5]}\sigma_8$',\
                   'bias_lens--b1e_sig8_bin6':r'$b_1^{[6]}\sigma_8$',\
                   'bias_lens--b2e_sig8sq_bin1':r'$b_2^{[1]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin2':r'$b_2^{[2]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin3':r'$b_2^{[3]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin4':r'$b_2^{[4]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin5':r'$b_2^{[5]}\sigma_8^2$',\
                   'bias_lens--b2e_sig8sq_bin6':r'$b_2^{[6]}\sigma_8^2$',\
                   'bias_lens--rmean_bin':r'$X_{\rm lens}$',\
                   'shear_calibration_parameters--m1':r'$m_1$',\
                   'shear_calibration_parameters--m2':r'$m_2$',\
                   'shear_calibration_parameters--m3':r'$m_3$',\
                   'shear_calibration_parameters--m4':r'$m_4$',\
                   'shear_calibration_parameters--m5':r'$m_5$',\
                   'intrinsic_alignment_parameters--a':R'$A_{\rm IA}$',\
                   'intrinsic_alignment_parameters--alpha':R'$\alpha_{\rm IA}$',\
                   'wl_photoz_errors--bias_1':r'$\Delta z_{s}^1$',\
                   'wl_photoz_errors--bias_2':r'$\Delta z_{s}^2$',\
                   'wl_photoz_errors--bias_3':r'$\Delta z_{s}^3$',\
                   'wl_photoz_errors--bias_4':r'$\Delta z_{s}^4$',\
                   'wl_photoz_errors--bias_5':r'$\Delta z_{s}^5$',\
                   'wl_photoz_errors--bias_6':r'$\Delta z_{s}^6$',\
                   'lens_photoz_errors--bias_1':r'$\Delta z_{l}^1$',\
                   'lens_photoz_errors--bias_2':r'$\Delta z_{l}^2$',\
                   'lens_photoz_errors--bias_3':r'$\Delta z_{l}^3$',\
                   'lens_photoz_errors--bias_4':r'$\Delta z_{l}^4$',\
                   'lens_photoz_errors--bias_5':r'$\Delta z_{l}^5$',\
                   'lens_photoz_errors--bias_6':r'$\Delta z_{l}^6$',\
                   'lens_photoz_errors--width_1':r'$W_{l}^1$',\
                   'lens_photoz_errors--width_2':r'$W_{l}^2$',\
                   'lens_photoz_errors--width_3':r'$W_{l}^3$',\
                   'lens_photoz_errors--width_4':r'$W_{l}^4$',\
                   'lens_photoz_errors--width_5':r'$W_{l}^5$',\
                   'lens_photoz_errors--width_6':r'$W_{l}^6$',\
                   'intrinsic_alignment_parameters--a':r'$a_{\rm IA}$',\
                   'intrinsic_alignment_parameters--alpha':r'$\alpha_{\rm IA}$',\
                   'intrinsic_alignment_parameters--z_piv':r'$z_{\rm piv}^{\rm IA}$',\
                   'intrinsic_alignment_parameters--a1':r'$A_1^{\rm IA}$',\
                   'intrinsic_alignment_parameters--a2':r'$A_2^{\rm IA}$',\
                   'intrinsic_alignment_parameters--adel':r'$A_{\rm del}^{\rm IA}$',\
                   'intrinsic_alignment_parameters--alpha1':r'$\alpha_1^{\rm IA}$',\
                   'intrinsic_alignment_parameters--alpha2':r'$\alpha_2^{\rm IA}$',\
                   'intrinsic_alignment_parameters--alphadel':r'$\alpha_{\rm del}^{\rm IA}$',\
                   'intrinsic_alignment_parameters--bias_ta':r'$b_{\rm ta}$',\
                   'intrinsic_alignment_parameters--bias_tt':r'$b_{\rm tt}$',\
                   'mag_alpha_lens--alpha_1':r'$\alpha_1$',\
                   'mag_alpha_lens--alpha_2':r'$\alpha_2$',\
                   'mag_alpha_lens--alpha_3':r'$\alpha_3$',\
                   'mag_alpha_lens--alpha_4':r'$\alpha_4$',\
                   'mag_alpha_lens--alpha_5':r'$\alpha_5$',\
                   'mag_alpha_lens--alpha_6':r'$\alpha_6$',\

                   'delta_neff':r'$\Delta N_{\rm eff}$',\
                   'ranks--rank_hyperparm_1':r'$\mathcal{H}_1$',\
                   'ranks--rank_hyperparm_2':r'$\mathcal{H}_2$',\
                   'ranks--rank_hyperparm_3':r'$\mathcal{H}_3$',\
                   #'RANKS--REALISATION_ID':"\mathrm{ HR~rlzn \#}",\
                   #'ranks--realisation_id':"\mathrm{ HR~rlzn \#}",\
                   'cosmological_parameters--tau':r'$\tau$',\
                   'planck--a_planck':r'$A_{\rm planck}$',\
                   'cosmological_parameters--yhe':r'$Y_{\rm He}$',
                   'xlens--xlens_all':r'$X_{\rm lens}$',
                   'shear_calibration_parameters--m1_uncorr':r'$m_1^{\rm uncor}$',\
                   'shear_calibration_parameters--m2_uncorr':r'$m_2^{\rm uncor}$',\
                   'shear_calibration_parameters--m3_uncorr':r'$m_3^{\rm uncor}$',\
                   'shear_calibration_parameters--m4_uncorr':r'$m_4^{\rm uncor}$',\
                   'source_photoz_u--u_0_uncorr':r'$u^{\rm s-uncor}_0$',\
                   'source_photoz_u--u_1_uncorr':r'$u^{\rm s-uncor}_1$',\
                   'source_photoz_u--u_2_uncorr':r'$u^{\rm s-uncor}_2$',\
                   'source_photoz_u--u_3_uncorr':r'$u^{\rm s-uncor}_3$',\
                   'source_photoz_u--u_4_uncorr':r'$u^{\rm s-uncor}_4$',\
                   'source_photoz_u--u_5_uncorr':r'$u^{\rm s-uncor}_5$',\
                   'source_photoz_u--u_6_uncorr':r'$u^{\rm s-uncor}_6$',\
                   'lens_photoz_u--u_0_0':r'$u^{\rm l}_{0,0}$',\
                   'lens_photoz_u--u_0_1':r'$u^{\rm l}_{0,1}$',\
                   'lens_photoz_u--u_0_2':r'$u^{\rm l}_{0,2}$',\
                   'lens_photoz_u--u_1_0':r'$u^{\rm l}_{1,0}$',\
                   'lens_photoz_u--u_1_1':r'$u^{\rm l}_{1,1}$',\
                   'lens_photoz_u--u_1_2':r'$u^{\rm l}_{1,2}$',\
                   'lens_photoz_u--u_2_0':r'$u^{\rm l}_{2,0}$',\
                   'lens_photoz_u--u_2_1':r'$u^{\rm l}_{2,1}$',\
                   'lens_photoz_u--u_2_2':r'$u^{\rm l}_{2,1}$',\
                   'lens_photoz_u--u_3_0':r'$u^{\rm l}_{3,0}$',\
                   'lens_photoz_u--u_3_1':r'$u^{\rm l}_{3,1}$',\
                   'lens_photoz_u--u_3_2':r'$u^{\rm l}_{3,2}$',\
                   'lens_photoz_u--u_4_0':r'$u^{\rm l}_{4,0}$',\
                   'lens_photoz_u--u_4_1':r'$u^{\rm l}_{4,1}$',\
                   'lens_photoz_u--u_4_2':r'$u^{\rm l}_{4,2}$',\
                   'lens_photoz_u--u_5_0':r'$u^{\rm l}_{5,0}$',\
                   'lens_photoz_u--u_5_1':r'$u^{\rm l}_{5,1}$',\
                   'lens_photoz_u--u_5_2':r'$u^{\rm l}_{5,2}$',\
}


# providing some parameter fid val and range dicts for convenience
# but use with caution; probably better to just read in an values ini
# fid values used for extensions, where sim DV had NLA IA model
Y3FIDVALDICT = {
    'cosmological_parameters--w':-1. ,\
    'cosmological_parameters--wa':0.0,\
    'cosmological_parameters--wp':-1. ,\
    'cosmological_parameters--omega_k':0.0,\
    'modified_gravity--sigma0':0.0,\
    'modified_gravity--mu0':0.0,\
    'cosmological_parameters--h0':.69,\
    'cosmological_parameters--n_s': .97,\
    'COSMOLOGICAL_PARAMETERS--SIGMA_8':0.82594,\
    'cosmological_parameters--sigma_8':0.82594,\
    'S8':0.82594,\
    'cosmological_parameters--a_s':2.19e-9,\
    'cosmological_parameters--tau':0.08,\
    'asx1.e9':2.19,\
    'cosmological_parameters--alens':1,\
    'cosmological_parameters--omega_b':0.048,\
    'cosmological_parameters--omega_m':0.3, \
    'cosmological_parameters--omnuh2':0.00083 ,\
    'cosmological_parameters--ommh2':.69*.69*.3,\
    'cosmological_parameters--ombh2': .69*.69*0.048,\
    'cosmological_parameters--log10_fr0':-6.0,\
    'summnu':0.00083*OMNUH2_TO_MNU_FACTOR ,\
    'intrinsic_alignment_parameters--z_piv' : 0.62,\
    'intrinsic_alignment_parameters--a1' : 0.7,\
    'intrinsic_alignment_parameters--a2' : 0. ,\
    'intrinsic_alignment_parameters--alpha1': -1.7,\
    'intrinsic_alignment_parameters--alpha2': 0.,\
    'intrinsic_alignment_parameters--bias_ta': 0.,\
    'shear_calibration_parameters--m1':0.,\
    'shear_calibration_parameters--m2':0.,\
    'shear_calibration_parameters--m3':0.,\
    'shear_calibration_parameters--m4':0.,\
    'wl_photoz_errors--bias_1':0.,\
    'wl_photoz_errors--bias_2':0.,\
    'wl_photoz_errors--bias_3':0.,\
    'wl_photoz_errors--bias_4':0.,\
    'lens_photoz_errors--bias_1':0.,\
    'lens_photoz_errors--bias_2':0.,\
    'lens_photoz_errors--bias_3':0.,\
    'lens_photoz_errors--bias_4':0.,\
    'lens_photoz_errors--bias_5':0.,\
    'lens_photoz_errors--bias_6':0.,\
    'lens_photoz_errors--width_1':1.,\
    'lens_photoz_errors--width_2':1.,\
    'lens_photoz_errors--width_3':1.,\
    'lens_photoz_errors--width_4':1.,\
    'lens_photoz_errors--width_5':1.,\
    'lens_photoz_errors--width_6':1.,\
    'mag_alpha_lens--alpha_1' : 1.21,\
    'mag_alpha_lens--alpha_2' : 1.15,\
    'mag_alpha_lens--alpha_3' : 1.88,\
    'mag_alpha_lens--alpha_4' : 1.97,\
    'mag_alpha_lens--alpha_5' : 1.78,\
    'mag_alpha_lens--alpha_6' : 2.48,\
    'bias_lens--b1': 1.5,\
    'bias_lens--b2': 1.8,\
    'bias_lens--b3': 1.8,\
    'bias_lens--b4': 1.9,\
    'bias_lens--b5': 2.3,\
    'bias_lens--b6': 2.3,\
    'bias_lens--b1wt_bin1': 1.5,\
    'bias_lens--b1wt_bin2': 1.8,\
    'bias_lens--b1wt_bin3': 1.8,\
    'bias_lens--b1wt_bin4': 1.9,\
    'bias_lens--b1wt_bin5': 2.3,\
    'bias_lens--b1wt_bin6': 2.3,\
    'bias_lens--rmean_bin':1.0,\
    'ranks--rank_hyperparm_1':0.5,\
    'ranks--rank_hyperparm_2':0.5,\
    'ranks--rank_hyperparm_3':0.5,\
    'npg_parameters--a1':1.,\
    'npg_parameters--a2':1.,\
    'npg_parameters--a3':1.,\
    'npg_parameters--a4':1.,\
    'npg_parameters--a5':1.,\
    'npg_parameters--a_cmb':1.,\
    'supernova_params--m':-19.,\
    'modified_gravity--b0':0.1,\
    'cosmological_parameters--neff':3.046,\
    'cosmological_parameters--meffsterile':0.0,\
    'cosmological_parameters--delta_neff':0.0,\
    }

# These were used for extensions, which used NLA model
Y3RANGES = {'cosmological_parameters--h0':[.55,.91],\
                    'cosmological_parameters--n_s':[.87,1.07] ,\
                    'bin_bias--b4':[0.8,3.0],\
                    'bin_bias--b1':[0.8,3.0], \
                    'bin_bias--b2':[0.8,3.0],\
                    'bin_bias--b3':[0.8,3.0],\
                    'bias_lens--b1':[0.8,3.0],\
                    'bias_lens--b2':[0.8,3.0],\
                    'bias_lens--b3':[0.8,3.0],\
                    'bias_lens--b4':[0.8,3.0],\
                    'bias_lens--b1wt_bin1':[0.8,3.0],\
                    'bias_lens--b1wt_bin2':[0.8,3.0],\
                    'bias_lens--b1wt_bin3':[0.8,3.0],\
                    'bias_lens--b1wt_bin4':[0.8,3.0],\
                    'bias_lens--rmean_bin':[0.6,1.4],\
                    'shear_calibration_parameters--m1':[-.1,.1],\
                    'shear_calibration_parameters--m2':[-.1,.1],\
                    'shear_calibration_parameters--m3':[-.1,.1],\
                    'shear_calibration_parameters--m4':[-.1,.1],\
                    'COSMOLOGICAL_PARAMETERS--SIGMA_8':[.5,1.],\
                    'S8':[None,None],\
                    'cosmological_parameters--a_s':[.5e-9,5.e-9],\
                    'cosmological_parameters--omega_b':[0.03,0.07],\
                    'cosmological_parameters--omega_m':[.1,.9], \
                    'cosmological_parameters--w':[-3.,-.3333], \
                    'cosmological_parameters--tau':[0.01,0.8], \
                    'cosmological_parameters--log10_fr0':[-8.,-2.],\
                    'supernova_params--m':[-20,-18.], \
                    'intrinsic_alignment_parameters--a':[-5.,5.],\
                    'intrinsic_alignment_parameters--alpha':[-5.,5.],\
                    'intrinsic_alignment_parameters--a1' : [-5.,5.],\
                    'intrinsic_alignment_parameters--a2' : [-5.,5.],\
                    'intrinsic_alignment_parameters--alpha1': [-5.,5.],\
                    'intrinsic_alignment_parameters--alpha2': [-5.,5.],\
                    'intrinsic_alignment_parameters--bias_ta': [0.,2],\
                    'cosmological_parameters--omnuh2':[.0006,  0.00644],\
                    'summnu':[.0006*OMNUH2_TO_MNU_FACTOR ,0.00644*OMNUH2_TO_MNU_FACTOR ],\
                    'wl_photoz_errors--bias_1':[-.1,.1],\
                    'wl_photoz_errors--bias_2':[-.1,.1],\
                    'wl_photoz_errors--bias_3':[-.1,.1],\
                    'wl_photoz_errors--bias_4':[-.1,.1],\
                    'lens_photoz_errors--bias_1':[-.05,.05],\
                    'lens_photoz_errors--bias_2':[-.05,.05],\
                    'lens_photoz_errors--bias_3':[-.05,.05],\
                    'lens_photoz_errors--bias_4':[-.05,.05],\
                    'lens_photoz_errors--width_1':[0.1,1.9],\
                    'lens_photoz_errors--width_2':[0.1,1.9],\
                    'lens_photoz_errors--width_3':[0.1,1.9],\
                    'lens_photoz_errors--width_4':[0.1,1.9],\
                    'cosmological_parameters--alens':[0.,10],\
}
#=================================================================
# Functions for handling filenames and labels
#=================================================================

def get_sampler_from_chain(chainfile):
    '''Get the sampler used in the chain.
    It assumes the second line of a chain always starts with #sampler=
    and it reads the following string.
    '''
    with open(chainfile, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 1:
                sampler = line.strip()
            elif i > 1:
                break
    return sampler[9:]

def get_param_label(p,texdict = DEFAULT_PLABELS,forgetdist=True):
    if p in texdict.keys():
        label= texdict[p]
    else:
        label= p
    if forgetdist: #get dist adds text formatting automatically
        outlabel =  label.replace('$','')
    else:
        outlabel = label
    return outlabel

def get_label(m,mdict):
    if m in mdict.keys():
        return mdict[m]
    else:
        return m

def get_fidval(p,valdict):
    """
    This will be used to get values to center ellipses on.
    If we haven't set up a parameter's fiducial value in the
    dictionary, just center on zero.
    """
    if p in valdict.keys():
        return valdict[p]
    else:
        return 0.


def get_pscaling(p):
    """
    for certain parameters (mostly a_s), to handle plotting, need to rescale
    """
    if p=='cosmological_parameters--a_s':
        return 1.e9
    else:
        return 1.

def parse_truthvals(truthvals,pkeys):
    """
    Given list of parameter keys inpkeys, and truthvals, which is either string or dictionary, return list of defaults (nan if not in dictionary)


    """
    outtruth = []

    for k in pkeys:
        if k in truthvals.keys():
            outtruth.append(truthvals[k])
        else:
            outtruth.append(np.nan)
    return outtruth

#==================================================================
# Functions for reading in values files, generating fid and range dicts
#==================================================================
def get_bashvars_fromshfile(shfile,force_testsampler=True,callfrom='.'):
    """
    given sh file where environment variables for extensions chains
    are set, reads that file and sets environment variables in a way
    this script can use.

    Will only parse lines in the sh file that start with either
    export or %include.
    """
    if callfrom not in shfile:
        f = open(os.path.join(callfrom,shfile),'r')
    else:
        f = open(shfile,'r')
    for line in f:
        line = os.path.expandvars(line)
        if line.startswith('export'):
            # get rid of white space, export, everything after commment character
            cleaned = line.replace('export','').replace('"','').split("#")[0].strip()
            key,value = cleaned.split("=")
            if key=='SAMPLER' and force_testsampler:
                os.environ[key]='test'
            else:
                os.environ[key]=value

        if line.startswith('source'):
            # read in another file to set more variables
            fname = line.split()[1]
            get_bashvars_fromshfile(fname,force_testsampler)
    f.close()
    return 0


def get_bashvars_fromshfile(shfile,force_testsampler=True,callfrom='.'):
    """
    given sh file where environment variables for extensions chains
    are set, reads that file and sets environment variables in a way
    this script can use.

    Will only parse lines in the sh file that start with either
    export or %include.
    """
    if callfrom not in shfile:
        f = open(os.path.join(callfrom,shfile),'r')
    else:
        f = open(shfile,'r')
    for line in f:
        line = os.path.expandvars(line)
        if line.startswith('export'):
            # get rid of white space, export, everything after commment character
            cleaned = line.replace('export','').replace('"','').split("#")[0].strip()
            key,value = cleaned.split("=")
            if key=='SAMPLER' and force_testsampler:
                os.environ[key]='test'
            else:
                os.environ[key]=value

        if line.startswith('source'):
            # read in another file to set more variables
            fname = line.split()[1]
            get_bashvars_fromshfile(fname,force_testsampler)
    f.close()
    return 0

def get_valuesini_from_paramsini(paramsini,callfrom=default_callfrom, bashvarf = None):
    """
    Check whether params vile sets values file. If it does, return values name.
    Note that this function won't dig through
    """
    if bashvarf is not None:
        # get bash variables set up
        get_bashvars_fromshfile(bashvarf)

    inpipelinesec =False
    valuesf = None
    #print("opening:",paramsini)
    f = open(paramsini)
    #print(' success')
    for line in f:
        line = line.strip()
        if not line: # empty line
            continue
        line = os.path.expandvars(line)
        if line.startswith('%include'): #include line
            incfile = os.path.join(callfrom,line.replace('%include ',''))
            # call recursively include that other ini, but don't reset bash vars
            tempvaluesf = get_valuesini_from_paramsini(incfile,callfrom)
            if tempvaluesf is not None: # overwriting previous value
                valuesf = tempvaluesf
        elif line.startswith(';') or line.startswith('#'): #comment
            continue
        elif line.startswith('['): #section header
            section = line.replace('[','').replace(']','')
            inpipelinesec = section=='pipeline'
            #print( 'section=',section, inpipelinesec)
        elif not inpipelinesec: # variable, but not in [pipeline]
                continue
        else: #variable in [pipeline]
            items = line.replace('=','').split()
            var = items[0]
            #print('  >',items,var,var=='values')
            if var=='values':
                valuesf = items[1]
                #print(" >> valuesf=",valuesf)

    f.close()
    return valuesf

def read_values_ini(valuesini,callfrom=default_callfrom, vardict={}, bashvarf = None):
    """
    Given path to values file ini, reads in that values file and relevant %includes
    and return parameter info in dictionary

    vardict - dictionary containing parameter info
    valuesini - path of values ini file we want to read
    callfrom - string path that cosmosis will be called from, used to find %included files
    bashvarf - name of sh job file defining bash variables that might be used in ini
    """
    if bashvarf is not None:
        # get bash variables set up
        if not os.path.isfile(bashvarf):
            print("Can't find bash file, skipping:",bashvarf)
            return None

        get_bashvars_fromshfile(bashvarf)

    if not os.path.isfile(valuesini):
        print("Can't find values.ini file, skipping:",valuesini)
        return None
    # keys will be (section, paramname)
    # values will be lists of 3 numbers; second and third will be nan if param fixed
    section=None
    #print(">>>>>>READING",valuesini)
    f = open(valuesini)
    for line in f:
        line = line.strip()
        if not line: # empty line
            continue
        line = os.path.expandvars(line)
        if line.startswith('%include'): #include line
            incfile = os.path.join(callfrom,line.replace('%include ',''))
            # call recursively include that other ini, but don't reset bash vars
            read_values_ini(incfile,callfrom, vardict, None)
        elif line.startswith(';') or line.startswith('#'): #comment
            continue
        elif line.startswith('['): #section header
            section = line.replace('[','').replace(']','')
            # cosmosis isn't case sensitive, switch everything to lowercase
            section = section.lower()
        else: # variable!
            # remove any comment stuff
            line = line.split(';')[0]
            line = line.split('#')[0]
            # now get the info
            items = line.replace('=','').split()
            var = items[0].lower()
            valsin = [float(x) for x in items[1:]]
            nvals = len(valsin)
            vals = np.nan*np.ones(3)
            vals[:nvals] = np.array(valsin)
            #print('    ',var,valsin,nvals,vals)
            vardict[(section,var)] = vals

    f.close()
    #for k in vardict.keys():
    #    print('    ',k,vardict[k])

    return vardict
#--------------------------------------------------
def get_fidvals_and_ranges_from_values_ini(valuesini,callfrom=default_callfrom, bashvarf = None):
    vardict = read_values_ini(valuesini, callfrom, bashvarf =bashvarf)
    fidvals = {}
    ranges = {}
    #print('>>>>>>>>>>>')
    for key in vardict.keys():
        sec = key[0]
        var = key[1]
        vals = vardict[key]
        savekey = sec+'--'+var
        if np.isnan(vals[1]): #fixed param, just one number
            fidvals[savekey] = vals[0]
        else:
            fidvals[savekey] = vals[1]
            ranges[savekey] = [vals[0],vals[2]]

        #print('    ',savekey,vals,vals[1],np.isnan(vals[1]),fidvals[savekey])
    return fidvals, ranges


#==================================================================
# Functions for reading in chains
#==================================================================

def get_chain_fname(runname,datadir = '.',suffix = DEFAULT_FILE_SUFFIX,prefix=DEFAULT_FILE_PREFIX,fname=None,alt4file = {},infitsfile=None, scalecutfile = None, joinchar = '.',  modelsubdirs = False, y3style=True):
    """
    If you pass in fname, will look in [datadir]/fname for chain data.
    If fname is None (default), will use datastr, modelstr, suffix and prefix to
    construct filename.

    if y3style == False:
      filename is datadir/prefix_runname_suffix (underscores adjusted if suffix is just .txt)
    if y3stulle == True:
      datadir/prefix_infitsfile.scalecutfile.runname.txt (.txt as chain suffix), periods = joinchar

    returns string name of file containin chain data

    For some cases, a run name might need to look in a different  file;
      -- e.g. for BAO ggsplit, there are only geometric params, so
         the model b_l is equivalent to b_l-sOm, so when this function gets b_l-sOm it just load b_l
    To handle this, you should pass a dict alt4file which has these run name translations

    """

    if fname is not None:
        return os.path.join(datadir,fname)


    if modelsubdirs: #separate subdis for l, w, ok, base models
        model = runname.split('_')[1]
        mbase = model.split('-')[0]
        lookindir = os.path.join(datadir,mbase)
    else:
        lookindir = datadir


    if y3style:
        if runname in alt4file.keys():
            userun=alt4file[runname]
        else:
            userun = runname
        prefixstr = ''
        suffixstr = ''
        if prefix and (prefix[-1] not in ('_','-')):
            prefixstr = prefix+'_'
        else:
            prefixstr = prefix
        if suffix and ((suffix!='.txt') and (suffix[0] not in ('_','-'))):
            suffixstr = '_'+suffix
        else:
            suffixstr = suffix

        fnamebase = lookindir +prefixstr+joinchar.join([infitsfile,scalecutfile,runname])+suffixstr
        fname = os.path.join(lookindir,fnamebase)
        return fname

    else:
        if runname in alt4file.keys():
            userun=alt4file[runname]
        else:
            userun = runname
        prefixstr = ''
        suffixstr = ''
        if prefix:
            prefixstr = prefix+'_'
        if suffix and (suffix!='.txt'):
            suffixstr = '_'+suffix
        else:
            suffixstr = suffix
        fnamebase = ''.join([prefixstr,userun,suffixstr])
        return os.path.join(datadir,fnamebase)


#--------------------------------------------------
def get_nsample(filename):
    """
    For multinest or polychord files, the last line tells you how many lines
    to keep. This function pulls that info from th file.
    """
    nsamples=None
    fi =  open(filename,"r")
    for ln in fi:
        if (ln.startswith("nsample=")) or (ln.startswith('#nsample=')):
            nsamples = int(ln.replace('nsample=','').replace('#',''))
            break
    fi.close()
    return nsamples

#--------------------------------------------------
def get_nlive(filename,useboost=False):
    """
    for mn or pc files, get live points out of file header

    if useboost==True, will also look for the boosted_posterior setting,
    will return Nlive_boosted=Nlive*(1+boost_posterior)
    """

    if not os.path.isfile(filename):
        return None
    fi =  open(filename,"r")
    # use "#live_points=" rather than "## live_points ="
    # the latter might be there just from a pc module def
    # single comment and no spaces means what was actually used to run

    lines = fi.readlines()

    nlive = None
    boost=None
    for ln in lines:
        if ln.startswith("#live_points="):
            nlive = int(ln.replace("#live_points=",''))
            if (not useboost) or (boost is not None):
                # we have everything we need
                break
        elif ln.startswith("#boost_posteriors="):
            boost = float(ln.replace("#boost_posteriors=",""))
            if nlive is not None: # have both pieces of info
                break
    fi.close()
    print("nlive=",nlive,useboost,'boost',boost)
    if useboost:
        return nlive*(1.+boost)
    else:
        return nlive
#--------------------------------------------------
def get_logz(filename):
    lastone=False
    fi =  open(filename,"r")
    for ln in fi:
        if lastone:
            logz_error= float(ln.replace('#log_z_error=',''))
            break
        if ln.startswith("#log_z="):
            logz = float(ln.replace('#log_z=',''))
            lastone=True
    fi.close()
    return logz,logz_error
#--------------------------------------------------
def get_paramdict(filename,getlist=False):
    """
    Reads in first line of chain file to make dictionary
    to translate parameter names to column indices.
    """
    #print("Getting parameter->index dictionary from ",filename)
    f = open(filename,'r')
    firstline = f.readline().split()
    #print(firstline)
    f.close()
    pdict = {}
    plist = []
    userightmost=['weight','post','like','prior', 'old_weight','old_post', 'log_weight']
    N_log_weight=0 # for some chains IS run on nautilus, will have two log_weight columsn
    #  want to store leftmost as old_log_weight, rightmost as log_weight
    for i in range(len(firstline)):
        s = firstline[i].replace('#','')
        if s=='log_weight':
            N_log_weight+=1
            if N_log_weight>1:# there are two of these!
                # set up existing entries to be old_log_weight instead of log_weight
                plist[pdict[s]] = 'old_log_weight'
                pdict['old_log_weight']  = pdict[s]
                pdict.pop(s)
                # then continue as usual
        
        plist.append(s)

        if s.lower() not in pdict.keys():
            # if we ran with IS and a parameter is saved as extra output
            # for both original and IS run, it'll appear twice.
            # we want to use the leftmost one (that's the one that uses the IS calcs
            # the original extra output and samplter output goes to the right of that
            # thanks to Noah and Otavio for catching this
            pdict[s] = i
            pdict[s.lower()]=i
        elif s in userightmost:
            # for specific list of columns, use rightmost
            # for sampled and derived params, want to use leftmost version
            # but for sampler output want rightmost
            # (there will only be one copy though so this doesn't really matter)
            pdict[s] = i
            pdict[s.lower()]=i
    if getlist:
        return pdict, plist
    else:
        return pdict
#--------------------------------------------------
def get_ranges_from_chainheader(filename,chaindir=''):
    """
    Read through cosmosis chain header to get hard prior boundaries
    so that getdist can correct for these.
    """
    ranges = {}

    try:
        f =  open(os.path.join(chaindir,filename),'r')
    except:
        return None
    invalues=False
    for line in f:
        if 'END_OF_VALUES_INI' in line:
            invalues = False
            # done reading values, stop reading the file
            break
        if 'START_OF_VALUES_INI' in line:
            invalues=True
            # we're entering the part of the header with values info
            continue
        if invalues:
            line = line.replace('##','').strip()
            if not line: #empty line
                continue
            elif line.startswith('['): # section header
                valsect = line.replace(']','').replace('[','')
            else:
                linesplit = line.replace('=','').split()
                if len(linesplit)==4:
                    paramkey = valsect+'--'+linesplit[0]
                    minval = float(linesplit[1])
                    maxval = float(linesplit[3])
                    ranges[paramkey]=(minval,maxval)
                else:
                    # if there are less than 4 entries (name bound start bound)
                    # parameter is fixed, don't count it
                    continue

    f.close()

    return ranges

#--------------------------------------------------
def rangedict_from_gdchain(gdchain):
    if gdchain is None:
        return None
    bounds = gdchain.ranges
    rangedict = {}

    for name in bounds.names:
        lowerval = bounds.getLower(name)
        upperval = bounds.getUpper(name)
        rangedict[name]=(lowerval,upperval)

    return rangedict

#--------------------------------------------------
def get_Nparams(filename):
    """
    Given chain file, read through header to get number of parameters sampled over
    """
    params = []
    f = open(filename,'r')
    invalues=False
    for line in f:
        if 'END_OF_VALUES_INI' in line:
            invalues=False
            break
        if 'START_OF_VALUES_INI' in line:
            invalues=True
            continue
        if invalues:
            line = line.replace('##','').strip()
            if line.startswith('[') or not line:
                continue
            else:
                linesplit = line.split()
                if len(linesplit)==5:
                    params.append(linesplit[0])
                else:
                    # if there are less than 5 entries (name = bound start bound)
                    # parameter is fixed, don't count it
                    continue

    f.close()

    return len(params),params
#--------------------------------------------------
def prep_chain(chainfname,chainlabel,kdesmooth=-1., paramlabels = DEFAULT_PLABELS, rangedict=None, chaindir='',params=None , ignoreisweights=False):
    """
    Read in chain, add some derived parameters that we're likely to want
    May want to add/remove things here
    Note that these add_x() functions will just do nothing if necessary
        sampled parameters aren't in the chain

    If rangedict is None, will find hard prior boundaries in chain header
    and automatically pass them to getdist (recommended)

    if params is empty, reads in all prameters in the input file

    ignoreisweights - to be used rarely, mostly for checking PPD IS chains
         where model used to save extra DV output isn't what we want to
         use for the posterior; if True, just uses old weight without IS scaling
    """
    if rangedict is None:
         rangedict = get_ranges_from_chainheader(chainfname,chaindir)
    gdchain = get_gdchain(chainfname,flabel=chainlabel,params=params,\
                          kdesmooth=kdesmooth ,indatdir = chaindir,\
                          paramlabels = paramlabels, rangedict = rangedict,\
                          ignoreisweights=ignoreisweights)
    #print('>>>',chainfname,gdchain)
    if gdchain is not None:
        add_S8(gdchain)
        add_W8(gdchain)
        add_As_scaled(gdchain)
        add_physical_densities(gdchain)
        add_wp(gdchain) #if we have w0 and wa, will compute wp
        add_omega_c(gdchain)
        add_mnu(gdchain)
        add_bias_marg_w_params(gdchain)
        add_bias_marg_params(gdchain)
        add_omega_L(gdchain)

    #print(chainfname)
    #print([n.name for n in gdchain.getParamNames().names])
    return gdchain

#--------------------------------------------------
def get_gdchain(inputf,flabel=None,params=None,\
                kdesmooth=-1.,indatdir = '',paramlabels = DEFAULT_PLABELS,\
                rangedict = None, ignoreisweights=False):
    """
    given cosmosis output chain filename, read in chain
    data and return  getdist MCSamples objects, one per file.

    if params is empty, reads in all parameters in the input file.
    Otherwise just stores just the parameters in that list.

    rangedict - hard prior boundaries used by getdist to
                correct estimated posterior for these bounds.
                if None, will automatically get this from the chain header
                (recommend leaving as None default)

    ignoreisweights - to be used rarely, mostly for checking PPD IS chains
         where model used to save extra DV output isn't what we want to
         use for the posterior; if True, just uses old weight without IS scaling
    """
    #all of these will have one entry per input file
    data = []
    nsamples = []
    pdicts = []
    paramindices = [] #each entry is a list of indces
    paramnames = [] #each entry is a list of strings

    if flabel is None: #label for each file (shows up in legend)
        flabel = inputf

    infile = os.path.join(indatdir,inputf)
    #print('>>>>>infile',infile)
    if not os.path.isfile(infile):
        #print("NO FILE",infile)
        return None
    elif os.stat(infile).st_size == 0:
        #print("EMPTY FILE",infile)
        return None

    print('...getting data from ',inputf)

    # figure out if sampler is nautilus
    isNautilus='nautilus' in get_sampler_from_chain(infile)

    try:
        dataf = np.loadtxt(infile)
        pdictf = get_paramdict(infile)
        #print(">>",pdictf.keys())
        if rangedict is None:
            rangedict = get_ranges_from_chainheader(infile)

        if 'old_post' in pdictf.keys():
            isIS = True # this is importance sampling ouutput
            nsampf = dataf.shape[0]
            if ignoreisweights:
                isnested=False #not fully logical here but works for something i need it for
            else:
                isnested=True
        else:
            isIS = False
            isnested = 'weight' in pdictf.keys() or 'log_weight' in pdictf.keys()

            if isnested: #assuming unnested chains have already been cut for burnin
                if 'weight' in pdictf.keys():
                    windf=pdictf['weight']
                elif 'log_weight' in pdictf.keys():
                     windf=pdictf['log_weight']
                weights = dataf[:,windf]
                if np.all(weights==weights[0]): #equal weight samples
                    print("  All weights are equal. treating as non-nested")
                    isnested=False
                else:
                    nsampf = get_nsample(infile)
                    if nsampf is None:
                        return None

    except:
        print("FILE EXISTS BUT SOMETHING WENT WRONG FOR:",infile)
        return None

    pkeysf = []
    pindf = []
    pnamef = []
    needweight=False

    if isIS:
        if 'weight' not in pdictf.keys(): # need to computed based on IS output
            needweight=True
            if 'old_weight' in pdictf.keys(): #original chain had weight column
                windf_old = pdictf['old_weight']
                oldw_was_log = False
            else:
                windf_old = pdictf['old_log_weight']
                oldw_was_log = True
            windf_is = pdictf['log_weight']
        else:
            windf=pdictf['weight']
    else:
        if isnested:
            if isNautilus:
                windf =  pdictf['log_weight']
            else:
                windf =  pdictf['weight']
    postindf = pdictf['post']

    if params is None:
        #read in all params in file
        for k in pdictf.keys():
            if k not in pkeysf:
                pindf.append(pdictf[k])
                pkeysf.append(k)
                pnamef.append(get_param_label(k,texdict=paramlabels))
    else:
        for key in params:
            try: #see if key is in this file
                indfk = pdictf[key]
            except:
                print("     ",key,'not in',infile, key in pdictf.keys())
                pass #if it's not, move on

            else: #if it is, et that info
                pindf.append(indfk)
                pkeysf.append(key)
                pnamef.append(get_param_label(key,texdict=paramlabels))
    if isnested:
        sampledat = dataf[-nsampf:,pindf]
    else:
        sampledat = dataf[:,pindf]

    print("ISNESTED",isnested,sampledat.shape, dataf.shape)

    print('...adding chain for',flabel,', samples have shape',sampledat.shape)

    if isIS and needweight:
        # computing IS weights; mask used to avoid infs
        if oldw_was_log:
            oldweights = np.exp(dataf[-nsampf:,windf_old])
            nonzeromask = np.ones_like(oldweights,dtype=bool)

        else:
            oldweights = dataf[-nsampf:,windf_old]
            nonzeromask = oldweights!=0
            if  np.all(oldweights==0.):
                gdchain = None
        if ignoreisweights:
            print("  IGNORING IS WEIGHTS")
            weights = oldweights
        else:
            isweights = dataf[-nsampf:,windf_is]
            wexp =np.exp(isweights[nonzeromask])
            weights = np.zeros(oldweights.size)
            weights[nonzeromask] = oldweights[nonzeromask]*wexp

    else:
        print("isIS=False")
        print("isNautilus=",isNautilus)
        if isnested:
            weights = dataf[-nsampf:,windf]
            if isNautilus:
                print("Using Nautilus sampler")
                # remove infinite posterior values
                weights = np.exp(weights)
                inf_ii = []
                for i in range(len(weights)):
                    inf_ii.append(math.isinf(weights[i]))
                weights[inf_ii] = 0
    if np.all(weights==weights[0]): #equal weight samples
        print("  All weights are equal. treating as non-nested")
        isnested=False

    if isnested:
        gdchain = MCSamples(samples = sampledat,weights= weights,loglikes=dataf[-nsampf:,postindf],names=pkeysf,labels=pnamef,name_tag = flabel,ranges = rangedict,settings={'smooth_scale_1D':kdesmooth,'smooth_scale_2D':kdesmooth},sampler='nested')
    else:
        
        gdchain = MCSamples(samples = sampledat,loglikes=dataf[:,postindf],names=pkeysf,labels=pnamef,name_tag = flabel,ranges = rangedict,settings={'smooth_scale_1D':kdesmooth,'smooth_scale_2D':kdesmooth},sampler='mcmc')

    # note that if you to check the prior ranges used to
    # read in this file, you can use rangedict_from_gdchain

    return gdchain

#--------------------------------------------------
def add_mnu(gdchains):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds the derived parameter sum mnu = omnuh2*93.14
    """
    newparamname = 'summnu'
    newlabel = r'\sum m_{{\nu}}\,[{{\rm eV}}]'
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains

    for ch in gdchainlist:
        if ch is None:
            continue

        #don't do anything if derived param is already there
        ind = ch.paramNames.numberOfName(newparamname)
        if ind!=-1:
            continue


        rangedict = rangedict_from_gdchain(ch)
        omnuh2ind = ch.paramNames.numberOfName('mnu_parameters--omnuh2')
        if omnuh2ind ==-1:
            logomnuh2ind = ch.paramNames.numberOfName('mnu_parameters--log_omnuh2')
        if omnuh2ind ==-1 and logomnuh2ind==-1: # see if its in cosm p arams
            omnuh2ind = ch.paramNames.numberOfName('cosmological_parameters--omnuh2')

        if omnuh2ind !=-1 or logomnuh2ind !=-1:        #don't compute if neccesary numbers not there
            if omnuh2ind !=-1:
                omnuh2 = ch.samples[:,omnuh2ind]
            else:
                log_omnuh2 = ch.samples[:,logomnuh2ind]
                omnuh2 = 10**log_omnuh2
            summnu = omnuh2*OMNUH2_TO_MNU_FACTOR
            omnuh2range = rangedict['cosmological_parameters--omnuh2']
            newparammin = omnuh2range[0]*OMNUH2_TO_MNU_FACTOR
            newparammax = omnuh2range[1]*OMNUH2_TO_MNU_FACTOR

            #print('...adding derived parameters:',newkey,'to',ch.name_tag)
            ch.addDerived(summnu,name=newparamname,label = newlabel,range=[newparammin,newparammax])


    # for ch in gdchainlist:
    #     if ch is None:
    #         continue
    #     #don't do anything if derived param is already there
    #     chhname = 'cosmological_parameters--omch2'
    #     cname = 'cosmological_parameters--omega_c'
    #     chhind = ch.paramNames.numberOfName(chhname)
    #     cind = ch.paramNames.numberOfName(cname)
    #     chhlabel = r'\Omega_c h^2'
    #     clabel = r'\Omega_c'

    #     hind = ch.paramNames.numberOfName('cosmological_parameters--h0')
    #     omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
    #     obind = ch.paramNames.numberOfName('cosmological_parameters--omega_b')

    #     omnuh2ind = ch.paramNames.numberOfName('mnu_parameters--omnuh2')
    #     if omnuh2ind ==-1: # see if its in cosm p arams
    #         omnu2ind = ch.paramNames.numberOfName('cosmological_parameters--omnuh2')

    #     if chhind==-1: #need omch2
    #         hvals = ch.samples[:,hind]
    #         omvals = ch.samples[:,omind]
    #         obvals = ch.samples[:,obind]
    #         mvals = omvals*hvals*hvals
    #         bvals = obvals*hvals*hvals
    #         if omnuh2ind==-1: # assume min mass active neutrinos
    #             nvals = 0.06/93.14
    #         else:
    #             nvals =  ch.samples[:,omnuh2ind]
    #         chhvals = mvals - bvals - nvals
    #         cvals = chhvals/(hvals*hvals)
    #         ch.addDerived(chhvals,name=chhname,label = chhlabel,range=[None,None])
    #         ch.addDerived(cvals,name=cname,label = clabel,range=[None,None])
#--------------------------------------------------
def add_As_scaled(gdchains):
    """
    Adds derived parameter 10^9A_s
    """
    mult=1.e9
    #print(">>>>>>ADDING MNU")
    newparamname = 'asx1.e9'
    newlabel = r'10^9A_s '
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        ind = ch.paramNames.numberOfName(newparamname)
        if ind!=-1:
            continue


        rangedict=rangedict_from_gdchain(ch)
        asind = ch.paramNames.numberOfName('cosmological_parameters--a_s')
        if asind !=-1:        #don't compute if neccesary numbers not there

            asvals = ch.samples[:,asind]
            asscaled = mult*asvals
            asrange = rangedict['cosmological_parameters--a_s']
            newparammin = asrange[0]*mult
            newparammax = asrange[1]*mult
            #print(">>>",asrange,newparammin,newparammax)

            #print('...adding derived parameters:',newkey,'to',ch.name_tag)
            ch.addDerived(asscaled,name=newparamname,label = newlabel,range=[newparammin,newparammax])
#--------------------------------------------------
def add_physical_densities(gdchains, bbncprior = True):
    """
    adds Omh2 and Obh2.  Doesn't try to correct bounds

    bbncprior - should be True of bbn_consistency module is used
           this accounts for the flat prior implicit in that module
    """
    mname = 'cosmological_parameters--ommh2'
    bname = 'cosmological_parameters--ombh2'

    mlabel = r'$\Omega_m h^2$'
    blabel = r'$\Omega_b h^2$'

    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        mind = ch.paramNames.numberOfName(mname)
        bind = ch.paramNames.numberOfName(bname)

        hind = ch.paramNames.numberOfName('cosmological_parameters--h0')
        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        obind = ch.paramNames.numberOfName('cosmological_parameters--omega_b')
        if mind==-1: #need ommh2
            hvals = ch.samples[:,hind]
            omvals = ch.samples[:,omind]
            mvals = omvals*hvals*hvals
            ch.addDerived(mvals,name=mname,label = mlabel,range=[None,None])

        if bind==-1: #need ombh1
            hvals = ch.samples[:,hind]
            obvals = ch.samples[:,obind]
            bvals = obvals*hvals*hvals
            if bbncprior:
                brange = [0.005,0.04]
            else:
                brange = [None,None]

            ch.addDerived(bvals,name=bname,label = blabel,range=brange)
#--------------------------------------------------
def add_omega_c(gdchains):
    """
    adds Omega_c = Omega_m - Omega_b - Omega_nu
    """
    chhname = 'cosmological_parameters--omch2'
    cname = 'cosmological_parameters--omega_c'

    chhlabel = r'$\Omega_c h^2$'
    clabel = r'$\Omega_c$'

    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        chhind = ch.paramNames.numberOfName(chhname)
        cind = ch.paramNames.numberOfName(cname)

        hind = ch.paramNames.numberOfName('cosmological_parameters--h0')
        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        obind = ch.paramNames.numberOfName('cosmological_parameters--omega_b')

        omnuh2ind = ch.paramNames.numberOfName('mnu_parameters--omnuh2')
        if omnuh2ind ==-1: # see if its in cosm p arams
            omnu2ind = ch.paramNames.numberOfName('cosmological_parameters--omnuh2')

        if chhind==-1: #need omch2
            hvals = ch.samples[:,hind]
            omvals = ch.samples[:,omind]
            obvals = ch.samples[:,obind]
            mvals = omvals*hvals*hvals
            bvals = obvals*hvals*hvals
            if omnuh2ind==-1: # assume min mass active neutrinos
                nvals = 0.06/93.14
            else:
                nvals =  ch.samples[:,omnuh2ind]
            chhvals = mvals - bvals - nvals
            cvals = chhvals/(hvals*hvals)
            ch.addDerived(chhvals,name=chhname,label = chhlabel,range=[None,None])
            ch.addDerived(cvals,name=cname,label = clabel,range=[None,None])

#--------------------------------------------------
def add_omega_L(gdchains):
    """
    adds Omega_L = 1.-Omega_m - Omega_k
    """
    olname = 'cosmological_parameters--omega_l'
    ollabel = r'$\Omega_{\Lambda}$'

    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        olind = ch.paramNames.numberOfName(olname)
        # or if we have time-dep DE
        wind=  ch.paramNames.numberOfName('cosmological_parameters--w')

        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        okind = ch.paramNames.numberOfName('cosmological_parameters--omega_k')

        if (olind==-1) and (wind==-1): #need omega_L and don't have DE eos w
            omvals = ch.samples[:,omind]
            if okind!=-1:
                okvals = ch.samples[:,okind]
            else:
                okvals=0
            olvals = 1.-omvals -okvals

            ch.addDerived(olvals,name=olname,label = ollabel,range=[None,None])

#--------------------------------------------------
def add_wp(gdchains,cutsum=True):
    """
    compute w(a) at pivot redshift, defined such that
    w(a) = w_p + (a_p - a)*w_a = w_0 + (1-a)*w_a
    and a_p = 1+C_{w_0,w_a}/C_{w_a,w_a}
    where C is the parameter covariance computed from the chain
    (a_p is scale factor where we have the strongest constraint on w(a)

    cutsum - add w0+wa as derived param, including hard upper bound of 0
    """
    woname = 'cosmological_parameters--w'
    waname = 'cosmological_parameters--wa'
    wpname = 'cosmological_parameters--wp'
    apname = 'a_pivot'
    zpname = 'z_pivot'
    wplabel = DEFAULT_PLABELS[wpname]
    aplabel = DEFAULT_PLABELS[apname]
    zplabel = DEFAULT_PLABELS[zpname]

    # ^there will only be one value of a_pivot, so it's a little silly to store
    # for every sample, but this seems the easiest way to automate this for now
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        woind = ch.paramNames.numberOfName(woname)
        waind = ch.paramNames.numberOfName(waname)
        wpind = ch.paramNames.numberOfName(wpname)
        if (waind==-1) or (woind==-1):
            # if we don't have both w and wa, there's nothing to compute
            continue
        elif wpind==-1: #need wp
            wo = ch.samples[:,woind]
            wa = ch.samples[:,waind]
            weights = ch.weights
            cov = np.cov(wo,wa,aweights = weights)
            Cw0wa = cov[0,1]
            Cwawa = cov[1,1]
            ap = 1.+ Cw0wa/Cwawa
            zp = 1./ap -1.
            #print(">>>>>ADDING WP", Cw0wa, Cwawa, ap, zp)
            wp = wo + wa*(1.-ap)
            ch.addDerived(wp,name=wpname,label = wplabel,range=[None,None])
            ch.addDerived(ap*np.ones(wp.size),name=apname,label = aplabel,range=[None,None])
            ch.addDerived(zp*np.ones(wp.size),name=zpname,label = zplabel,range=[None,None])
            if cutsum:
                ch.addDerived(wo+wa,name='wowasum',label =r'w_0+w_a',range=[-30,0])


#--------------------------------------------------
def add_S8(gdchains):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds the derived parameter S8. [TODO need to test this]
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        S8ind = ch.paramNames.numberOfName('S8')
        if S8ind!=-1:
            continue

        omgrowind = ch.paramNames.numberOfName('cosmological_parameters_growth--omega_m_growth')
        newkey = 'S8'
        newlabel = 'S_8'
        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8')
        if sig8ind==-1:# wasn't found
            # try lowercase version
            sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8'.lower())
        om = ch.samples[:,omind]
        sig8 = ch.samples[:,sig8ind]
        if not np.all(np.isnan(sig8)):
            S8 = sig8*((om/0.3)**.5)
            #print('>>>>S8 RANGE',np.min(S8),np.max(S8))
            S8range = [None,None] #no input range for sigma8
            #print('...adding derived parameters:',newkey,'to',ch.name_tag)
            ch.addDerived(S8,name=newkey,label = newlabel,range=S8range)
            
def add_W8(gdchains):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds the derived parameter W8 = S8 * |w|^0.15. 
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        #don't do anything if derived param is already there
        W8ind = ch.paramNames.numberOfName('W8')
        if W8ind!=-1:
            continue

        newkey = 'W8'
        newlabel = 'W_8'
        wind = ch.paramNames.numberOfName('cosmological_parameters--w')
        # we need w sampled, avoid breaking for LCDM chains
        if wind == -1:
            continue
        w = ch.samples[:,wind]
        # we also need S8, which is added previously
        S8ind = ch.paramNames.numberOfName('S8')
        if S8ind == -1:
            continue
        S8 = ch.samples[:,S8ind]
        if not np.all(np.isnan(S8)):
            W8 = S8 * np.abs(w) ** 0.15
            #print('>>>>S8 RANGE',np.min(S8),np.max(S8))
            W8range = [None,None] #no input range for sigma8
            #print('...adding derived parameters:',newkey,'to',ch.name_tag)
            ch.addDerived(W8,name=newkey,label = newlabel,range=W8range)
            
#--------------------------------------------------
def add_bias_marg_params(gdchains, maxn=10):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds bias*sig8 derived parameters
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        # figure out what parametes we need,
        #don't do anything if derived param is already there #
        lin_b1ind = ch.paramNames.numberOfName('bias_lens--b1')
        b1eind = ch.paramNames.numberOfName('bias_lens--b1e_bin1')
        b1e_sig8ind = ch.paramNames.numberOfName('bias_lens--b1e_sig8_bin1')
        sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8')
        if sig8ind==-1:# wasn't found
            # try lowercase version
            sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8'.lower())
        havebe = b1eind != -1
        havebesig = b1e_sig8ind != -1

        #print("havebe",havebe)
        #print('havebesig',havebesig)

        if (not (havebe or havebesig)) and (lin_b1ind==-1):
            # no recognized bias terms in chain
            continue
        if havebesig:
            if b1eind==-1:
                # fill in bias terms
                n = 1
                sig8 = ch.samples[:,sig8ind]
                while n<=maxn:
                    b1sig8ind = ch.paramNames.numberOfName('bias_lens--b1e_sig8_bin{}'.format(n))
                    b2sig8sqind = ch.paramNames.numberOfName('bias_lens--b2e_sig8sq_bin{}'.format(n))
                    #print('bsiginds',b1sig8ind,b2sig8sqind)
                    if b1sig8ind>0: #have parameter for this bin
                        b1key = 'bias_lens--b1e_bin{}'.format(n)
                        b1sig8 = ch.samples[:,b1sig8ind ]
                        b1 = b1sig8/sig8
                        ch.addDerived(b1,name=b1key,label = DEFAULT_PLABELS[b1key],range=[None,None])
                        if b2sig8sqind>0:
                            b2key = 'bias_lens--b2e_bin{}'.format(n)
                            b2sig8sq = ch.samples[:,b2sig8sqind]
                            b2 = b2sig8sq/sig8/sig8
                            ch.addDerived(b2,name=b2key,label = DEFAULT_PLABELS[b2key],range=[None,None])
                        if lin_b1ind==-1:
                            # copy b1 in to param matching linear bias
                            basebkey = 'bias_lens--b{}'.format(n)
                            ch.addDerived(b1,name=basebkey,label = DEFAULT_PLABELS[basebkey],range=[None,None])
                        n+=1

                    else: # covered all the lens bins
                        n=maxn+1
        elif havebe:
            # set up the b*sig8 params
            sig8 = ch.samples[:,sig8ind]
            n=1
            while n<=maxn:
                b1ind = ch.paramNames.numberOfName('bias_lens--b1e_bin{}'.format(n))
                b2ind = ch.paramNames.numberOfName('bias_lens--b2e_bin{}'.format(n))
                if b1ind>0: #have parameter for this bin
                        b1 = ch.samples[:,b1ind]
                        b1sig8 = b1*sig8
                        b1sig8key = 'bias_lens--b1e_sig8_bin{}'.format(n)
                        b2sig8sqkey = 'bias_lens--b2e_sig8sq_bin{}'.format(n)
                        ch.addDerived(b1sig8,name=b1sig8key,label = DEFAULT_PLABELS[b1sig8key],range=[None,None])

                        if b2ind>0:
                            b2 = ch.samples[:,b2ind]
                            b2sig8sq = b2*sig8*sig8
                            b2sig8sqkey = 'bias_lens--b2e_sig8sq_bin{}'.format(n)
                            ch.addDerived(b2sig8sq,name=b2sig8sqkey,label = DEFAULT_PLABELS[b2sig8sqkey],range=[None,None])
                        if lin_b1ind==-1:
                            # copy b1 in to param matching linear bias
                            basebkey = 'bias_lens--b{}'.format(n)
                            rangedict = rangedict_from_gdchain(ch)
                            b1range = rangedict['bias_lens--b1e_bin{}'.format(n)]
                            ch.addDerived(b1,name=basebkey,label = DEFAULT_PLABELS[basebkey],range=b1range)
                        n+=1

                else: # covered all the lens bins
                    n=maxn+1
        elif lin_b1ind!=-1:
            # don't have any of the be parameters, just copy lin params to be1
            # in case we want to compare to nlbias run
            n=1
            sig8 = ch.samples[:,sig8ind]
            while n<=maxn:
                bind = ch.paramNames.numberOfName('bias_lens--b{}'.format(n))
                if bind>0: #have parameter for this bin
                    b1 = ch.samples[:,bind ]
                    b1sig8 = b1*sig8
                    b1key = 'bias_lens--b1e_bin{}'.format(n)
                    b1sig8key = 'bias_lens--b1e_sig8_bin{}'.format(n)

                    ch.addDerived(b1sig8,name=b1sig8key,label = DEFAULT_PLABELS[b1sig8key],range=[None,None])

                    rangedict = rangedict_from_gdchain(ch)
                    b1range = rangedict['bias_lens--b{}'.format(n)]
                    ch.addDerived(b1,name=b1key,label = DEFAULT_PLABELS[b1key],range=b1range)

                    n+=1

                else: # covered all the lens bins
                    n=maxn+1
        else: #no recognized bias parameters sampled
            continue

#--------------------------------------------------
def add_bias_marg_w_params(gdchains, maxn=10):
    """
    Takes in a single MCSample object or a list of them, goes through and
    adds bias*sig8*w derived parameters
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue
        # figure out what parametes we need,
        #don't do anything if derived param is already there #
        lin_b1ind = ch.paramNames.numberOfName('bias_lens--b1')
        b1eind = ch.paramNames.numberOfName('bias_lens--b1e_bin1')
        b1e_sig8ind = ch.paramNames.numberOfName('bias_lens--b1e_sig8_bin1')
        sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8')
        wind = ch.paramNames.numberOfName('cosmological_parameters--w')
        if sig8ind==-1:# wasn't found
            # try lowercase version
            sig8ind = ch.paramNames.numberOfName('COSMOLOGICAL_PARAMETERS--SIGMA_8'.lower())
        havebe = b1eind != -1
        havebesig = b1e_sig8ind != -1

        #print("havebe",havebe)
        #print('havebesig',havebesig)

        if (not (havebe or havebesig)) and (lin_b1ind==-1):
            # no recognized bias terms in chain
            continue
        elif havebe:
            # set up the b*sig8*w^-0.25 params
            sig8 = ch.samples[:,sig8ind]
            n=1
            b1ind = ch.paramNames.numberOfName('bias_lens--b1e_bin{}'.format(n))
            b2ind = ch.paramNames.numberOfName('bias_lens--b2e_bin{}'.format(n))
            if b1ind>0: #have parameter for this bin
                    b1 = ch.samples[:,b1ind]
                    w = ch.samples[:,wind]
                    b1sig8w = b1*sig8*np.abs(w)**(-0.25)
                    b1sig8wkey = 'bias_lens--b1e_sig8_w_bin{}'.format(n)
                    ch.addDerived(b1sig8w,name=b1sig8wkey,label = DEFAULT_PLABELS[b1sig8wkey],range=[None,None])



#--------------------------------------------------
def add_OmegaDE(gdchains,Nz=500,zfile='z_dist.txt'):
    """
    Assuming tabulated H(z) is saved as extra output, compute Omega_m(z) and Omega_DE(z) as derived parameters

    for now just reading in zs from a files since for some reason i'm not pulling it out of the chain right
    """
    ckmpers=299792.
    z_dist = np.loadtxt(zfile)
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains
    for ch in gdchainlist:
        if ch is None:
            continue

        omind = ch.paramNames.numberOfName('cosmological_parameters--omega_m')
        h0ind = ch.paramNames.numberOfName('cosmological_parameters--h0')
        om = ch.samples[:,omind]
        h0 = ch.samples[:,h0ind]*100
        okind = ch.paramNames.numberOfName('cosmological_parameters--omega_k')
        if okind!=-1:
            print("Not set up to compute Omega_DE(z) for non-flat Universe, skipping...")
            continue
        for i in range(1,Nz+1):
            #zind = ch.paramNames.numberOfName('DISTANCES--Z_{0:d}'.format(i))
            hind = ch.paramNames.numberOfName('DISTANCES--H_{0:d}'.format(i))
            #print(i,'hind',hind,'zind',zind)
            if hind==-1: #isn't in file
                break
            else:
                h = ckmpers*ch.samples[:,hind]
                ##z = ch.samples[:,zind]
                z = z_dist[i-1]

                Omz = om*((1+z)**3)/((h/h0)**2)
                ODEz = 1.-Omz
                newkey = 'DISTANCES--Omega_DE_{0:d}'.format(i)

                ch.addDerived(ODEz,name=newkey,label = newkey,range=[None,None])


#--------------------------------------------------
def copy_NLA_params_to_TATT_names(gdchains):
    """
    Copies the A and alpha parameters used for  NLA model into the A_1 and alpha_1 tatt  parameter names, to facilitate comparison of TATT and NLA chains
    """
    A_nla="intrinsic_alignment_parameters--a"
    alpha_nla="intrinsic_alignment_parameters--alpha"
    A_tatt="intrinsic_alignment_parameters--a1"
    alpha_tatt="intrinsic_alignment_parameters--alpha1"
    bias_ta = "intrinsic_alignment_parameters--bias_ta"
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
    else:
        gdchainlist = gdchains

    for ch in gdchainlist:
        if ch is None:
            continue
        rangedict = rangedict_from_gdchain(ch)
        #don't do anything if derived param is already there
        A_tatt_ind = ch.paramNames.numberOfName(A_tatt)
        if A_tatt_ind!=-1:
            continue

        A_nla_ind = ch.paramNames.numberOfName(A_nla)
        alpha_nla_ind = ch.paramNames.numberOfName(alpha_nla)
        ch.addDerived(ch.samples[:,A_nla_ind],name=A_tatt,range=rangedict[A_nla])
        ch.addDerived(ch.samples[:,alpha_nla_ind],name=alpha_tatt,range=rangedict[alpha_nla])


#==================================================================
# contour plotting
#==================================================================

def make_chainplot(inputflist,params,outname,inlabels,\
                   intruthvals={},\
                   kdesmooth=.5,linestyles=None,colors=None,outdir='.',\
                   ftype='png',shade=True,indatdir = '.',legtitle='',\
                   figwidth=None,legfontsize=10,\
                   rangedict = None,\
                   graytext="",notedict=None,showtruth=False,truthline_width=1.,\
                   showmap=False,mapvalslist=None,\
                   axes_fontsize = None, lab_fontsize = None,
                   linewidths=None,\
                   singleplot=False,rectplot=False,rectratio=None,\
                   legloc=None,hideleg=False,\
                   vlines=[],hlines=[],linedict=None,savefracs=True,\
                   ticks = None, minorticks = None,\
                   param_limits=None,figheight=None, contour_args=None,\
                   singleplot_filldict = None,ylim_multiplier=1.,\
                   legkwargs={},legframe=False,legaxinds=None,legncol=None,\
                   clearfig=False,pointlist=[],showprofilelike=False,\
                   showprofilelike_2d=False,showhist_2d=False,profilepost=False,\
                   proflikerange=5):
    """
    Given input list of chain  files, params to plot, and
    (optional) truth values, and (optional) labels, and a number of formatting
    options, plots posteriors.

    This function can make 4 times of plot:
    - 1d posterior: If only one parameter name is given (len(params)==1),
       plots 1d marginalized posteriors
    - 2d single panel: If two parameters and singleplot=True,
       does a single contour plot without 1d projected posteriors along edges
    - triangle plot: if singleplot is False and nparams>1, does triangle plot
    - rectplot: rectangular grid of plots, used if e.g. you wanto show a
        row of plots looking at how one parameter covaries with a view others

    Features:
      - If there's an issue reading in or plotting one of the chains
        (e.g. if the file is missing), this script automatically accounts
        for that and makes sure labels, colors, linestyles stay with the
        correct posteriors.
      - Flexible figure sizing, font size control so that while plots can
        automatically be generated for quick testing, they can also be
        nicely formatted for publication.
      - Uses dictionaries to account for parameter ranges, add dashed lines
        for truth value.
      - Can add watermark to note if data is simulated, blinded, etc

    Arguments: (sorted based on similar use rather than input order)

    #--------
    # File naming
    #--------
    inputflist - list of file names, or list of MCSamples objects
        If filenames, reads in chains to generate MCSamples objects, then plots.
        Note: if you're making multiple plots of the same chains,
        your script will run faster if you read in the chains once
        and just pass the MCSamples objects. Also, if different chains
        have different prior ranges, should read them in separately then pass
        to this function.
    inlabels - list of labels for chains to appear in the legend;
    indatdir - path to directory where chain files are. Only used if
        doing getdist chain read-in.
    outname - name of output file, without file suffix (no .png or .pdf)
    outdir - directory where to put output file, defaults to '.'
    ftype - filetype for output, default is 'png'
    clearfig - if True, does fig.clf after saving figure; useful
               if you're generating a lot of plots at once

    #--------
    # Parameters to show
    #--------
    params - list of parameter names to include in plot. Name should match
             formatting in column header list of chain file.
             If rectplot==True, then instead pass nested list in format
              [[col1,col2,... colN],[row1,row2,..., rowN]]
    rangedict - dictionary, keys = parameter names, values = (min,max) tuples
        giving prior range that chains were run with. Used for getdist
        edge corrections. Mainly just used by getdist read-in, but also
        gets used to evaulate posterior fraction if 'savefracs' options used.
        If default None, ranges inferred from chain header (recommended)
    param_limits - parameter ranges for plots. Defaults set based on chain range
         Generally matches format expected by getdist for the various plot types
         1d plot: [xmin,xmax]
         2d 1 panel: [xmin, xmax, ymin, ymax]
         rectplot & triangle: dictionary,
              {‘param1’:(min1, max1),’param2’:(min2,max2)}.
              If this includes parameters that are not plotted, samples outside
              limits will still be removed

    #--------
    # Plots style and size
    #--------
    singleplot - bool, if only 2 params do 1-panel plot instead of triangle
    rectplot - bool, default False, if True do rectangular array of subplots
               (needs different format of parameter list than other plot types)
    rectratio - if using rectplot, ratio of width and height of subplots
    figwidth - width of figure in inches. Default is 3.5" for 1d and
         single panel plots (one column width). For triangle plots,
         defaults to minumum of 7" (2 column width) or size that makes each
         subplot column 2" wide.
    figheight - Optional Height of fig in inches. Mostly set based on width.
         For 1d plot, default is 0.6*width,
         for singlepanel 2d default is height=width, if rectplot it is set
         automatically given width and rectratio, for triangle set auto
         by getdist given width and number of parameters

    ylim_multiplier -  Only used for 1d plot. default 1. If >1, raises ymax
         of marginal posterior plots, if you want more white space for labels

    #--------
    # Contour styles
    #--------

    kdesmooth - KDEsmoothing for getdist; only used by getdist readin
        (not used if MCSamples objects passed in instead of filenames)
    linestyles - list of linestyles for each chain. Defaults to all seet to '-'
         (may not have an impact fro shaded contours)
    colors - list of colors for each chain. If not passed, uses a list
         of 12 default colors.
    shade - bool or list of one bool per chain, specifying if we want
         filled contours. If 1 bool, sets same choice for all chains.
    linewidths - optional list of linewidths for contours. Not used
         if contour is filled. Defaults to 1 for all lines
    contour_args - for passing contour formatting to getdist
    singleplot_filldict - used if you want to shade a region of a 1d posterior or
                          2d singlepanel plot

    #--------
    # Legend formatting, font sizes
    #--------
    legtitle - legend title, default is not to have one
    legfontsize - legend font size, default is 10
    legloc - legend location argument, default is different for each plot type
    hideleg - bool. If true, don't show a legend. default false
    legframe - bool; if true, show legend outline. Default false (no frame)
    legaxinds - Used only for rectplot. If None (default) - legend location
        is wrt figure coordinates if tuple of integers, specifies indices of
        subplot to associate legend with.
    legkwargs - other matplotlib legend arguments; dictionary with
        keys as argument variables, values = what you want to set them
    legncol - number of columns of legend entries

    axes_fontsize - font size for plot axis tick size. default 8
    lab_fontsize - font size for axis labels, default 10
    ticks - option, probably only used for paper plots, manually set ticks
       1d - list of tick values
       single plot  - [[xticks],[yticks]]
       rectplot, triangle plot - dictionary with parameter names as keys,
                  values are lists of ticks
    minorticks - same as ticks, but only set up for rectplot and triangle plot

    #--------
    # Annotation: text, lines, filled regions
    #--------
    graytext - Text for watermark text
    notedict - optional dictionary used to format watermark text, can have
         keys: fontsize, xloc, yloc, ha (horizontal alignment),
         va (vertical alignment), rotation. If given, values are used
         as arguments for call to fig.text() to write watermark. 1d,
         snglepanel, and triangle plots have different defaults.

    intruthvalues - dictionary of fiducial values that you want to mark on
        the plot as dashed gray lines
    truthline_width - linewidth for fiducial value dashed line
    showtruth - bool, do you want to plot fid value lines? Default is False

    vlines - Only used for 1d and single planel plots: list of values where to
        plot vertical dotted gray lines
    hlines - Only used for 1d and single planel plots: list of values where to
        plot horizontal dotted gray lines
    linedict - for triangle plots, dictionary of parameter names and values
        that we want to add lines for. Keys are parameter names, values
        are associated parameter values.
    savefrac - if True and plotting 1d, will save an output txt file with
        same name as plot, containing fraction of posterior above and below
        each vline value
    pointlist - list of dictionaries corresponding to sets of
        markers to plot on top of contours.  ONLY IMPLEMENTED FOR TRIANGLE PLOTS
        Some keys will be parameter names
        with values corresponding to where to put the marker. Other
        keys can have formatting info:
         - label - string label for points if you want it in the legend
         - c - color
         - markerkwargs - dictionary of keyword arguments for formatting markers
         - linekwargs - dicinotary of keyword args for line formatting
    showprofilelike - ONLY IMPLEMENTED FOR TRIANGLE PLOTS
        for each chain, in diagonal 1d panels also show pseudo profile likelihood
        normalize so max is at 1, min at 0.
        use color matching chain, just with dotted line and alpha=0.5 or something
        (not recommended for pub quality plots, but useful for diagnostics)

    showprofilelike_2d -ONLY IMPLEMENTED FOR TRIANGLE PLOTS, ONLY FOR ONE CHAIN!
        in 2d panels, plot heat map of max likelihood for each range of parameter values
    showhist_2d -ONLY IMPLEMENTED FOR TRIANGLE PLOTS, ONLY FOR ONE CHAIN!
        in 2d panels, plot heat map of sum of sample weights each bin of parameter values
    profilepost = if True profile likehoods will be for post rather than like
    """
    dotight=False
    Nparams = len(params)
    Nfile = len(inputflist)

    if showmap and mapvalslist is None:
        mapvalslist = [{}]*Nfile
    if showtruth and not rectplot:
        truthvals = parse_truthvals(intruthvals,params)
    #print(truthvals)
    if linestyles is None:
        linestyles = ['-']*Nfile
        #linestyles.append('--')
    if linewidths is None:
        linewidths = [1]*Nfile
    if colors is None:
        colors = DEFAULT_COLORS
    if linestyles is None:
        linestyles = Nfile*['-']
    if notedict is None:
        notedict = {}

    if type(inputflist[0])==str:
        ingdchains = []
        for i,inputf in enumerate(inputflist):
            ingdchains.append(get_gdchain(inputf,inlabels[i],params,\
                                          kdesmooth,indatdir, rangedict = rangedict))
    else:
        needMCsamples = False
        ingdchains = inputflist

    #go through list and set up version of lists with missing data removed
    gdchains = []
    labels = []
    usecolors = []
    uselinestyles = []
    uselinewidths = []
    useshade = []
    for i in range(Nfile):
        if ingdchains[i] is not None:
            gdchains.append(ingdchains[i])
            #labels.append('test')

            if inlabels is not None:
                labels.append(inlabels[i])
            else:
                labels.append(ingdchains[i].name_tag[0])

            usecolors.append(colors[i])
            uselinestyles.append(linestyles[i])
            uselinewidths.append(linewidths[i])
            if type(shade)==list:
                useshade.append(shade[i])
            else:
                useshade.append(shade)
        else:
            if inlabels is not None:
                print("missingdat for ",inlabels[i])
            else:
                print("missingdat for chain number",i)

    print( "...plotting")

    if showprofilelike_2d and len(gdchains)>1:
        raise ValueError("Can't use showprofilelike_2d with more than one chain")
    if showhist_2d and len(gdchains)>1:
        raise ValueError("Can't use showhist_2d with more than one chain")
    if showprofilelike_2d and showhist_2d:
        raise ValueError("Can't use showprofilelike_2d and showhist_2d at the same time")

    if axes_fontsize is None:
        axes_fontsize = 8
    if lab_fontsize is None:
        lab_fontsize = 10

    if len(params)==1: #only 1d plot

        # set up margins
        dotight=True
        tmarg = .05
        rmarg = .05

        bmarg = 2.3*(axes_fontsize + lab_fontsize)/72.

        lmarg = 1.5*lab_fontsize/72.
        if figwidth is None:
            figwidth = min(lmarg+rmarg +3,3.5)

        if figheight is None:
            ratio = .6
            figheight = figwidth*ratio
        else:
            ratio = figheight/figwidth
        #print('ratio',ratio)

        if legloc is None:
            legloc = "upper right"
        gdp = gdplots.getSinglePlotter(width_inch=figwidth,ratio=ratio)
        gdp.settings.scaling=False
        gdp.settings.tight_layout=False
        gdp.settings.rc_sizes(axes_fontsize=axes_fontsize,lab_fontsize=lab_fontsize)
        param_limits = param_limits[params[0]]
        gdp.plot_1d(gdchains,param = params[0],ls = uselinestyles,lws=uselinewidths,colors = usecolors, lims = param_limits)

        if showtruth:
            gdp.add_x_marker(truthvals[0],'gray',ls='--',lw=truthline_width)
        if showmap:
            for cc in range(Nfile): #loop over the chains
                if mapvalslist[cc] is None:
                    continue
                else:
                    gdp.add_x_marker(mapvalslist[cc][0],color=usecolors[cc],marker='o')

        if vlines:
            for x in vlines:
                gdp.add_x_marker(x,'gray',ls=':',lw=1)


        if not hideleg:
            leglabels = []
            for lab in labels:
                if (lab is not None) and lab:
                    leglabels.append(lab)
                else:
                    leglabels.append("_nolegend_")

            if 'framealpha' not in legkwargs.keys():
                leg = gdp.add_legend(legend_labels = leglabels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,framealpha=0,**legkwargs)
            else:
                leg = gdp.add_legend(legend_labels = leglabels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,**legkwargs)

            if not legframe:
                leg.get_frame().set_linewidth(0.)
            leg.set_title(legtitle)
            leg.get_title().set_fontsize(legfontsize)
        else:
            leg = gdp.add_legend(legend_labels = [],legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,framealpha=0)
            leg.set_bbox_to_anchor((1.,1.))
            if not legframe:
                leg.get_frame().set_linewidth(0.)
            leg.set_title(legtitle)
            leg.get_title().set_fontsize(legfontsize+2)

        fig = plt.gcf()
        ax = plt.gca()

        plt.subplots_adjust(right=1.-rmarg/figwidth,left=lmarg/figwidth,bottom=bmarg/figheight,top=1.-tmarg/figheight)



        #get a little more white space at top
        ymin,ymax = ax.get_ylim()
        ax.set_ylim(ymin,ymax*ylim_multiplier)

        # set x limits they are passed as arguments
        if param_limits is None:
            xmin,xmax = ax.get_xlim()
            ax.set_xlim(xmin,xmax) #this helps make room for the legend
        else:
            ax.set_xlim(param_limits[0],param_limits[1])
        if ticks is not None:
            # expects array of values
            ax.xticks(ticks)

        plt.ylabel('$\\mathcal{P}$',fontsize = lab_fontsize)

        if graytext:
            #print(notedict)
            if 'fontsize' not in notedict.keys():
                notedict['fontsize']=24
            if 'xloc' not in notedict.keys():
                notedict['xloc'] = .05
            if 'yloc' not in notedict.keys():
                notedict['yloc'] = .5
            if 'ha' not in notedict.keys():
                notedict['ha'] = 'left'
            if 'va' not in notedict.keys():
                notedict['va'] = 'center'
            if 'rotation' not in notedict.keys():
                notedict['rotation'] = 0.
            if 'color' not in notedict.keys():
                notedict['color'] = 'gray'
            if 'alpha' not in notedict.keys():
                notedict['alpha'] = 0.4
            if 'kwargs' not in notedict.keys():
                notedict['kwargs']={}
            fig.text(notedict['xloc'], notedict['yloc'], graytext,\
                     color=notedict['color'], alpha = notedict['alpha'],\
                     fontsize= notedict['fontsize'], wrap=True, \
                     ha=notedict['ha'], va=notedict['va'], rotation=notedict['rotation'],**notedict['kwargs'])

        if singleplot_filldict is not None:
            if type(singleplot_filldict) is not list:
                singleplot_filldict=[singleplot_filldict]
            for filldict in singleplot_filldict:
                fillx=filldict['x']
                #print(">>>>",fillx.shape)
                fillx_min=fillx[0]
                fillx_max = fillx[1]
                fill_kwargs = filldict['kwargs']
                if 'y' in filldict.keys(): # we just want to plot a line,  not a filled region
                    ax.plot(fillx,fidict['y'],**fill_kwargs)
                else:
                    ax.axvspan(fillx_min,fillx_max,**fill_kwargs)


        if savefracs:
            fracfname =   os.path.join(outdir,outname+'.probfracs.txt')
            print("Saving prob fracs to ",fracfname)
            fracf = open(fracfname,'w')
            paramname = params[0]
            fracf.write(paramname+': [value] [prob below] [prob above] [Nsig equiv of singlet-ail prob for Normal distrib.]\n')
            for i,chain in enumerate(gdchains):
                if chain.label is None:
                    fracf.write('\n\n'+inlabels[i]+'\n')
                else:
                    fracf.write('\n\n'+chain.label+'\n')
                # get dist 1d density object
                dens1d = (chain.get1DDensity(paramname))
                post = dens1d.P #posterior array
                vals = dens1d.x
                rangedict = rangedict_from_gdchain(chain)
                xmin = rangedict[paramname][0]
                xmax = rangedict[paramname][1]
                if xmin is None:
                    pind = chain.paramNames.numberOfName(paramname)
                    xmin = np.min(chain.samples[:,pind])
                    xmax = np.max(chain.samples[:,pind])
                for xline in vlines:
                    #print('>>> ON savefracs',paramname,xmin,xmax,xline)
                    tot = sp.integrate.quad(lambda x: dens1d(x),xmin,xmax,epsabs=1.e-4,limit=100)[0]
                    #print('  tot',tot)
                    less = sp.integrate.quad(lambda x: dens1d(x),xmin,xline,epsabs=1.e-4,limit=100)[0]
                    more = sp.integrate.quad(lambda x: dens1d(x),xline,xmax,epsabs=1.e-4,limit=100)[0]
                    fracless = less/tot
                    fracmore = more/tot
                    #print('ON',paramname,fracless,fracmore,min(fracless,fracmore))
                    tailNsig = calc_1dfractail_to_sigma(min(fracless,fracmore))
                    fracf.write('{0:+0.4f}  {1:0.4f}  {2:0.4f}  {3:0.4f}\n'.format(xline,fracless,fracmore,tailNsig))
            fracf.close()

    elif (len(params)==2 and singleplot): # one 2d plot

        dotight=True
        if legloc is None:
            legloc = 'upper right'

        tmarg = .1
        rmarg = .15
        if axes_fontsize is None or lab_fontsize is None:
            bmarg = .4
            lmarg = .5
        else:
            bmarg = 2.3*(axes_fontsize + lab_fontsize)/72.
            lmarg = bmarg
        if figwidth is None:
            figwidth = 3.5

        if figheight is None:
            ratio = 1.
            figheight = ratio*figwidth
        else:
            ratio = figheight/figwidth

        gdp = gdplots.getSinglePlotter(width_inch=figwidth,ratio=ratio)
        dotight=True
        gdp.settings.scaling=False
        gdp.settings.tight_layout=False
        gdp.settings.rc_sizes(axes_fontsize=axes_fontsize,lab_fontsize=lab_fontsize)
        gdp.settings.lw_contour=2

        if param_limits is None:
            gdp.plot_2d(gdchains,params,filled=useshade,\
                    colors=usecolors,\
                    ls = uselinestyles,\
                        legend_labels=False, contour_args=contour_args)
        else:
            gdp.plot_2d(gdchains,params,filled=useshade,\
                        colors=usecolors,\
                        ls = uselinestyles,\
                        legend_labels=False,\
                        lims = param_limits,contour_args=contour_args)


        fig = plt.gcf()
        ax = fig.axes[0]
        if figheight is None:
            figheight = fig.get_size_inches()[1]
        plt.subplots_adjust(right=1.-rmarg/figwidth,left=lmarg/figwidth,bottom=bmarg/figheight,top=1.-tmarg/figheight)

        if singleplot_filldict is not None:
            if type(singleplot_filldict) is not list:
                singleplot_filldict=[singleplot_filldict]
            for filldict in singleplot_filldict:
                fillx = filldict['x']
                fill_kwargs = filldict['kwargs']
                if 'y' in filldict.keys():
                    filly = filldict['y']
                    plt.plot(fillx,filly,**fill_kwargs)
                    continue
                else:
                    filly_max = filldict['ymax']
                    filly_min = filldict['ymin']

                if ((filly_max is None) and (filly_min is None)):
                    filly = filldict['y']
                    plt.plot(fillx,filly,**fill_kwargs)

                elif filly_max is None:
                    plt.plot(fillx,filly_min,**fill_kwargs)
                else:
                    plt.fill_between(fillx,filly_min,filly_max,**fill_kwargs)


        if not hideleg:
            leglabels = []
            for lab in labels:
                if (lab is not None) and lab:
                    leglabels.append(lab)
                else:
                    leglabels.append("_nolegend_")
            if 'framealpha' not in legkwargs.keys():
                leg = gdp.add_legend(legend_labels = leglabels,legend_loc=legloc,fontsize = legfontsize,framealpha=0,**legkwargs)
            else:
                leg = gdp.add_legend(legend_labels = leglabels,legend_loc=legloc,fontsize = legfontsize,**legkwargs)


            if not legframe:
                leg.get_frame().set_linewidth(0.)

            leg.set_title(legtitle)
            leg.get_title().set_fontsize(legfontsize+2)
        else:
            leg = gdp.add_legend(legend_labels = [],legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,framealpha=0)
            if not legframe:
                leg.get_frame().set_linewidth(0.)

            leg.set_title(legtitle)
            leg.get_title().set_fontsize(legfontsize+2)

        #add lines for truthvals
        if vlines:
            for x in vlines:
                gdp.add_x_marker(x,'gray',ls=':',lw=1)
        if hlines:
            for y in hlines:
                gdp.add_y_marker(y,'gray',ls=':',lw=1)
        if showtruth:
            plt.axvline(truthvals[0],color='gray',ls='--',lw=truthline_width)
            plt.axhline(truthvals[1],color='gray',ls='--',lw=truthline_width)
        if showmap:
            for cc in range(Nfile):
                if mapvalslist[cc] is None:
                    continue
                else:
                    plt.axvline(mapvalslist[cc][0],mapvalslist[cc][1],color=usecolors[cc],marker='*')

        if ticks is not None:
            ax.set_xticks(ticks[0])
            ax.set_yticks(ticks[1])
        #for ax in fig.axes:
        #    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if graytext:
            if 'fontsize' not in notedict.keys():
                notedict['fontsize']=20
            if 'xloc' not in notedict.keys():
                notedict['xloc'] = 0.01
            if 'yloc' not in notedict.keys():
                notedict['yloc'] = .5
            if 'ha' not in notedict.keys():
                notedict['ha'] = 'left'
            if 'va' not in notedict.keys():
                notedict['va'] = 'center'
            if 'rotation' not in notedict.keys():
                notedict['rotation'] = 90.

            ax.text(notedict['xloc'], notedict['yloc'], graytext,\
                      transform=ax.transAxes,\
                     color='gray', alpha = .4,\
                     fontsize= notedict['fontsize'], wrap=True, \
                     ha=notedict['ha'], va=notedict['va'], rotation=notedict['rotation'])


    elif rectplot:
        # expects dictionaries for param info, but params in
        # [[col1,col2],[row1,row2]] format
        colkeys = params[0]
        rowkeys = params[1]

        Ncol = len(colkeys)
        Nrow = len(rowkeys)
        if not legloc:
            legloc = "upper left"

        if showtruth:
            rowtruthvals = parse_truthvals(intruthvals,rowkeys)
            coltruthvals = parse_truthvals(intruthvals,colkeys)

        # set up margins

        if param_limits is None:
            param_limits = {}
        #print(param_limits)

        tmarg = .05
        rmarg = .05

        bmarg =1.5*(axes_fontsize + lab_fontsize)/72.

        lmarg = 1.8*(lab_fontsize+axes_fontsize)/72.

        if figwidth is None:
            figwidth = min(lmarg+rmarg +Ncol*2,7)

        gdp = gdplots.getSubplotPlotter(width_inch=figwidth,subplot_size_ratio=rectratio)
        gdp.settings.scaling=False
        gdp.settings.tight_layout=False
        gdp.settings.rc_sizes(axes_fontsize=axes_fontsize,lab_fontsize=lab_fontsize)
        gdp.settings.lw_contour=2
        gdp.settings.norm_1d_density = True

        gdp.rectangle_plot(colkeys,rowkeys,roots=gdchains,\
                           filled=useshade,\
                          colors=usecolors,\
                          ls = uselinestyles,\
                          lws = uselinewidths,\
                          legend_labels=[],\
                          param_limits=param_limits,\
        )


        fig = plt.gcf()
        if figheight is None:
            figheight = fig.get_size_inches()[1]
        plt.subplots_adjust(right=1.-rmarg/figwidth,left=lmarg/figwidth,bottom=bmarg/figheight,top=1.-tmarg/figheight)


        #add lines, truthvals, adjust ticks, if desired
        for i in range(Nrow):
            for j in range(Ncol):
                ax = gdp.subplots[i,j] #0,0
                if linedict is not None:
                    pi = rowkeys[i]
                    pj = colkeys[j]
                    if pj in linedict.keys():
                        ax.axvline(linedict[pj],color='gray',ls='-',lw=0.5)
                    if pi in linedict.keys():
                        ax.axhline(linedict[pi],color='gray',ls='-',lw=0.5)
                if showtruth:
                    ax.axvline(coltruthvals[j],color='gray',ls='--',lw=truthline_width)
                    ax.axhline(rowtruthvals[i],color='gray',ls='--',lw=truthline_width)

                if (ticks is not None):
                    ax.set_xticks(ticks[pj])
                    #print("xticks ",pj,ticks[pj])
                    if i!=j:
                        ax.set_yticks(ticks[pi])
                if minorticks is not None:
                    ax.set_xticks(minorticks[pj],True)
                    #print("xticks ",pj,ticks[pj])
                    if i!=j:
                        ax.set_yticks(minorticks[pi],True)

        if not hideleg:
            leglabels = []
            for lab in labels:
                if (lab is not None) and lab:
                    leglabels.append(lab)
                else:
                    leglabels.append("_nolegend_")

            if legaxinds is not None:
                if 'framealpha' not in legkwargs.keys():
                    leg = gdp.add_legend(legend_labels = labels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,ax=gdp.subplots[legaxinds],framealpha=0.,**legkwargs)
                else:
                    leg = gdp.add_legend(legend_labels = labels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,ax=gdp.subplots[legaxinds],**legkwargs)
            else: #legend uses figure coords
                if 'framealpha' not in legkwargs.keys():
                    leg = gdp.add_legend(legend_labels = labels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,figure=True,figure_legend_outside=False,framealpha=0.,**legkwargs)
                else:
                    leg = gdp.add_legend(legend_labels = labels,legend_loc=legloc,legend_ncol=legncol,fontsize = legfontsize,figure=True,figure_legend_outside=False,**legkwargs)

            if not legframe:
                leg.get_frame().set_linewidth(0.)
            if legloc=='upper left':
                leg.set_bbox_to_anchor(((lmarg+0.01)/figwidth,.98))

            leg.set_title(legtitle)


        leg.get_title().set_fontsize(legfontsize+2)
        if graytext:
            if 'fontsize' not in notedict.keys():
                notedict['fontsize']=20
            if 'xloc' not in notedict.keys():
                notedict['xloc'] = .5
            if 'yloc' not in notedict.keys():
                notedict['yloc'] = .25
            if 'ha' not in notedict.keys():
                notedict['ha'] = 'center'
            if 'va' not in notedict.keys():
                notedict['va'] = 'center'
            if 'rotation' not in notedict.keys():
                notedict['rotation'] = 0.

            fig.text(notedict['xloc'], notedict['yloc'], graytext,\
                     color='gray', alpha = .4,\
                     fontsize= notedict['fontsize'], wrap=True, \
                     ha=notedict['ha'], va=notedict['va'], rotation=notedict['rotation'])


    else: #triangle plot
        # set up margins
        if param_limits is None:
            param_limits = {}
        tmarg = .05
        rmarg = .05
        if axes_fontsize is None or lab_fontsize is None:
            bmarg = .4
            lmarg = .5
        else:
            bmarg = 1.8*(axes_fontsize + lab_fontsize)/72.
            lmarg = 1.5*bmarg
        if figwidth is None:
            figwidth = min(lmarg+rmarg +Nparams*2,9)

        gdp = gdplots.getSubplotPlotter(width_inch=figwidth)
        gdp.settings.scaling=False
        gdp.settings.tight_layout=False
        gdp.settings.rc_sizes(axes_fontsize=axes_fontsize,lab_fontsize=lab_fontsize)
        gdp.settings.lw_contour=2
        gdp.settings.norm_1d_density = True
        #gdp.settings.norm_prob_label=r"$\mathcal{P}$"
        gdp.triangle_plot(gdchains,params,filled=useshade,\
                          contour_colors=usecolors,\
                          contour_ls = uselinestyles,\
                          contour_lws = uselinewidths,\
                          legend_labels=[],\
                          param_limits=param_limits,\
                          diag1d_kwargs={'normalized':True},\
                          contour_args=contour_args\
        )


        fig = plt.gcf()
        if figheight is None:
            figheight = fig.get_size_inches()[1]
        plt.subplots_adjust(right=1.-rmarg/figwidth,left=lmarg/figwidth,bottom=bmarg/figheight,top=1.-tmarg/figheight)

        if showprofilelike: # get profile likelihoods
            profiledat = []
            for c in gdchains:
                profiledat.append(get_profile_like(c,params,15,profilepost,param_limits))
                # ^ will be None if no chain

            #print(profiledat)

        #add lines, truthvals, ticks, if desired
        pointhandles= []
        pointlabels = []
        for i in range(Nparams):
            for j in range(i+1):
                pi = params[i]
                pj = params[j]
                ax = gdp.subplots[i,j] #0,0
                #if not i*j:
                #    plt.plot([],[],color='k',linestyle=':',label='dummylabel')
                if linedict is not None:
                    #pi = params[i]
                    #pj = params[j]
                    #print(i,j,pi,pj)
                    if pj in linedict.keys():
                        ax.axvline(linedict[pj],color='gray',ls=':',lw=1)
                    if pi in linedict.keys() and i!=j:
                        ax.axhline(linedict[pi],color='gray',ls=':',lw=1)
                if showtruth:
                    ax.axvline(truthvals[j],color='gray',ls='--',lw=truthline_width)
                    if i>j:
                        ax.axhline(truthvals[i],color='gray',ls='--',lw=truthline_width)
                        ax.axhline(truthvals[i],color='gray',ls='--',lw=truthline_width)

                if i==j and showprofilelike:
                    for n in range(len(gdchains)):
                        x = profiledat[n][pi][0,:]
                        y = copy.deepcopy(profiledat[n][pi][1,:])


                        # normalize to have max and max-loglikerange of profile span panel
                        # (So! don't read too much into values, just look at shape)
                        loglikerange = proflikerange
                        plotmaxy = np.nanmax(y)
                        plotminy = plotmaxy-loglikerange

                        # set all values lower than plotrange span to nan
                        y[y<plotminy] = np.nan

                        # rescale values so it'll span the axis range
                        y = y - np.nanmin(y)

                        ymin,ymax = ax.get_ylim() #scale to put peak towards top of panel
                        y = ymax*.95*y/np.nanmax(y)

                        ax.plot(x,y,alpha=0.7,color=usecolors[n],ls='--',lw=0.7*uselinewidths[n])

                if i!=j and (showprofilelike_2d or showhist_2d):
                 ## Commenting this out as I wasn't able to get the imshow step to work
                 ## the grid setup works fine, so could eventually be usd to put a 2d
                 ## pseudo profile likelihood behind the contours as a heatmap.
                    if showprofilelike_2d:
                        likegrid, likebins = get_profile_like_2d(gdchains[0],pi,pj,15,param_limits,profilepost)
                        vmax=np.nanmax(likegrid)
                        vmin = vmax-proflikerange
                    elif showhist_2d:
                        # these aren't likelihoods, just using the same name to be able to use plotter below
                        likegrid, likebins = get_hist_2d(gdchains[0],pi,pj,15,param_limits)
                        vmin=np.nanmin(likegrid)
                        vmax=np.nanmax(likegrid)
                    #if j==0:
                        #print('>>>',pi,pj)
                        #print("   ",likegrid,likebins)
                    if likegrid is not None:
                        Nlbin = likegrid.shape[0]
                        db = likebins[:,1] - likebins[:,0] # bin with for the 2 params
                        #db = np.tile(db,(likebins.shape[1],1)).T
                        #likebinedges = np.arange((2,likebins.shape[1]+1))
                        #likebinedges[:,:-1] = likebins - db
                        #likebinedges[:,-1] = likebins[:,-1]+db[:,0]

                        xmin,xmax = ax.get_xlim()
                        ymin,ymax = ax.get_ylim()
                        #yellows
                        #mincol = '#fffdf4'#beige1
                        #maxcol =  '#ffd208'#'#fee16c' # yellowish
                        # blues
                        mincol = '#ffffcc'#'#eff3ff'
                        maxcol = '#8cd3db'# '#41b6c4'# '#3182bd'
                        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
                        usecmap = matplotlib.colors.LinearSegmentedColormap.from_list('mymap',[mincol,maxcol],N=256)
                        cmap = matplotlib.cm.ScalarMappable(norm=norm,cmap=usecmap)
                        cmap.set_array([])
                        #gridcols = cmap.to_rgba(likegrid)
                        for a in range(Nlbin): #rows, going through pi values
                            for b in range(Nlbin): # cols, pj vals

                                if np.isfinite(likegrid[a,b]):
                                    ax.add_patch(matplotlib.patches.Rectangle((likebins[1,b]-0.5*db[1],likebins[0,a]-0.5*db[0]),db[1],db[0],facecolor=cmap.to_rgba(likegrid[a,b]),lw=0,zorder=0))

                        # imshow seems like it'd be cleaner but I can't get it to wkr
                        #ax.imshow(likegrid,cmap=usecmap, norm=norm,\
                        #          interpolation = 'none', origin='lower',\
                        #          #extent = [xmin,xmax,ymin,ymax],\
                        #          zorder=-1\
                        #          )

                for p in pointlist:
                    havei = pi in p.keys()
                    havej = pj in p.keys()
                    if 'label' in p.keys():
                        plabel = p['label']
                    else:
                        plabel = ''

                    if 'linekwargs' not in p.keys():
                        p['linekwargs'] = {}
                    if 'markerkwargs' not in p.keys():
                        p['markerkwargs'] = {}

                    if 'dolines' not in p.keys():
                        p['dolines'] = True

                    if 'domarkers' not in p.keys():
                        p['domarkers'] = True



                    #print('  i',havei,pi,p[pi])
                    #print('  j',havej,pj,p[pj])

                    if (i!=j) and (havei and havej) and p['domarkers']: #plot markers
                        pointhandle = ax.scatter(p[pj],p[pi],color=p['c'],**p['markerkwargs'],label=plabel)
                        # ^ this could work if parameter values are arrays, e.g. to plot multiple points
                        #    # want to make sure each set of markers gets into legend
                        if plabel and (plabel not in pointlabels):
                            pointhandles.append( pointhandle)
                            pointlabels.append(plabel)
                    elif havej and p['dolines']: # if we have just the x axis param, or we're on the diagonal
                        #ax.axvline(p[pj],color=p['c'],**p['linekwargs'])
                        if not hasattr(p[pj],'__len__'): # just a number
                            ax.axvline(p[pj],color=p['c'],**p['linekwargs'])
                        else: # array or list of numbers
                            for pointcount,pointval in enumerate(p[pj]):
                                if type(p['c'])!= str:
                                    ax.axvline(pointval,color=p['c'][pointcount],**p['linekwargs'])
                                else:
                                    ax.axvline(pointval,color=p['c'],**p['linekwargs'])
                    elif havei and p['dolines']: # if we have just the y axis param
                        if not hasattr(p[pi],'__len__'):
                            ax.axhline(p[pi],color=p['c'],**p['linekwargs'])
                        else:
                            for pointcount,pointval in enumerate(p[pi]):
                                if type(p['c']) != str:
                                    ax.axhline(pointval,color=p['c'][pointcount],**p['linekwargs'])
                                else:
                                    ax.axhline(pointval,color=p['c'],**p['linekwargs'])


                if (ticks is not None):
                    ax.set_xticks(ticks[pj])
                    #print("xticks ",pj,ticks[pj])
                    if i!=j:
                        ax.set_yticks(ticks[pi])
                if minorticks is not None:
                    ax.set_xticks(minorticks[pj],True)
                    #print("xticks ",pj,ticks[pj])
                    if i!=j:
                        ax.set_yticks(minorticks[pi],True)

        if not hideleg:
            leg = gdp.add_legend(legend_labels = labels,legend_loc='upper right',legend_ncol=legncol,fontsize = legfontsize,figure=True,figure_legend_outside=False,framealpha=0.)

            if not legframe:
                leg.get_frame().set_linewidth(0.)

            leg.set_bbox_to_anchor((1.- rmarg/figwidth,1.))
            leg.set_title(legtitle)
            leg.get_title().set_fontsize(legfontsize)

            # I can't figure out how to add to the existing legend, but we can add another one
            # below it
            if pointhandles:
                legheight = leg.get_window_extent().height/fig.dpi/figheight # fractional height of legend
                fig.legend(handles=pointhandles,labels=pointlabels,fontsize=legfontsize,frameon=False,loc='upper right',framealpha=0.,bbox_to_anchor=(1,1-legheight))


        if graytext:
            if 'fontsize' not in notedict.keys():
                notedict['fontsize']=40
            if 'xloc' not in notedict.keys():
                notedict['xloc'] = .65
            if 'yloc' not in notedict.keys():
                notedict['yloc'] = .65
            if 'ha' not in notedict.keys():
                notedict['ha'] = 'center'
            if 'va' not in notedict.keys():
                notedict['va'] = 'center'
            if 'rotation' not in notedict.keys():
                notedict['rotation'] = -45.

            fig.text(notedict['xloc'], notedict['yloc'], graytext,\
                     color='gray', alpha = .4,\
                     fontsize= notedict['fontsize'], wrap=True, \
                     ha=notedict['ha'], va=notedict['va'], rotation=notedict['rotation'])

    outf = os.path.join(outdir,outname+'.'+ftype)
    print("Saving plot to ",outf)
    if dotight:
        plt.tight_layout()
    plt.savefig(outf,dpi=300)
    if clearfig:
        plt.clf()
        plt.close()
    return gdchains

#==================================================================
#==================================================================
#==================================================================
# Below this point: postprocessing functions for computing
# marginalized constraints, model comparisons, tensions
#
# COPIED FROM Y3EXT, should work generally but hasn't been
# explicitly tested yet in Y6 workspce - may 11, 2023
#==================================================================
#==================================================================
#==================================================================
# functions for creating stats files
#==================================================================
def comp_1d_stat(params,dens1ds,stattype,interp=True,disp=False):
    """
    Given MCSample object for getdist, compute 1d stat statype.
    Return Nparam x Noutdat size array with number in for each param in chain.
    (Noutdat will usually be one, but may include multiple numbers)

    disp is an argument for scipy.optimize, controls output printed to terminal

    """
    Nparams = len(params)

    #print('dens1ds 1:',dens1ds)
    # thise next block of stuff is to handle cases where objects are None
    temp1ds=dens1ds.copy()
    dens1ds = []
    isnone=np.ones(Nparams)

    Nkeep= 0
    for i in range(Nparams):
        if temp1ds[i] is not None:
            isnone[i]=0
            dens1ds.append(temp1ds[i])
            Nkeep +=1
        else:
            isnone[i]=1

        #print(params[i],'is none:',isnone[i])
    #print('about to enter if block for comp_1d_stat')
    if stattype == 'peak1d':
        if interp:
            guesses = np.array([dens1ds[i].x[np.argmax(dens1ds[i].P)] for i in range(Nkeep)])
            outdat = []
            for i in range(Nkeep):
                try:
                    probfunc = lambda x: -1*dens1ds[i].Prob(x)
                    minresult=sp.optimize.fmin(probfunc,guesses[i],disp=disp)[0]
                    outdat.append(minresult)
                except:
                    outdat.append(np.nan)
            outdat = np.array(outdat)
            #outdat = np.array([sp.optimize.fmin(probfuncs[i],guesses[i],disp=disp)[0] for i in range(Nkeep)])
            header = 'param  1dpeak(interp=True)'
        else:
            outdat = np.array([dens1ds[i].x[np.argmax(dens1ds[i].P)] for i in range(Nkeep)])
            header = 'parameter   1dpeak(interp=False)'
    elif stattype == 'lerr68':
        header = 'parameter    lower_68%_bound    has_bound'
        lims = [dens1ds[i].getLimits([.68]) for i in range(Nkeep)]
        outdat = np.array([[lims[i][0],lims[i][2]] for i in range(Nkeep)])
    elif stattype == 'uerr68':
        header = 'parameter    upper_68%_bound    has_bound'
        lims = [dens1ds[i].getLimits([.68]) for i in range(Nkeep)]
        outdat = np.array([[lims[i][1],lims[i][3]] for i in range(Nkeep)])
    elif stattype == 'lerr95':
        lims = [dens1ds[i].getLimits([.95]) for i in range(Nkeep)]
        header = 'parameter    lower_95%_bound    has_bound'
        outdat = np.array([[lims[i][0],lims[i][2]] for i in range(Nkeep)])
    elif stattype == 'uerr95':
        lims = [dens1ds[i].getLimits([.95]) for i in range(Nkeep)]
        header = 'parameter    upper_95%_bound    has_bound'
        outdat = np.array([[lims[i][1],lims[i][3]] for i in range(Nkeep)])
    elif stattype == 'mean': #following cosmosis postprocessing utils
        header = 'parameter mean'
        means = np.array([(dens1ds[i].x*dens1ds[i].P).sum()/dens1ds[i].P.sum() for i in range(Nkeep)])
        outdat = means

    elif stattype == 'median': #following cosmosis postprocessing utils
        header = 'parameter median(interp=True)'
        outdat = []
        for i in range(Nkeep):
            a = np.argsort(dens1ds[i].x)
            P = dens1ds[i].P[a]
            x = dens1ds[i].x[a]
            Pc = np.cumsum(P)
            Pc/=Pc[-1]
            outdat.append(np.interp(0.5,Pc,x))
        outdat = np.array(outdat)
    elif stattype == 'lerr50':
        header = 'parameter    lower_50%_bound    has_bound'
        lims = [dens1ds[i].getLimits([.5]) for i in range(Nkeep)]
        outdat = np.array([[lims[i][0],lims[i][2]] for i in range(Nkeep)])
    elif stattype == 'uerr50':
        header = 'parameter    upper_50%_bound    has_bound'
        lims = [dens1ds[i].getLimits([.5]) for i in range(Nkeep)]
        outdat = np.array([[lims[i][1],lims[i][3]] for i in range(Nkeep)])
    elif stattype== 'std':
        header = 'parameter standard_deviation'
        means = np.array([(dens1ds[i].x*dens1ds[i].P).sum()/dens1ds[i].P.sum() for i in range(Nkeep)])
        stdevs = np.array([np.sqrt((((dens1ds[i].x-means[i])**2)*dens1ds[i].P).sum()/dens1ds[i].P.sum()) for i in range(Nkeep)])
        outdat= stdevs

    elif stattype== 'map':
        raise ValueError("This function can't do MAP")
    else:
        raise ValueError('Stattype {0:s} not recognized.'.format(stattype))

    #print('done with  if block for comp_1d_stat')

    # put back original list into label
    dens1ds = temp1ds
    #print('>>',outdat.shape)

    if len(outdat.shape)==1:
        outdat=np.atleast_2d(outdat).T
        outNnum=1

    tempout = np.ones((Nparams,outdat.shape[1]))
    count=0
    for i in range(Nparams):
        if isnone[i]:
            tempout[i,:]*=np.nan
        else:
            tempout[i] = outdat[count,:]
            count+=1
    #print('test2')

    outdat = tempout


    return outdat, header
#--------------------------------------------------
def comp_1d_margpostcomps(compdict,dens1ds,paramlist,rangedict):
    """
    Given MCSample object for getdist, compute 1d stat statype.
    Return Nparam x Noutdat size array with number in for each param in chain.
    (Noutdat will usually be one, but may include multiple numbers)

    Returns dictionary with keys=parameter names
    values are a array containing [comparison value, frac of post above, frac below, single tail equiv 1d nornal sigma]

    """
    outdict={}
    #print(type(rangedict))
    #print(rangedict.keys())
    for i,p in enumerate(paramlist):

        if (dens1ds[i] is not None) and (p in compdict.keys()):
            #post = dens1ds[i].P #posterior array
            xvals = dens1ds[i].x #parameter values
            # have 1d marg post, and have a value we want to compare it to
            #print('     have marg post and comp val')
            compvals=compdict[p]

            if type(compvals) in [int, float]:
                compvals=[compvals]

            try:
                xmin = rangedict[paramname][0]
                xmax = rangedict[paramname][1]
            except:
                xmin = None
                xmax = None
            if xmin is None:
                xmin = np.min(xvals)
                xmax = np.max(xvals)

            outdat = []
            for val in compvals:
                #print('comparing val',val,xmin,xmax)
                tot = sp.integrate.quad(lambda x: dens1ds[i](x),xmin,xmax,epsabs=1.e-4,limit=100)[0]
                #print('  tot',tot)
                less = sp.integrate.quad(lambda x: dens1ds[i](x),xmin,val,epsabs=1.e-4,limit=100)[0]
                more = sp.integrate.quad(lambda x: dens1ds[i](x),val,xmax,epsabs=1.e-4,limit=100)[0]
                fracless = less/tot
                fracmore = more/tot
                tailNsig= calc_1dfractail_to_sigma(min(fracless,fracmore))
                #tailNsig_twosides = calc_1dfractail_to_sigma(min(fracless,fracmore),singletail=False)
                outdat.append([val,fracless,fracmore,tailNsig])

            outdat = np.array(outdat)
            outdict[p]=outdat

        else:
            pass
            #print('skipping param')
    header = 'parameter   value 1dmargpost_frac_below 1dmargpost_frac_above equiv_Gaussian_Nsigma '
    return outdict, header



                    #fracf.write('{0:+0.4f}  {1:0.4f}  {2:0.4f}  {3:0.4f}\n'.format(xline,fracless,fracmore,tailNsig))

#--------------------------------------------------
def get_1d_stats_single(gdchain, chainfbase, statlist=['peak1d','lerr68','uerr68','lerr95','uerr95','mean','median','std','chainmin','chainmax'],outdir = 'output_fromchains/stats',outprefix='stats',smooth_in_fname=True, savefile=True,  substats = None, fnameonly=False,inkdesmooth=None,postprocnote=None,margpostcomps={}):
    """
    Given getdist chain object, outputs stat info for quantities in statlist
    for all available parameters

    if substats, this is an Nparam len array to be subtracted
    from all parameters to blind:

    if fnameonly, can do without chain object, just gets output filenames
    """
    #print("in cp, outdir=",outdir)
    if smooth_in_fname and (inkdesmooth is not None):
        smoothstr = '-smooth{0:d}'.format(int(inkdesmooth*100))
    else:
        smoothstr = ""

    #print('>>>>>in cp, postprocnote=',postprocnote)
    if postprocnote is not None:
        # can be used e.g. if we're applying a prior when reading the file
        if smoothstr:
            smoothstr = smoothstr + '-'+postprocnote
        else:
            smoothstr = postprocnote
    #print('smoothstr=',smoothstr)

    if fnameonly:
        #print("OUTNAME ONLY")
        # if smooth_in_fname and (inkdesmooth is not None):
        #     smoothstr = '-smooth{0:d}'.format(int(inkdesmooth*100))
        # else:
        #     smoothstr = ""
        outnames = [outdir+'/'+'.'.join([outprefix+smoothstr,chainfbase,s,'txt']) for s in statlist]
        if ('global' in statlist):
            globi = statlist.index('global')
            #no smooth string for global stats file
            outnames[globi] = outdir+'/'+'.'.join([outprefix,chainfbase,'globalstats','txt'])


        return outnames


    paramlist = [paraminfo.name for paraminfo in gdchain.getParamNames().names]
    #paramlist=['cosmological_parameters--omega_m']

    dens1ds = []
    for p in paramlist:
        print(" getting 1ds for p=",p)
        # if all the numbers are the same, skip the parameter
        pind = gdchain.paramNames.numberOfName(p)

        pvals = gdchain.samples[:,pind]
        if np.max(pvals)==np.min(pvals):
            #print("HERE")
            margdens = None
        else:
            try:
                margdens = gdchain.get1DDensity(p)
            except:
                #print("OR HERE")
                margdens = None
        #print('>>>',pind,p,margdens is None)
        dens1ds.append(margdens)
    #dens1ds = [gdchain.get1DDensity(p) for p in paramlist]

    Nparam = len(paramlist)
    Nstat = len(statlist)
    outdat = np.nan*np.ones((Nparam,Nstat))


    kdesmooth = gdchain.smooth_scale_1D
    if smooth_in_fname:
        smoothstr = '-smooth{0:d}'.format(int(kdesmooth*100))
    else:
        smoothstr = ""
    if postprocnote is not None:
        # can be used e.g. if we're applying a prior when reading the file
        if smoothstr:
            smoothstr = smoothstr + '-'+postprocnote
        else:
            smoothstr = postprocnoet


    if (inkdesmooth is not None) and (inkdesmooth!=kdesmooth):
        raise ValueError("Input kdesmooth {0} != value extracted from chain {1}".format(inkdesmooth,kdesmooth))


    blind = substats is not None
    if blind:
        blindstr = ", BLINDED"
    else:
        blindstr = ""

    if postprocnote is not None:
        if postprocnote=='mth10':
            postnote = '   mthermal<10eV'
        else:
            postnote = '   '+postprocnote
    else:
        postnote=''


    #print(">>>statlist",statlist)
    for i,s in enumerate(statlist):
        outname = '.'.join([outprefix+smoothstr,chainfbase,s,'txt'])
        outf = outdir+'/'+outname
        print("  Working on stat:",s)
        if s.startswith('chainm'):
            # can specify chainmax_[paramname] or chaimin_[paramname];
            #   will find values associated with that sample with max/min value of that parameter

            if 'chainmax' in s:
                mparam = s.replace('chainax_','') #name of param to max or min
                mpind = gdchain.paramNames.numberOfName(mparam)
                mpvals = gdchain.samples[:,mpind]
                useind = np.argmax(mpvals)

            elif 'chainmin' in s:
                mparam = s.replace('chainmin_','') #name of param to max or min
                mpind = gdchain.paramNames.numberOfName(mparam)
                mpvals = gdchain.samples[:,mpind]
                useind = np.argmin(mpvals)
            
            sdat = []
            for p in paramlist:
                pind = gdchain.paramNames.numberOfName(p)
                result = gdchain.samples[useind,pind]
                #if s.startswith('chainmax'):
                #    result = np.max(pvals)
                #elif s.startswith('chainmin'):
                #    result = np.min(pvals)
                header = 'parameter {}'.format(s)
                sdat.append(result)
            sdat = np.array(sdat)
            if len(outdat.shape)==1:
                outdat=np.atleast_2d(outdat).T
        else:
            sdat,header = comp_1d_stat(paramlist, dens1ds, s)

        if len(sdat.shape)==1:
            sdat = np.reshape(sdat,(sdat.size,1)) #set up so it is Nx1
        if blind:
            print('    BLINDED; storing difference with fid')
            sdat[:,0] = sdat[:,0] - substats
        outdat[:,i] = sdat[:,0]
        print("   ...saving output to ",outf)


        if savefile:
            f = open(outf,'w')
            f.write('## '+header + '  [kdesmooth 1D={0:f}{1:s}{2:s}]\n'.format(kdesmooth,blindstr,postnote))
            for i,p in enumerate(paramlist):
                f.write('{0:s}    '.format(p)+''.join(['{0:g}    '.format(sij) for sij in sdat[i,:]])+'\n' )
            f.close()
            print('   ...closed file')


    #print('about to handle margpost')
    if len(margpostcomps.keys()):
        s = 'margpostcomps'
        outname = '.'.join([outprefix+smoothstr,chainfbase,s,'txt'])
        outf = outdir+'/'+outname
        #print("  Working on stat:",s)
        outdict,header = comp_1d_margpostcomps(margpostcomps,dens1ds,paramlist,gdchain.ranges)
        #print('test for margpost')
        if len(outdict.keys()): #skip if relevant parameters aren't varied for this chain
            print('     writing to',outf)
            f = open(outf,'w')
            f.write('## '+header + '  [kdesmooth 1D={0:f}{1:s}{2:s}]\n'.format(kdesmooth,blindstr,postnote))
            for k in outdict.keys():
                for row in outdict[k]:
                    f.write('{0:s}    '.format(k)+''.join(['{0:g}    '.format(dat) for dat in row])+'\n' )


            f.close()
            print('   ...closed file')



    return paramlist, outdat

#--------------------------------------------------
def get_1d_stats_batch(inputflist, gdchains =None, statlist=['peak1d','lerr68','uerr68','lerr95','uerr95','mean','median','std'],outdir = 'output_fromchains/stats',outprefix='stats', addderived=['summnu','S8','asx1.e9'],indatdir = '', rangedict = None, kdesmooth=.3,smooth_in_fname=True, blind=True, substat = 'mean'):
    """
    Given input list of chain output files, stats to compute, computes and
    saves stat info about the marginalized posterios.

    If you pass a list of MCSamples objects
    instead of strings, will skip reading files and just use those chains.

    Goal is to make output similar to cosmosis postprocessing output.
    One outfile per input file & stat, contains stats for all params

    if blind==True, will assume the first input file is fiducial, will report all
    means, peak1d, median only as diferences from that fiducial case.
    """
    if indatdir and indatdir[-1]!= "/" :
        indatdir+='/'

    if gdchains is None:
        gdchains = [get_gdchain(inputf,kdesmooth= kdesmooth, indatdir = indatdir, rangedict = rangedict) for inputf in inputflist]
        # if file doesn't exist or isn't processable, has MISSINGDAT as entry

    infbases = []
    for f in inputflist:
        startind = f.rfind('/')+1
        endind = f.rfind('.txt')
        infbases.append(f[startind:endind])

    for p in addderived:
        if p=='summnu':
            add_mnu(gdchains)
        elif p=='S8':
            add_S8(gdchains)
        elif p=='asx1.e9':
            add_As_scaled(gdchains)
        elif p in ('cosmological_parameters--ommh2','cosmological_parameters--ombh2'):
            add_physical_densities(gdchains)

    if blind:
        fidparams, fidstats = get_1d_stats_single(gdchains[0],infbases[0],statlist=[substat],savefile=False)
        substats = fidstats[:,0]
        print("substats.shape",substats.shape)
    else:
        substats = None

    for i,chain in enumerate(gdchains):
        if chain==MISSINGDAT:
            continue
        paramlist, statdat = get_1d_stats_single(chain,infbases[i], statlist, outdir, outprefix, smooth_in_fname=smooth_in_fname, savefile=True, substats = substats)

#==================================================================
# Functions for reading in and plotting stat info
#==================================================================
def getstatfilebase(testkey='fidsim',runstr='d_l',prefix = 'stats-smooth03',statdir ='output_fromchains/stats',chainprefix='chain_',chainsuffix='.txt'):
    """
    THIS IS FOR COSMOSIS-like POSTPROCESSING OUTPUT
    given
    testkey - what input datavec is used? (fidsim, baryons, etc)
    paramkey - what parameter are we looking at
    indat - can be d, djp5b, etc
    returns string representing stat filename, with '[STATTYPE]' left in
    where peak1, lerr68, uerr69 might be.
    """
    if statdir is None:
        statdir = 'stats'

    if chainprefix and chainprefix[-1] not in ('_','-'):
        chainprefixstr = chainprefix+'_'
    else:
        chainprefixstr = chainprefix

    if chainsuffix:
        chainsuffixstr = chainsuffix.replace('.txt','')
        if chainsuffixstr and chainsuffixstr[0] not in ('_','-'):
            chainsuffixstr = '_'+chainsuffixstr

    fname = ''.join([prefix,'.',chainprefixstr,runstr,'_',testkey,chainsuffixstr,'.[STATTYPE].txt'])
    fbase = statdir+'/'+fname

    return fbase

def getdat_fromstatfile(fname,getparams=None):
    """
    THIS IS FOR COSMOSIS-like POSTPROCESSING OUTPUT
    Given filename and a list of cosmological parameters identified by
    the same string they're listed as in the files, returns a list
    of numbers, one per parameter.

    This is set up so that it can pull out multiple params at once,

    # if getparams is None, return all params
    This needs to be tested
    """
    if getparams is not None:
        getparamslc = [p.lower() for p in getparams]
        #getparamsuc = [p.upper() for p in getparams]
        paramnames = []
        haveparamslc = []
        for p in getparams: #assumes getparams is in cosmosis section--parameter format
            paramnames.append(p)
        Nparams = len(paramnames)
        pdict = {paramnames[i].lower():i for i in range(Nparams)}
        outnums = np.nan*np.ones(Nparams)
        count = 0
    else:
        paramnames = []
        outnums = []

    #print('>>>getparamslc',getparamslc)
    #print(pdict.keys())
    if not os.path.isfile(fname):
        return outnums,paramnames
    f = open(fname)
    #print(fname)
    for line in f:
        #print(line)
        if line.startswith('#'): #comment
            continue
        words = line.split()
        #if ('s8' in words) or ('S8' in words):
        #    print("-----> ",words)
        if getparams is not None:
            if (words[0].lower() in getparamslc) and (words[0].lower() not in haveparamslc):
                #print(words[0],checkstrings)
                haveparamslc.append(words[0].lower())
                outnums[pdict[words[0].lower()]] = float(words[1])
                count +=1
                if count == Nparams:
                    break
        else:
            paramnames.append(words[0])
            outnums.append(float(words[1]))
    f.close()

    if getparams is None:
        outnums = np.array(outnums)

    return outnums, paramnames


#====================
def get_covmat(chainfile, outfile, weightname = 'weight',lastparam='COSMOLOGICAL_PARAMETERS--SIGMA_8',save=True):
    pdict =get_paramdict(chainfile)


    nsamp = get_nsample(chainfile)
    alldat = np.loadtxt(chainfile)[-nsamp:,:]
    weights = alldat[:,pdict[weightname]]
    lastcol = pdict[lastparam]
    dat = alldat[:,:lastcol+1]
    cov = np.cov(dat,rowvar=False,aweights=weights)

    if save:
        f = open(chainfile,'r')
        firstline = f.readline().replace('#','').split()
        f.close()
        paramheader = '#'+' '.join(firstline[:lastcol+1])

        print("Writing cov to ",outfile)
        np.savetxt(outfile,cov,header=paramheader)

    return cov



#==================================================================
# functions for getting info for individual chains which will get used for tension/modelcomp
#==================================================================
# want: logZ, D, d
# DIC = -2 ln L(theta_MAP) + 2 p_DIC, where p_DIC = 2 ln L(theta_MAP) - 2 <ln L>
# REMOVED WAIC (calc is not right, thanks to Noah W for figuring that out)
# logZ, chisq_MAP

def calc_1dfractail_to_sigma(probfrac,singletail=True):
    """
    Given fraction of a 1d marginalized probability distrubution, return the
    equivalent N such that Nsigma from mean of Normal distribu would have same single tail prob

    if singletail=False, instead find N such that Nsigma has frac outside the Nsigma range
    of a 1d normal distribution
    """

    if singletail:
        if probfrac>0.5:
            tailfrac=1.-probfrac
        else:
            tailfrac = probfrac
        mult=2

    else:
        tailfrac=probfrac
        mult=1
    #Number of sigma that would give same probability tail for Normal distribution
    return np.sqrt(2)*erfcinv(tailfrac*mult)


# #==================================================================
# # functions for doing bayesian suspiciousness calcs for tension + model comp
# #==================================================================

def calculate_Stension_probability(logS, bmd,singletailsigma=False, lowerq = 0.16, upperq = 0.84):
    """
    copy of function used for y1 ggsplit analysis

    From Pablo, for model and tension tests;  S is suspiciousness,
    bmd is bayesian model dimensionality. uses fact that bmd-2logS
    approximately follows a chisq distribution

    ALWAYS ASSUMES THAT MORE NEGATIVE logS means more tension
    i.e. if evidence ratio that went into this was [unsplit params]/[split params]
    more negative logS means we prefer the [split params] model.
    the p-value reports thte chance that we'd find that small of a logS
    if the model were really [unsplit params];so, small p values -> amount of tension

    If we're using this for model comparison, think carefully about sign. e.g.
    you may be looking at R=[extension]/[lcdm], but if you want small p values
    here to indicate tension with lcdm, you need to flip the sign of S to match R=[lcdm]/[extension]

    if singletailsigma=True, Nsigma corresponds to p as single-tail probability
    if not, uses double-tail prob for normal distribution

    Returns tension probability  mean, tension prob sdt, as well as
    equivalent number of sigma mean and std
    """
    #d-2*logS
    d_2logS = bmd - 2*logS

    #Tension probability
    p = chi2.sf(d_2logS,bmd)
    if not np.all(np.isnan(p)):
        p[negative]=1.
        #    p[np.isnan(p)] = 1. # S is so positive that d-2logS<0; prefer base model over split/extension
    else:
        p = np.array([])
    if p.size:
        pcentral = np.median(p)
        p_lowerq = np.quantile(p,lowerq)
        p_upperq = np.quantile(p,upperq)
        if singletailsigma:
            mult=2
        else:
            mult=1
        #Number of sigma
        #ns = np.sqrt(2)*erfcinv(p*mult)
        ns_from_pcentral = (np.sqrt(2)*erfcinv(pcentral*mult))

        return pcentral,p.std(),p_lowerq,p_upperq,ns_from_pcentral #,ns.std()
    else:
        return np.nan,np.nan,np.nan,np.nan,np.nan #,np.nan


def get_anest_output_manual(chainfname, chainprefix = 'chain_',chainsuffix='.txt',usepost=False, Nlive=None,paramlabels = DEFAULT_PLABELS):
    """
    chainfname  - full path to chain file
    pcdir - if pc files are in a subdirectory of where the chain is, put that name here

    Get anesthetic Nested Sampling object from chain file using
    method figured out by Noah Weaverdyck, and compute output stats.
    Noah has tested that this gives the same results as the live/dead point readin
    at least to within sampling error

    (avoids need for extra polychord output files)

    follows a similar read-in structure to get_gdchains

    if usepost: will pass logpost instead of logL to NestedSamples init
    if not usepost, will do loglike
    """
    if (not os.path.isfile(chainfname)):
        return None

    try:
        pdictf,plist =get_paramdict(chainfname,True)
        #print('plist=',plist)
        wantcols = []
        uniquep = []
        for p in plist:
            if p.lower() not in uniquep:
                uniquep.append(p.lower())
                wantcols.append(pdictf[p])
                # if there are duplicate column names, want the one furthest left
            else:
                #print(' skipping duplicate',p)
                pass
        #print('initial pdictf',pdictf)
        pdictf = {uniquep[i]:i for i in range(len(uniquep))}
        #print(len(uniquep),'pdictf',pdictf)
        plist = uniquep
        #print(wantcols)
        #dataf = np.loadtxt(chainfname)
        #columns=plist
        datachain = pd.read_csv(chainfname,comment='#', delim_whitespace=True, names=uniquep, usecols = wantcols)
        columns = datachain.columns
        dataf = datachain.to_numpy()
        if 'old_post' in pdictf.keys():
            isIS = True # this is importance sampling ouutput
            nsampf = dataf.shape[0]
        else:
            isIS = False
            nsampf = get_nsample(chainfname)
    #else:
    except:
        print("FILE EXISTS BUT SOMETHING WENT WRONG FOR:",chainfname)
        return None
    needweight=False

    if isIS:
        if 'weight' not in plist: # need to computed based on IS output
            needweight=True
            windf_old = pdictf['old_weight']
            windf_is = pdictf['log_weight']
            #print('windf_old',windf_old,'windf_is',windf_is,len(plist))
            #print(' chechding windf_is entry in plist:', plist[windf_is])
            #print(' 2dataf.shape',dataf.shape)
        else:
            windf=pdictf['weight']
        if Nlive is None:
            print(" on",chainfname)
            print("Need Nlive for IS runs")
            return None
    else:
        windf =  pdictf['weight']
        Nlive = get_nlive(chainfname)


    postindf = pdictf['post']
    likeindf = pdictf['like']
    #priorindf = pdictf['prior']

    sampledat = dataf[-nsampf:,:]
    if sampledat.shape[0]==0:
        return None
    #print('chainfname=',chainfname)
    if isIS and needweight:
        print("ASSESSING IS WEIGHTS",nsampf,windf_old,chainfname)
        oldweights = dataf[-nsampf:,windf_old]
        nonzeromask = np.logical_and((oldweights!=0), np.isfinite(oldweights))
        # ^ using this helps avoid some points where wexp is inf;
        # all points with oldweights=0 will also have new weights =0
        #print('datafline whatever',dataf[9128,:])
        isweights = dataf[-nsampf:,windf_is]
        if  np.all(oldweights==0.):
            print(' ..but all weights are zero?')
            return None
        if np.all(np.isnan(oldweights)):
            print(' ..but all old weights are nan?')
            return None
        wexp =np.exp(isweights[nonzeromask])
        weights = np.zeros(oldweights.size)
        weights[nonzeromask] = oldweights[nonzeromask]*wexp
        datshape = sampledat.shape
        datachain.assign(weight=weights)
        dataf = datachain.to_numpy()
        sampledat = dataf[-nsampf:,:]
        windf = len(columns)-1
        pdictf['weight']=windf
    else:
        weights = dataf[-nsampf:,windf]


    if usepost:
        logLarg = sampledat[:,postindf]
    else:
        logLarg = sampledat[:,likeindf]

    Nlive=int(Nlive)
    #print('isIS',isIS,'Nlive',Nlive,logLarg.shape,len(plist),sampledat.shape)


    outputobj = NestedSamples(data=sampledat, columns = plist, logL=logLarg, logL_birth = Nlive).ns_output()
    # From Noah:
    #ns_out_bl = an.NestedSamples(data=chain_df[get_params2plot(param_labels)].to_numpy(), columns=get_params2plot(param_labels), logL=chain_df[like_col], logL_birth=nlive).ns_output)(

    return outputobj

def get_anest_output_pcfiles(chainfname, pcprefix = 'pc_',pcdir='pcfiles',chainprefix = 'chain_',chainsuffix='.txt', crashforerrors=False):
    """
    chainfname  - full path to chain file
    pcdir - if pc files are in a subdirectory of where the chain is, put that name here

    Get anesthetic Nested Sampling object, and compute output stats.
    Requires polychord output files beyoond just the main chain file:
    this follows setup in anesthetic tutorials, example by Pablo
    """
    if pcdir:
        pcdirstr = pcdir+'/'
    else:
        pcdirstr =''
    pcfbase = chainfname.replace(chainprefix,pcdirstr+pcprefix).replace(chainsuffix,'_')
    #print("pcfbase",pcfbase)

    if crashforerrors:
        samplesobj = NestedSamples(root=pcfbase)
        outputobj = samplesobj.ns_output()
    else:
        try:
            samplesobj = NestedSamples(root=pcfbase)
            outputobj = samplesobj.ns_output()
        except:
            #print("Skipping. Error reading anest output for",chainfname)
            outputobj = np.nan

    return outputobj

def get_Stension_info_modcomp(anestoutbase,anestoutextended):
    """
    Tension info for model comp! first arge is baseline model, second is extended model
    Base model parameters should be a subspace of extended model's:
    i.e. this computes tension of extended model constraints with base model's subspace

    Given two anesthetic nested sampler output objects, computes
    Bayesian suspiciousness to compute tension probability
    returns S, tension prob, prob std, number of sigma, prob std
    """
    if (type(anestoutbase)==int) or (type(anestoutextended)==int):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # # LCDM in numerator
    # logR = + anestoutbase.logZ - anestoutextended.logZ
    # logI =  -1*anestoutbase.D + anestoutextended.D
    # bmd = -1*anestoutbase.d + anestoutextended.d

    # extension in numerator
    logR =  anestoutbase.logZ - anestoutextended.logZ
    logI = -1* anestoutbase.D + anestoutextended.D
    # dimensionality we're checking for tensions is the number of extra params
    bmd = -1*anestoutbase.d + anestoutextended.d
    #print('basebmd',np.array(anestoutbase.d))
    #print('extebmd',np.array(anestoutextended.d))
    logS = logR - logI

    p,pstd,plow1sig,pupp1sig,sigofp = calculate_Stension_probability(logS,bmd)

    return logR, logI, bmd, logS,p,pstd,plow1sig,pupp1sig,sigofp

def get_Stension_info_datacomp(anestout1,anestout2,anestoutcombo):
    """
    Tension info for data comp!
    """
    if (type(anestout1)==int) or (type(anestout2)==int) or (type(anestoutcombo)==int):
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    logR = anestoutcombo.logZ - anestout1.logZ - anestout2.logZ
    logI =  - anestoutcombo.D + anestout1.D + anestout2.D
    bmd =  - anestoutcombo.d + anestout1.d + anestout2.d
    logS = logR - logI
    #print('making plot for',anestoutcombo)
    p,pstd,plow1sig,pupp1sig,sigofp = calculate_Stension_probability(logS,bmd)

    return logR,logI,bmd,logS,p,pstd,plow1sig,pupp1sig,sigofp

#==================================================================
# various information criterion model comparison metrics
#==================================================================

# REMOVING WAIC, THIS CALC ISN"T RIGHT
# def get_waic(weights,loglikes):
#     """
#     Given weights and likelihoods of chain samples, compute WAIC
#     WAIC = Watanabe-Akaike information criterion aka widely applicable info crterioin
#     (bayesian generalization of DIC which doesn't depend on point estimates
#     Adapting code from Noah Weaverdyck

#     optional to pass in max_loglike, in case we've gotten it form
#     maxlike sampler instead of directly form chain
#     """
#     max_loglike = np.max(loglikes) #just usesed to help with averaging
#     loglikes_shift = loglikes - max_loglike
#     p_waic = 2* (np.log(np.average(np.exp(loglikes_shift), weights=weights)) + max_loglike) - 2*np.average(loglikes, weights=weights)
#     return -2 * (np.log(np.average(np.exp(loglikes_shift), weights=weights)) + max_loglike) + 2*p_waic

def get_dic(weights,loglikes,map_loglike):
    """
    Given weights, loglikes, and optional max loglike, computes DIC
    DIC = deviance information criterion

    Note that sometimes the loglike of the mean parameters
    is used instead of the map.
    """

    p_dic = 2* map_loglike -  2*np.average(loglikes, weights=weights)

    return -2* map_loglike + 2*p_dic

def get_2xavglike(weights,loglikes):
    return  2*np.average(loglikes, weights=weights)




# CITE Liddle 2007 https://arxiv.org/pdf/astro-ph/0701113.pdf
def get_aic(map_loglike,k=np.nan):
    """
    AIC = -2*log_like_max + 2*k
    k = number of parameters
    """
    if   np.isnan(k):
        return np.nan
    return -2*map_loglike + 2*k


def get_bic(map_loglike, k=np.nan, N=np.nan):
    """
    BIC = -2*log_like_max + k*logN
    k = number of parameters
    N = number of datapoints used to fit
    """
    if  np.isnan(k):
        return np.nan
    if  np.isnan(N):
        return np.nan
    return -2*map_loglike +k*np.log(N)


#==================================================================
# Parameter difference tension metrics
#   Never got this fully working for Y3ext, if interest could implement
#   for Y6
#==================================================================

# def parameter_shift(gdchain1,gdchain2,paramlist=None,skipparams=[],feedback=1):
#     """
#     Given two getdist MCSamples objects and list of parameters
#     use Marco's tensiometer code to compute parameter difference
#     and assess probability of tension in given set of parameter directions


#     Note that difference will only be taken in shared parameter directions.
#     Could just do all parameters, but note that e.g. for A_s very small
#     numbers might cause problems.

#     if paramlist is passed, will compare all params in that list that
#     appear in both chain1 and chain2

#     if paramlist is not passed, will compare all params that appear in both
#     chains, except parameters listed in skipparams
#     """
#     if paramlist is None:
#         if skipparams:
#             useparams = []
#             params1= [p.name for p in gdchain.getParamNames()]
#             params2= [p.name for p in gdchain.getParamNames()]
#             inbothset = set(params1) & set(params2)
#             useparams = list(inbothset.difference(set(skipparams)))
#         else:
#             useparams= None # just use all params
#     else:
#         useparams = paramlist



#     # copy settings from other chains
#     settings = {'smooth_scale_1D': gdchain1.smooth_scale_1D, \
#                 'smooth_scale_2D': gdchain1.smooth_scale_2D,
#                 'boundary_correction_order': gdchain1.boundary_correction_order, \
#                 'mult_bias_correction_order': gdchain1.mult_bias_correction_order}
#     diffchain = mcmc_tension.parameter_diff_chain(gdchain1,gdchain2)
#     diffchain.updateSettings(settings)
#     diffchain.updateBaseStatistics()
#     shift_P, shift_low, shift_hi = mcmc_tension.flow_parameter_shift(diffchain,useparams
#     sigofp = tensiometer.utilities.from_confidence_to_sigma(shift_P)


#==================================================================
# sigma8(z) plots (can be used more generally if needed)
#==================================================================
def plot_func_bands(xvals,ybandmax,ybandmin,labels,shadecols,outname,ymeans=None,ymeds = None,meanls='--',medls = '-',linecols=None,shadealpha=0.6,ylabel=None,xlabel=None,ylims=None,xlims=None,figwidth=3.4,legfontsize=10,legtitle='',ftype='pdf',outdir='.',graytext='',axes_fontsize=8, lab_fontsize=8, linewidths=1,legloc=None,figratio=0.8,vlines=[],hlines=[],dology=False,dologx=False,edgecols=None,hatches=None,bandlw=None,markers=None,markerzs=None,markersizes=None,capsizes=None,filled=None,xoffsets=None,legncol=1):
    """
    Plot series of bands for a function.
    Default assumption is sigma8(z), but trying to keep the setup general

    Need list or array of arrays
    xarrays - sets of x values (e.g. z) going with each band to plot
              note that e.g. npg has different length of an array here than lcdm
    ybandmax - list of arrays defining upper bound of shaded regions
    ybandmin - " lower bound
    labels = how to label colored bands in legend
    ymeans, ymed - these will be plotted as lines using line styles meanls and medls,
                   linecol color will be used
    ylabel, xlabel - labels for plot axes, defaults to sigma8(z)
    ylims,xlims used to set plot range. defaults to whatever is automatc

    """
    Nbands = len(labels)
    if ylabel is None:
        ylabel=r'$\sigma_8(z)$'
    if xlabel is None:
        xlabel = r'$z$'
    if linecols is None:
        linecols = shadecols.copy()
    if type(shadealpha)!=list:
        shadealpha = [shadealpha]*Nbands
    if type(linewidths)!=list:
        linewidths = [linewidths]*Nbands
    if edgecols is None:
        edgecols = ["none"]*Nbands
    elif type(edgecols)!=list:
        edgecols = [edgecols]*Nbands
    if hatches is None:
        hatches = [None]*Nbands
    elif type(hatches)!=list:
        hatches = [hatches]*Nbands
    if bandlw is None:
        bandlw = [0]*Nbands
    elif type(bandlw) != list:
        bandlw = [bandlw]*Nbands

    if (markers is None) or (markerzs is None):
        domarks = False
        markers = [False]*Nbands
    else:
        firsttype = markerzs[0]
        if np.isscalar(firsttype): # same z list for all sets of points
            mzs = [markerzs for i in range(Nbands)]
        else:
            mzs = markerzs
        if markersizes is None:
            markersizes = [4]*Nbands
        if capsizes is None:
            capsizes = [0]*Nbands
        if filled is None:
            filled = [True]*Nbands
        if xoffsets is None:
            xoffsets = [0.]*Nbands


    if type(xvals)==np.array and len(xvals.shape)==1: # just one array, use same one for all bands
        xvals = [xvals]*Nbands


    fig,ax = plt.subplots(1,1,figsize=(figwidth,figwidth*figratio))
    #plt.rcParams.update({'axes_fontsize':axes_fontsize})
    ax.tick_params(axis='x',which='both',bottom=True,top=True,direction="in")
    for i in range(Nbands):
        if markers[i]: # plot marker instead of band
            continue
        elif (ybandmax[i] is not None) and (ybandmin[i] is not None):
            plt.fill_between(xvals[i],ybandmax[i],ybandmin[i],facecolor=shadecols[i],label=labels[i],alpha=shadealpha[i],lw=bandlw[i],hatch=hatches[i], edgecolor=edgecols[i])
    # lines go on top
    for i in range(Nbands):
        if ymeans is not None:
            if ymeans[i] is not None:
                if markers[i]:
                    mxi = np.array(mzs[i])+xoffsets[i]
                    yi  = interp1d(xvals[i],ymeans[i])(mxi)
                    yerrmin = interp1d(xvals[i],ybandmin[i])(mxi)
                    yerrmax = interp1d(xvals[i],ybandmax[i])(mxi)
                    yerr = np.array([yi-yerrmin,yerrmax-yi])
                    if filled[i]:
                        facecol=None
                    else:
                        facecol='none'
                    plt.errorbar(mxi,yi,yerr=yerr,marker=markers[i],color=linecols[i],mfc=facecol,markeredgewidth=linewidths[i],elinewidth=linewidths[i],ls='none',lw=linewidths[i],markersize = markersizes[i],label=labels[i],capsize=capsizes[i])
                else:
                    plt.plot(xvals[i],ymeans[i],ls=meanls,color=linecols[i],lw=linewidths[i])
        if ymeds is not None:
            if ymeds[i] is not None:
                plt.plot(xvals[i],ymeds[i],ls=medls,color=linecols[i],lw=linewidths[i])

    leg=ax.legend(loc=legloc,fontsize=legfontsize,title=legtitle,ncol=legncol)
    leg.get_title().set_fontsize(legfontsize)
    ax.set_xlabel(xlabel,fontsize=lab_fontsize)
    ax.set_ylabel(ylabel,fontsize=lab_fontsize)
    plt.xticks(fontsize=axes_fontsize)
    plt.yticks(fontsize=axes_fontsize)
    for x in vlines:
        plt.axvline(x,color='gray',ls=':',lw=1)
    if dology:
        ax.set_yscale("log",nonposy='clip')
    if dologx:
        ax.set_xscale("log",nonposx='clip')
    if ylims is not None:
        ax.set_ylim(ylims)
    if xlims is not None:
        ax.set_xlim(xlims)
    if graytext:
        fig.text(0.1, 0.3, graytext, color='gray', alpha = .4,\
                 fontsize= 20, wrap=True, ha="left", va="center")

    outf = outdir+'/'+outname+'.'+ftype
    print("Saving plot to ",outf)
    plt.tight_layout()
    plt.savefig(outf,dpi=300)
    plt.clf()
    plt.close()


#==================================================================
def table_plot(outname, statdict, xparams, ykeys, ylabels, pointdicts, xlabels=None, centerstat='mean', lowerstat='lerr68', upperstat='uerr68', outdir='.', graytext='', figwidth=3.4, ftype='pdf', xtickdict=None, xrangedict=None, paramlabels = DEFAULT_PLABELS, hlines = None, vlines = None, vbands=None, yfontsize=12., xfontsize=12., legfontsize=10., ticksize=10., savediffs=True, savevals=False, vbandalpha=0.1, errors_arediffs=False, topheader=None, topheadfontsize=12,legncol=1,legloc="upper left",legkwargs={},hgrid=True, logxfor= [],fidzero=True, flip=False,heightperparam=1,heightperline=None,hideleg=False,highlightdiffs=False,diffcut=0.3,centered_labels=False):

    """
    General function for making table plots of either individual parameters
    or summary statistics. This was used for all parameter summary plots,
    as well as the model comparison and tension tables in Y3extKP

    xparams - list of parameter names, each will get a panel
              showing that param
    xlabels - list matching length of parameter names, used to label axes
              if not passed, will use paramlables to set the names

    ykeys - keys to identify rows
    ylabels  - text label for rows of points (should match ykeys in lenght)

    pointdicts -list of dictionaries, one for each set of points we want to plot
            should have entry for
            color
            label - text label to idenfity  marker in legend
                     (if _nolegend_ or empty, won't show)
            difflabel - text label to label set in output savediffs file
            xparams (if None or "all") plot on all panels
                    otherwise just show in panels where x param is in this list
            ykeydict - dictionary with keys matching ykeys, values matching
                    keys needed for (dattag,model,endtag) in statdict
            fidykey - if a ykey is passed, will shade error region.
                    if not there, will assume fid is first ykey; if None won't do shading
            offset - fractional offset from row (useful for showing multiple points)
            can also add in some marker style info

    vlines - list of either dictionaries or numbers that will add vertical lines
             if numbers are passed, will get plotted on all panels as gray dashed lines
             if dictionaries, there is one dictionary per line. it needs keys:
                value - number where to put line
                params - what parameters to plot this for. can say 'all' for all params
                kwargs - optional, kist of kwargs for line formatting. if not there
                         will default to dashed gray lines
    vbands - list of dictionaries for plotting vertical bands using axvspan. args
                min - lower bound of shaded region
                max - upper bound of shaded region
                params - what parameters to plot this for. can say 'all' for all params
                color - what color should the band be? defaults to gray
                alpha - set transparanency


    statdict - nested dictionary containing summary stat info
                [chainkey][parameter][stat]
          chainkeys could be e.g. the chain file name, or could be some other thing
          as long as the chain keys here match the ykeydict values in pointdict

    centerstat - what stat is the center point. note that if an xparam is a global
                stat, this will be ignored. e.g. if we're plotting logZ will set
                centerstat='logZ'
    lowerstat - stat for left errorbar
    upperstat - stat for right error bar
    errors_arediffs - if this is True, assumes lowerstand and upperstat should be
           added or subtracted from the centerstat. in this case,
           if lowerstat isn't already negative, will multiply by -1

    xtickdict - if passed can choose values of ticks shown for each parameter
    xrangedict - if passed, sued to set xranges shown for each param

    graytext - watermark

    if savediffs, will save an output text file reporting differences
    with fid for each parameter and pointset

    logxfor - if a key is in this list, will use a log scale for the plot

    fidzero - if the is true, only plot and report differences from baseline mean
             (can be used for pre-unblinding robustness checks)

    flip = if True, makes params columsn and variations rows (function of x and y settings will stay the same)
          > ykeys, ylabels etc will be used to label columns, different sets of points
          > xparams is parameters plotted for each row
          > hlines will conterintuitively plot vlines, and vice versa
          (not the tidiest or clearest, but this is the simplest way to get this set up for now)
    heightperparam - only used if flip=True, height of each panel
    centered_labels  - only used if flip=True; if False, labels will be rotated to 45 deg, if True, print horizontally
    """
    if type(xparams)==list:
        Npanels = len(xparams)
    else:
        xparams = [xparams]
        Npanels=1

    if xlabels is None:
        xlabels = [paramlabels[x] for x in xparams]

    # This won't get used for parameter shifts,
    # but if we're showing different model comparison stats, may need this
    if type(centerstat)!=list:
        centerstat=[centerstat]*Npanels
    if type(lowerstat)!=list:
        lowerstat=[lowerstat]*Npanels
    if type(upperstat)!=list:
        upperstat = [upperstat]*Npanels
    if type(errors_arediffs)!=list:
        errors_arediffs = [errors_arediffs]*Npanels


    # figure out legend entries, how much space is needed for legend
    # while going through pointsets also make sure needed info is there
    legend_elements = []

    for pointset in pointdicts:
        plabel= pointset['label']
        #print('test',plabel)
        if 'difflabel' not in pointset.keys():
            pointset['difflabel']=plabel
        if 'diffonly' not in pointset.keys():
            pointset['diffonly']=False
        if 'fidykey' not in pointset.keys():
            # this will be the point that defines a shaded region
            pointset['fidykey'] = ykeys[0]
        if pointset['fidykey'] is not None:
            pointset['fidyind'] = ykeys.index(pointset['fidykey'])
        else:
            pointset['fidyind'] = None
        if 'shadefid' not in pointset.keys():
            pointset['shadefid'] = True
        if 'offset' not in pointset.keys():
            pointset['offset']= 0.
        if 'params' not in pointset.keys():
            pointset['params']='all'
        if 'marker' not in pointset.keys():
            pointset['marker']='o'
        if 'markersize' not in pointset.keys():
            pointset['markersize'] = 4
        if 'markerfacecolor' not in pointset.keys():
            pointset['markerfacecolor']=pointset['color']
        if 'markeredgewidth' not in pointset.keys():
            pointset['markeredgewidth'] = 1
        if 'markeredgecolor' not in pointset.keys():
            pointset['markeredgecolor'] = pointset['color']
        if (not hideleg) and (plabel is not None) and plabel and (plabel not in ["_nolegend_"]):
            element = matplotlib.lines.Line2D([0],[0],marker=pointset['marker'],markerfacecolor=pointset['markerfacecolor'],markeredgecolor=pointset['markeredgecolor'],markersize=pointset['markersize'],color=pointset['color'],label=plabel)
            legend_elements.append(element)
    Nleglines = len(legend_elements)

    # assoiate ykeys with yvalues
    Nrows = len(ykeys)
    if not flip:
        yvals = np.arange(Nrows)[::-1] #baseline on top
    else:
        yvals = np.arange(Nrows) #order flips to put baseline on left
    #yvals = {ykeys[j]:Nrows-j-1 for j in range(Nrows)}

    # figure out fig height and margins
    if not flip:
        if hideleg:
            topmarg =0.0#inches
        else:
            #print(">>>>",Nleglines,legncol,np.ceil(Nleglines/legncol))
            topmarg =(np.ceil(Nleglines/legncol)+1.2)*(legfontsize/72)*1.2#inches allowing for legend
        botmarg = xfontsize/72*2
        if heightperline is None:
            heightperline = yfontsize/72*2
        figheight = topmarg+botmarg + heightperline*(Nrows+.5)

        #set up figure
        fig,axes = plt.subplots(1,Npanels,sharey=True,figsize = (figwidth,figheight))
    else:
        legheight = (np.ceil(Nleglines/legncol)+1.2)*(legfontsize/72)*1.2 #inches allowing for legend
        titleheight = 0
        #stack legend and title
        if topheader is not None:
            titleheight = topheadfontsize/72*1.2
        topmarg =legheight+titleheight
        botmarg = yfontsize/72*2 # not quite right, but doesn't really matter
        heightperline = heightperparam
        figheight = topmarg+botmarg + heightperline*(Npanels+.5)

        #set up figure
        fig,axes = plt.subplots(Npanels,1,sharex=True,figsize = (figwidth,figheight))


    #print('>>>TOPMARG',topmarg,legfontsize/72)
    plt.subplots_adjust(wspace=0.0,hspace=0.0,left=.0,top=1.-topmarg/figheight)
    rcParams['axes.linewidth']=1
    ylowerpad = 0.5
    yupperpad = 0.5
    ymin = -1*ylowerpad
    ymax = Nrows - 1 + yupperpad
    if not flip:
        plt.ylim((ymin,ymax))
    else:
        plt.xlim((ymin,ymax))

    if savediffs:
        difflines = []
        difflines.append('Columns are: Nsig(quad-asym-sig) Nsig(fidstd)\n')
    if savevals:
        vallines = []
        vallines.append('Columns are center stat, lower error, upper error, width of errors, uerr-center, center-lerr \n')
    # loop through panels
    for i in range(Npanels):
        centerstati=centerstat[i]
        lowerstati=lowerstat[i]
        upperstati=upperstat[i]
        errors_arediffsi = errors_arediffs[i]

        if Npanels>1:
            ax = axes[i]
        else:
            ax= axes
        inparam = xparams[i]
        if '.' in inparam:
            param = inparam[inparam.find('.')+1:]
        else:
            param = inparam

        # add xlabel
        if not flip:
            if fidzero:
                ax.set_xlabel(r'$\Delta$'+xlabels[i],size=xfontsize)
            else:
                ax.set_xlabel(xlabels[i],size=xfontsize)
            if inparam in logxfor:
                ax.set_xscale('log')
        else:
            if fidzero:
                ax.set_ylabel(r'$\Delta$'+xlabels[i],size=xfontsize)
            else:
                ax.set_ylabel(xlabels[i],size=xfontsize)
            if inparam in logxfor:
                ax.set_yscale('log')

        # if i==0 and topheader is not None: #if we have a plot title, add it
        #     ax.annotate(topheader,xy=(.98, 1.01), xycoords = 'axes fraction',\
        #                 color='gray', weight = 'bold', fontsize=topheadfontsize, ha='right',va='bottom')

        if savediffs:
            difflines.append("\n=================\n"+param)
        if savevals:
            vallines.append("\n=================\n"+param)





        # for each panel, identify pointdicts that should be plotted
        for pointset in pointdicts:
            if not ((pointset['params']=='all') or (inparam in pointset['params'])):
                # don't plot this set of points in this panel
                continue
            # if we want to plot this set, get the info out of the statdict
            xvals = np.nan*np.ones(Nrows)
            xstds = np.nan*np.ones(Nrows)
            xerr = np.nan*np.ones((2,Nrows))

            if fidzero:
                fidind=pointset['fidyind']
                if fidind is None:
                    raise ValueError("cant zero out fiducial value if we don't have fidind")
                else:
                    fidykey = ykeys[fidind]
                    usefidykey = pointset['ykeydict'][fidykey]
                    if usefidykey not in statdict.keys():
                        continue
                    if type(statdict[usefidykey][param])==dict:
                        subval = statdict[usefidykey][param][centerstati]
                    else:
                        subval= statdict[usefidykey][param]
                    if subval==np.nan: #can't plot if we can't figure out what to subtract
                        print('fidzero: missing subval for',fidykey,usefidykey,param)
                        continue
            else:
                subval = 0.

            for j,ykey in enumerate(ykeys):
                #print(pointset.keys())
                #print(j,ykey,pointset['ykeydict'].keys())
                if ykey not in pointset['ykeydict'].keys():
                    continue
                useykey = pointset['ykeydict'][ykey]

                if useykey not in statdict.keys():
                    #print('couldnt find',useykey,'in',statdict.keys())
                    continue

                if type(statdict[useykey][param])==dict:
                    #print('usekey=',usekey,'param=',param)
                    #normal param value
                    # statdict has keys with all lowercase letters
                    xj = statdict[useykey][param][centerstati]
                    if lowerstati is None:
                        xjl = np.nan
                    else:
                        xjl = statdict[useykey][param][lowerstati]
                    if upperstati is None:
                        xju = np.nan
                    else:
                        xju = statdict[useykey][param][upperstati]
                    if 'std' in statdict[useykey][param].keys():
                        xjstd= statdict[useykey][param]['std']
                    else:
                        xjstd = np.nan
                else:   #global stat
                    #print("global stat, using center",param)
                    #print("   errs are",lowerati,upperstati)
                    xj= statdict[useykey][param]
                    if lowerstati is None:
                        xjl = np.nan
                    else:
                        xjl = statdict[useykey][lowerstati]
                    if upperstati is None:
                        xju = np.nan
                    else:
                        xju = statdict[useykey][upperstati]
                    if 'std' in statdict[useykey].keys():
                        xjstd= statdict[useykey]['std']
                    else:
                        xjstd = np.nan
                if np.nan in [xj,xjl,xju]:
                    if fidzero: #set all to np.nan to avoid accidental unblinding
                        xj = np.nan
                        xjl = np.nan
                        xju = np.nan
                        continue
                    else:
                        if xj==np.nan:
                            xj = np.nan
                        if xjl==np.nan:
                            xjl = np.nan
                        if xju==np.nan:
                            xju= np.nan
                else:
                    xj  -= subval
                    xju -= subval
                    xjl -= subval
                xvals[j]=xj
                if xjstd==np.nan:
                    xjstd= np.nan
                xstds[j]=xjstd
                if errors_arediffsi:
                    xerr[1,j]=xju
                    if xjl>0:
                        xerr[0,j]=xjl
                    else:
                        xerr[0,j]=-1*xjl
                else:
                    xerr[0,j]=xj-xjl
                    xerr[1,j]=xju-xj
            #for j in range(Nrows):
            #    print(xvals[j],xerr[0,j],xerr[1,j])

            # plot shaded region for fid point
            fidind=pointset['fidyind']
            fidval = np.nan
            if fidind is not None:
                fidval = xvals[fidind]
                fidlerr = fidval - xerr[0,fidind]
                fiduerr = fidval + xerr[1,fidind]
                if not np.isnan(fidval) and pointset['shadefid'] and (not pointset['diffonly']):

                    if not flip:
                        ax.axvspan(fidlerr,fiduerr,alpha=vbandalpha,edgecolor=None,facecolor=pointset['color'],zorder=-50) #zorder makes sure this goes under points
                        ax.axvline(fidval,color=pointset['color'],zorder=-40,lw=1,alpha=min(1,vbandalpha+0.2)) #zorder makes sure this goes under points
                    else:
                        ax.axhspan(fidlerr,fiduerr,alpha=vbandalpha,edgecolor=None,facecolor=pointset['color'],zorder=-50) #zorder makes sure this goes under points
                        ax.axhline(fidval,color=pointset['color'],zorder=-40,lw=1,alpha=min(1,vbandalpha+0.2)) #zorder makes sure this goes under points





            # if savediffs, compute diffs and add lines to difflines
            highlights = np.zeros_like(xvals, dtype=bool)
            #print("1.>>>",j,ykey,fidval,fidind)

            if savevals:# and not (np.isnan(fidval) or (fidind is None)):
                vallines.append("  ------\n  "+pointset["difflabel"]+"")
                for j in range(Nrows):
                    centerval = xvals[j]
                    lerr = centerval - xerr[0,j]
                    uerr = centerval + xerr[1,j]
                    if np.isnan(lerr) and np.isnan(uerr):
                        vallines.append("    {0:0.5f}  {1:s}".format(centerval, str(ykeys[j])))
                    else:
                        vallines.append("    {0:0.5f}   {1:0.5f} {2:0.5f} {3:0.5f}   {4:0.5f} {5:0.5f}   {6:s}".format(centerval,lerr,uerr,uerr-lerr, uerr-centerval, centerval-lerr, str(ykeys[j])))
            if savediffs and not (np.isnan(fidval) or (fidind is None)):
                difflines.append("  ------\n  "+pointset["difflabel"]+"")
                for j in range(Nrows):
                    diff = xvals[j] - fidval

                    fidstd = xstds[j]
                    #print('>>>FIDSTD',fidstd)
                    if diff<0: # test<fid
                        testsig = xerr[1,j] # right side error bar
                        fidsig = xerr[0,fidind] # left side error bar
                    else:
                        testsig =xerr[0,j] #left error bar
                        fidsig = xerr[1,fidind] # right error bar
                    quadraturesig = np.sqrt(testsig**2 + fidsig**2)
                    diffsize = diff/quadraturesig

                    if not np.isnan(xvals[j]):
                        #difflines.append("    {0:+0.5f} {1:+0.5f} {2:0.5f} {3:0.5f} {4:0.5f} {5:s}".format(diff/quadraturesig,diff,testsig,fidsig, quardraturesig,ykeys[j]))
                        difflines.append("    {0:+0.5f} {1:+0.5f} {2:s}".format(diffsize,diff/fidstd, ykeys[j]))
                        if np.fabs(diffsize)>diffcut:
                            highlights[j]=True

            #print("2.>>>",highlights)
            # plot the points
            if not pointset['diffonly']:
                if not flip:
                    if highlightdiffs:
                        ax.scatter(xvals[highlights],(yvals+pointset['offset'])[highlights],marker='s',color='yellow',s=pointset['markersize']*20,lw=0,zorder=-45)
                        # trying to put extra line around marker
                        ax.scatter(xvals[highlights],(yvals+pointset['offset'])[highlights],marker=pointset['marker'],s=20*pointset['markersize'],facecolors='none',edgecolors='r',lw=1,zorder=-44)
                    ax.errorbar(xvals, yvals+pointset['offset'], xerr=xerr, marker=pointset['marker'],color=pointset['color'],markersize=pointset['markersize'],markeredgewidth=pointset['markeredgewidth'],markerfacecolor=pointset['markerfacecolor'],markeredgecolor=pointset['markeredgecolor'],ls='none',lw=1)
                else:
                    if highlightdiffs:
                        ax.scatter((yvals+pointset['offset'])[highlights],xvals[highlights],marker='s',color='yellow',s=pointset['markersize']*20,lw=0,zorder=-45)
                    ax.errorbar(yvals+pointset['offset'], xvals, yerr=xerr, marker=pointset['marker'],color=pointset['color'],markersize=pointset['markersize'],markeredgewidth=pointset['markeredgewidth'],markerfacecolor=pointset['markerfacecolor'],markeredgecolor=pointset['markeredgecolor'],ls='none',lw=1)

             # plot vlines and hlines if specified
        # first lets grab max and min; these are references so shouldn't affect plot range
        if not flip:
            axlims = ax.get_xlim()
        else:
            axlims = ax.get_ylim()
        if hgrid: #light gray lines separating each label
            for j in range(Nrows):
                if not flip:
                    ax.axhline(0.5+j ,ls='-',color='lightgray',lw=0.5,zorder=-70)
                else:
                    ax.axvline(0.5+j ,ls='-',color='lightgray',lw=0.5,zorder=-70)
        if hlines is not None:
            for hline in hlines:
                if type(hline)==dict:
                    # can be dictionary with keys params, value, kwargs
                    if 'params' in hline.keys():
                        if not ((hline['params']=='all') or (inparam in hline['params'])):
                            # if params is 'all' or not passed, will plot
                            # in all panels
                            # if passed as a list, will plot just for params
                            #  in that list
                            continue
                    if 'kwargs' in hline.keys():
                        if not flip:
                            ax.axhline(hline['value'],**hline['kwargs'])
                        else:
                            ax.axvline(hline['value'],**hline['kwargs'])
                    else:
                        if not flip:
                            ax.axhline(hline['value'],lw=1,ls='--',color='gray',zorder=-60)
                        else:
                            ax.axvline(hline['value'],lw=1,ls='--',color='gray',zorder=-60)
                else: # if just a number, will plot for all
                    if not flip:
                        ax.axhline(hline,lw=1,ls='--',color='gray',zorder=-60)
                    else:
                        ax.axvline(hline,lw=1,ls='--',color='gray',zorder=-60)
        if vlines is not None:
            for vline in vlines:
                if type(vline)==dict: #vlines will probably be dictionaries
                    if 'params' in vline.keys():
                        if not ((vline['params']=='all') or (inparam in vline['params'])):
                            continue
                    if 'colrange' in vline.keys():
                        # if tuple colange passed, just plots a line between those values
                        if 'kwargs' in vline.keys():
                            if not flip:
                                legend_elements.append(ax.plot([vline['value'],vline['value']],vline['colrange'],**vline['kwargs'])[0])
                            else:
                                legend_elements.append(ax.plot(vline['colrange'],[vline['value'],vline['value']],**vline['kwargs'])[0])
                        else:
                            if not flip:
                                ax.plot([vline['value'],vline['value']],vline['colrange'],lw=1,ls='--',color='gray',zorder=-60)
                            else:
                                ax.plot(vline['colrange'],[vline['value'],vline['value']],lw=1,ls='--',color='gray',zorder=-60)
                        
                    elif 'kwargs' in vline.keys():
                        if not flip:
                            ax.axvline(vline['value'],**vline['kwargs'])
                        else:
                            ax.axhline(vline['value'],**vline['kwargs'])
                    else:
                        if not flip:
                            ax.axvline(vline['value'],lw=1,ls='--',color='gray',zorder=-60)
                        else:
                            ax.axhline(vline['value'],lw=1,ls='--',color='gray',zorder=-60)
                else: # if just a number, will plot for all
                    if not flip:
                        ax.axvline(vline,lw=1,ls='--',color='gray',zorder=-60)
                    else:
                        ax.axhline(vline,lw=1,ls='--',color='gray',zorder=-60)

        if vbands is not None:
            for vband in vbands:
                if 'params' in vband.keys():
                    if not ((vband['params']=='all') or (inparam in vband['params'])):
                        continue
                if 'alpha' in vband.keys():
                    usealpha=vband['alpha']
                else:
                    usealpha=1
                if 'color' in vband.keys():
                    usecolor=vband['color']
                else:
                    usecolor='gray'

                if not flip:
                    ax.axvspan(vband['min'],vband['max'],edgecolor=None,facecolor=usecolor,alpha=usealpha,zorder=-80)
                else:
                    ax.axhspan(vband['min'],vband['max'],edgecolor=None,facecolor=usecolor,alpha=usealpha,zorder=-80)
        # put axis ranges back to where they were before these lines and bands were plotted
        if not flip:
            ax.set_xlim(axlims)
        else:
            ax.set_ylim(axlims)

        # set ranges
        if (xrangedict is not None) and (inparam in xrangedict.keys()):
            #print("setting xrange",param,xrangedict[param])
            if not flip:
                ax.set_xlim(xrangedict[inparam])
            else:
                ax.set_ylim(xrangedict[inparam])

        # set up ticks
        if not flip:
            ax.set_yticks([])# no yticks
            #print(ax.get_xticks())
            if xtickdict is not None:
                #print(xtickdict[param])
                ax.set_xticks(xtickdict[inparam])
           
            ax.set_xticklabels(ax.get_xticks(), size=ticksize)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        else:
            ax.set_xticks([])# no yticks
            #print(ax.get_xticks())
            if xtickdict is not None:
                #print(xtickdict[param])
                ax.set_yticks(xtickdict[inparam])

            ax.set_yticklabels(ax.get_yticks(), size=ticksize)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        #print(ax.get_xticks())

    # after doing plot, put row labels on the right
    # ax should be the rightmost panel at this point
    if  topheader is not None: #if we have a plot title, add it
        if not flip:
            if hideleg:
                ax.annotate(topheader,xy=(1.05,1.0), xycoords = 'axes fraction',\
                    color='gray', weight = 'bold', fontsize=topheadfontsize, ha='left',va='bottom')
            else:
                ax.annotate(topheader,xy=(1.05, 1.03), xycoords = 'axes fraction',\
                    color='gray', weight = 'bold', fontsize=topheadfontsize, ha='left',va='bottom')
        else:
            ax = axes[0]

            ax.annotate(topheader,xy=(1, 1+legheight/heightperparam), xycoords = 'axes fraction',\
                    color='gray', weight = 'bold', fontsize=topheadfontsize, ha='right',va='bottom')


    for j in range(Nrows):
        if not flip: # row labels go on right
            plt.annotate(ylabels[j],xy = (1.05, (yvals[j]-ymin)/(ymax-ymin)),\
                         xycoords='axes fraction',  color='k',\
                         fontsize = yfontsize,\
                         horizontalalignment='left', verticalalignment='center')
        else: # column labels go on bottom
            if centered_labels:
                plt.annotate(ylabels[j],xy = ((yvals[j]-ymin)/(ymax-ymin),-0.1),\
                         xycoords='axes fraction',  color='k',\
                         fontsize = yfontsize,\
                         horizontalalignment='center', verticalalignment='top',rotation=0)
            else:
                plt.annotate(ylabels[j],xy = ((yvals[j]-ymin)/(ymax-ymin),-0.05),\
                         xycoords='axes fraction',  color='k',\
                         fontsize = yfontsize,\
                        horizontalalignment='right', verticalalignment='top',rotation=45)
    # add watermark
    if  type(graytext)!=list: #can pass multiple
        graytext=[graytext]
    for gt in graytext:
        if gt:
            if type(gt)==str:
                if not flip:
                    fig.text(0.1, 0.3, gt, color='gray', alpha = .4,\
                             fontsize= 20, wrap=True, ha="left", va="center")
                else:
                    fig.text(0.1, 1.2, gt, color='gray', alpha = .4,\
                             fontsize= 20, wrap=True, ha="center", va="center",transform=fig.transFigure)
            else: # can also pass dictionary of text and kwargs
                #pass
                #print('gt[y]',gt['y'])
                trans = ax.get_xaxis_transform()
                #set this up for a specific thing I wanted to do; will need to adjust if we want to generalize
                ax.text(gt['x'],gt['y'], gt['text'],transform=trans,**gt['kwargs'])#,transform=fig.transFigure)



    # add legend
    print(legend_elements)
    if not flip:
        fig.legend(handles=legend_elements,loc=legloc,fontsize=legfontsize,ncol=legncol,**legkwargs)
    else:
        fig.legend(handles=legend_elements,loc='lower right',fontsize=legfontsize,ncol=legncol,**legkwargs,\
                   bbox_to_anchor=(0.93, (1.-topmarg/figheight)))

    outf = os.path.join(outdir,outname+'.'+ftype)
    print("Saving plot to ",outf)
    #plt.tight_layout()
    plt.savefig(outf,dpi=300,bbox_inches = 'tight')
    plt.clf()
    plt.close()

    if savediffs:
        diff_outf = outf.replace(ftype,'diffs.txt')
        print("Writing diffs to",diff_outf)
        f = open(diff_outf,'w')
        f.write('\n'.join(difflines))
        f.close()

    if savevals:
        vals_outf = outf.replace(ftype,'vals.txt')
        print("Writing vals to",vals_outf)
        f = open(vals_outf,'w')
        f.write('\n'.join(vallines))
        f.close()

#==================================================================
# approximate profile likelihood (not doing separate maximization,
#   but just estimating based on samples in chain
#==================================================================

def get_profile_like(gdchain,parameters=None,Nbins=50,dopost=False,paramlims=None,):
    """
    given getdist chain object and a list of parameters get the pseudo profile likelihood
    for those parameters

    parameters - list of parameter names. If none, does all of them.
    dopost - if False, gets max likelihood per bin, if True, uses post column
    paramlims - optional dictionary of parameter limits. If not passed will
         default to using ranges from getdist chain object

    Returns dictionary: keys are parameter names. values are arrays of shape 2,Nbins
       result[0,:] is the central parameter value within the bin
       result[1,:] is the max likelihood (or post) of all samples within that parameter range

    """
    if gdchain is None:
        return None
    if dopost:
        ycol = 'post'
    else:
        ycol = 'like'

    if parameters is None:
        parameters = gdchain.paramNames.list()

    yind = gdchain.paramNames.numberOfName(ycol)
    yvals = gdchain.samples[:,yind]

    chainbounds = gdchain.ranges

    result = {}
    for p in parameters:
        dat = np.nan*np.ones((2,Nbins))

        pind = gdchain.paramNames.numberOfName(p)
        pvals = gdchain.samples[:,pind]

        # set up bins
        pmax = None
        pmin = None
        if (paramlims is not None) and (p in paramlims.keys()):
            pmin = paramlims[p][0]
            pmax = paramlims[p][1]
        else:
            pmin = chainbounds.getLower(p)
            pmax = chainbounds.getUpper(p)
        if pmax is None:
            pmax = np.max(pvals)
        if pmin is None:
            pmin = np.min(pvals)

        binedges = np.linspace(pmin,pmax,Nbins+1)
        binwidth = binedges[1] - binedges[0]
        dat[0,:] = binedges[:-1]+0.5*binwidth

        # find profile values
        for a in range(Nbins):
            mask = (pvals>=binedges[a])*(pvals<binedges[a+1])
            if np.any(mask):
                dat[1,a] = np.max(yvals[mask])
                # if no points in sample, leave as nan (how array was initialized)

        result[p] = dat

    return result

#--------------------------------------------------
def get_profile_like_2d(gdchain,param1,param2,Nbins=15,paramlims=None,dopost=False,):
    """
    given getdist chain object and the names of 2 parameters get gridded pseudo profile likelihood
    for those parameters

    dopost - if False, gets max likelihood per bin, if false, uses post column
    paramlims - optional dictionary of parameter limits. If not passed will
         default to using ranges from getdist chain object

    Returns
     pvals - 2xNbins array, containing parameter 1 and parameter 2 values (1 is first, 2 second)
             corresponding to bins
     results - maximum loglike (or logpost) value of sample within each p1-p2 bin.
             first index is parameter1's bin, 2nd is for parameter 2
    """
    if gdchain is None:
        return None,None
    if dopost:
        zcol = 'post'
    else:
        zcol = 'like'

    zind = gdchain.paramNames.numberOfName(zcol)
    zvals = gdchain.samples[:,zind]
    xvals = np.nan*np.ones((2,zvals.size))

    pvals = np.nan*np.ones((2,Nbins))
    binedges =  np.nan*np.ones((2,Nbins+1))
    results = np.nan*np.ones((Nbins,Nbins))

    chainbounds = gdchain.ranges

    # figure out bin range
    for i,p in enumerate( [param1,param2]):
        pmax = None
        pmin = None
        pind = gdchain.paramNames.numberOfName(p)
        xvals[i,:] = gdchain.samples[:,pind]
        if (paramlims is not None) and (p in paramlims.keys()):
            pmin = paramlims[p][0]
            pmax = paramlims[p][1]
        else:
            pmin = chainbounds.getLower(p)
            pmax = chainbounds.getUpper(p)
        if (pmax is None) or pmax==np.nan:
            pmax = np.nanmax(xvals)
        if (pmin is None) or pmin==np.nan:
            pmin = np.nanmin(xvals)

        binedges[i,:] = np.linspace(pmin,pmax,Nbins+1)
        binwidth = binedges[i,1] - binedges[i,0]
        pvals[i,:] = binedges[i,:-1]+0.5*binwidth

    #print(pvals)

    # there's probably a quick numpy way to do this, but i'm just gonna loop....
    for i in range(Nbins):
        for j in range(Nbins):
            mask = (xvals[0,:]>=binedges[0,i])*(xvals[0,:]<binedges[0,i+1])*(xvals[1,:]>=binedges[1,j])*(xvals[1,:]<binedges[1,j+1])
            if np.any(mask):
                results[i,j] = np.nanmax(zvals[mask])
    return results,pvals

#--------------------------------------------------
def get_hist_2d(gdchain,param1,param2,Nbins=15,paramlims=None,useweight=True):
    """
    given getdist chain object and the names of 2 parameters get gridded counts
    of samples within different ranges for those parametes

    useweight - if True, reports sum of weights for each parameter bin. If false, reports number of samples
    paramlims - optional dictionary of parameter limits. If not passed will
         default to using ranges from getdist chain object

    Returns
     pvals - 2xNbins array, containing parameter 1 and parameter 2 values (1 is first, 2 second)
             corresponding to bins
     results - maximum loglike (or logpost) value of sample within each p1-p2 bin.
             first index is parameter1's bin, 2nd is for parameter 2
    """
    if gdchain is None:
        return None,None
    zcol = 'weight'

    zind = gdchain.paramNames.numberOfName(zcol)
    zvals = gdchain.samples[:,zind]
    xvals = np.nan*np.ones((2,zvals.size))

    pvals = np.nan*np.ones((2,Nbins))
    binedges =  np.nan*np.ones((2,Nbins+1))
    results = np.nan*np.ones((Nbins,Nbins))

    chainbounds = gdchain.ranges

    # figure out bin range
    for i,p in enumerate( [param1,param2]):
        pmax = None
        pmin = None
        pind = gdchain.paramNames.numberOfName(p)
        xvals[i,:] = gdchain.samples[:,pind]
        if (paramlims is not None) and (p in paramlims.keys()):
            pmin = paramlims[p][0]
            pmax = paramlims[p][1]
        else:
            pmin = chainbounds.getLower(p)
            pmax = chainbounds.getUpper(p)
        if (pmax is None) or pmax==np.nan:
            pmax = np.nanmax(xvals)
        else:
            pmax = min(pmax,np.nanmax(xvals)) # bin based on where samples are
        if (pmin is None) or pmin==np.nan:
            pmin = np.nanmin(xvals)
        else:
            pmin = max(pmin,np.nanmin(xvals)) # bin based on where samples are

        binedges[i,:] = np.linspace(pmin,pmax,Nbins+1)
        binwidth = binedges[i,1] - binedges[i,0]
        pvals[i,:] = binedges[i,:-1]+0.5*binwidth

    #print(pvals)

    # there's probably a quick numpy way to do this, but i'm just gonna loop....
    for i in range(Nbins):
        for j in range(Nbins):
            mask = (xvals[0,:]>=binedges[0,i])*(xvals[0,:]<binedges[0,i+1])*(xvals[1,:]>=binedges[1,j])*(xvals[1,:]<binedges[1,j+1])
            if np.any(mask):
                if useweight:
                    results[i,j] = np.sum(zvals[mask])
                else:
                    results[i,j]= np.sum(mask)
    return results,pvals


#--------------------------------------------------
# plot profile likelihood, potentially for multiple cahins
#--------------------------------------------------

def plot_profile_likelihoods(gdchains,labels, parameters,paramlims=None,Nbins=50,\
                             dopost=False,outdir='plots/',outnamebase='profile-likelihood',\
                             ftype='png',colors=None,linestyles=None,\
                             paramlabels=DEFAULT_PLABELS, figwidth=3.4,figheight=None,\
                             legfontsize=8,axes_fontsize=9,lab_fontsize=10,ylims=None,\
                             legloc=None,legtitle='',graytext="",notedict={}):
    """
    gdchains = list of getdist mcsample objects

    dopost - if True, does posterior profiles, if False, does likelihood

    Update 4/17/24: pulled out profile evaluation in separate function
     so it could potentially be used without this plotting setup.
    """
    if type(gdchains)!=list:
        gdchainlist = [gdchains]
        labels = [labels]
    else:
        gdchainlist = gdchains
    Nchains = len(gdchains)

    if type(parameters)!=list:
        parameters = [parameters]
        if type(paramlims)!=dict:
            useparamlims = [paramlims]
    Nparams = len(parameters)

    if type(paramlims)==dict:
        templims = []
        for p in parameters:
            templims.append(paramlims[p])
        useparamlims = templims
    else:
        useparamlims = paramlims

    if type(Nbins)!=list:
        Nbins = [Nbins]*Nparams

    if dopost:
        ycol = 'post'
        ylabel = r'profile $\log{[\mathcal{P}]}$'
    else:
        ycol = 'like'
        ylabel = r'profile $\log{[L]}$'
    pdicts = []
    likes = []
    weights = []
    allmissing=True
    dat = []
    for i,c in enumerate(gdchains):
        dati = get_profile_like(c,parameters,Nbins[i],dopost,useparamlims)
        if dati is not None:
            allmissing = False
        dat.append(dati)

    if allmissing:
        print("No chains in list:",labels)
        return

    tmarg = .05
    rmarg = .05
    if axes_fontsize is None:
        axes_fontsize = 8
    if lab_fontsize is None:
        lab_fontsize = 12
    bmarg =2*(axes_fontsize + lab_fontsize)/72.
    lmarg = 1.5*(lab_fontsize+2.5*axes_fontsize)/72.
    if figwidth is None:
        figwidth = min(lmarg+rmarg +3,4)
    if figheight is None:
        figheight = 2.45 #0.7*figwidth
    if colors is None:
        colors = DEFAULT_COLORS
    if linestyles is None:
        linestyles = Nchains*['-']
    if linestyles is None:
        linestyles = ['-']*Nchains

    if ylims is not None:
        if type(ylims[0])==tuple or type(ylims[0])==list:
            pass
        else:
            ylims = [[ylims]*Nchains]

    if graytext:
        if 'fontsize' not in notedict.keys():
            notedict['fontsize']=18
        if 'xloc' not in notedict.keys():
            notedict['xloc'] = (figwidth-1.5*rmarg)/figwidth#.1
        if 'yloc' not in notedict.keys():
            notedict['yloc'] = (figheight-1.5*tmarg)/figheight#1.2*bmarg/figheight
        if 'ha' not in notedict.keys():
            notedict['ha'] = 'right'
        if 'va' not in notedict.keys():
            notedict['va'] = 'top'
        if 'rotation' not in notedict.keys():
            notedict['rotation'] = 90.


    for i,p in enumerate(parameters):

        #make plot
        outfile = os.path.join(outdir,''.join([outnamebase,'.',p,'.',ftype]))
        fig,axes = plt.subplots(1,1,figsize=(figwidth,figheight))
        plt.subplots_adjust(right=1.-rmarg/figwidth,left=lmarg/figwidth,bottom=bmarg/figheight,top=1.-tmarg/figheight)
        plt.ylabel(ylabel,fontsize=lab_fontsize)
        if ylims is not None:
            plt.set_ylim(ylims[i][0],ylims[i][1])
        plt.xlabel(paramlabels[p],fontsize=lab_fontsize)
        if graytext:
            fig.text(notedict['xloc'], notedict['yloc'], graytext,\
                     color='gray', alpha = .4,\
                     fontsize= notedict['fontsize'], wrap=True, \
                     ha=notedict['ha'], va=notedict['va'], rotation=notedict['rotation'])

        for n in range(Nchains):
            # plot a line for each chain
            plt.plot(dat[n][p][0,:],dat[n][p][1,:],color=colors[n],ls=linestyles[n],label=labels[n])

        leg  = plt.legend(title=legtitle,loc=legloc,fontsize = legfontsize,framealpha=0)
        leg.get_title().set_fontsize(legfontsize)

        print("Saving to",outfile)
        plt.savefig(outfile,dpi=300)
        plt.clf()
        plt.close()

#------------------------------------------------------------
def get_maxpost_gdchain(gdchains,Nsamples=1,postlabel='post',getmin=False):
    """
    Given getdist chain objects, find parameter values for samples with
    highest posterior.
    For each chain, Will return dictionary with chain column headers as keys
    parameter values as values.

    if nsamples is set higher than 1, returns a list of dictionaries for each
    chain, one for each of the  nsamples highest posteriors

    NOTE THAT THIS FUNCTION ISN"T SET UP to check for higher density
    polychord outputs  you'll need to do that separately or add that in.
    See stat_utils get_maxpost function for an idea of how that could work

    if only given one chain, one sample,
        just returns a dictionary of parameter values
    if multiple chains, one sample,
        returns list of dictionaries, in which each value is a float
    if multiple chains, Nsample>1,
        returns list of dictionaries, in which each value is an size Nzample array
    getmin - if True, minimize parameter instead of maximize
    """

    if type(gdchains)!=list:
        onechain=True
        gdchainlist = [gdchains]
    else:
        onechain=False
        gdchainlist = gdchains

    if len(gdchainlist)==1:
        onechain=True
    outdicts = []

    # get posteriors
    for ch in gdchainlist:

        if ch is None:
            outdicts.append(None)
            continue
        if Nsamples>1:
            chdicts = []
        # find posteriors, find inices of top Nsamples psterior points
        postind = ch.paramNames.numberOfName(postlabel)
        postvals = ch.samples[:,postind]
        if not getmin: # order from largest to smallest
            sortinds = np.argsort(postvals)[::-1]
        else: # order from smallest to largest
            sortinds = np.argsort(postvals)

        for i in range(Nsamples):
            pdict = ch.getParamSampleDict(sortinds[i])
            pdict['index'] = sortinds[i]
            # this includes derived parametes, post, weight, loglike, prior, like, plost...
            if Nsamples==1:
                outdicts.append(pdict)
            else:
                chdicts.append(pdict)

        if Nsamples>1:
            chdict = {}
            for k in chdicts[0].keys():
                chdict[k] = np.array([chdicts[i][k] for i in range(Nsamples)])
            outdicts.append(chdict)

    if onechain:
        return outdicts[0]
    else:
        return outdicts

#------------------------------------------------------------
def setup_mapscatter(gdchain,Nsamples=10,color='k',marker='+',baselabel='test',postlabel='post',getmin=False,minchisq=False):
    """
    Given getdist chain object,
    set up list of marker dictionaries that can be used
    to add points to a contour plot.

    Uses color and marker to identify point with the highest posterior
    and if Nsample>1, will make smaller and lighter markers for the
    next Nsample-1 highest posteriors.

    Legend labels will note the range of change in post across the samples shown

    minchisq bool is just used to set legend labels
    """
    if gdchain is None:
        return None,None

    mapvals = get_maxpost_gdchain(gdchain, Nsamples=Nsamples, postlabel=postlabel,  getmin=getmin)
    # will be dictionary with arrays as values
    postvals = mapvals['post']
    postmin = np.min(postvals)
    postmax = np.max(postvals)




    if Nsamples>1:
        mapd={k:mapvals[k][0] for k in mapvals.keys()} # highest post point
        if minchisq:
            label = 'min chisq: {0:s}\n'.format(baselabel)+r'({0:d} lowest, $\Delta\chi^2=${1:0.2f})'.format(Nsamples,postmax-postmin)
        else:
            label = 'MAP: {0:s}\n'.format(baselabel)+r'({0:d} highest, $\Delta\log\mathcal{{P}}=${1:0.2f})'.format(Nsamples,postmax-postmin)
        nextd={k:mapvals[k][1:] for k in mapvals.keys()} # 2nd highest to Nth highest post
    else:
        mapd={k:mapvals[k] for k in mapvals.keys()} # highest post point
        if minchisq:
            label = 'min chisq: {0:s}'.format(baselabel)
        else:
            label = r'MAP: {0:s}'.format(baselabel)
        nextd = None

    # formatting for highest MAP
    mapd['label'] = label
    mapd['c'] = color
    mapd['markerkwargs'] = {'marker':marker}
    mapd['linekwargs'] = {'lw':1,'ls':'-'}

    if Nsamples>1:
        # make points lighter for lower posterior
        norm = matplotlib.colors.Normalize(vmin=postmin,vmax=postmax)
        #usecmap = matplotlib.cm.RdBu

        rgbcol = matplotlib.colors.to_rgb(color)

        hlscol = colorsys.rgb_to_hls(*rgbcol) # hue luminosity, saturation
        hlscol2 = (hlscol[0],1.5*hlscol[1],hlscol[2])
        # faint end of gradient
        rgbcol2 = colorsys.hls_to_rgb(*hlscol2)
        usecmap = matplotlib.colors.LinearSegmentedColormap.from_list('mymap',[rgbcol,rgbcol2],N=128)

        cmap = matplotlib.cm.ScalarMappable(norm=norm,cmap=usecmap)
        cmap.set_array([])

        postcols = cmap.to_rgba(postvals[1:])
        nextd['c'] = postcols

        basesize = rcParams['lines.markersize'] ** 2 #default markersize
        nextd['markerkwargs'] = {'marker':marker,'alpha':0.7,'s':0.6*basesize,'lw':0.7,'zorder':100}
        nextd['linekwargs'] = {'lw':0.5,'ls':'-','alpha':0.3}

    return mapd,nextd


#------------------------------------------------------------
def setup_randomscatter(gdchain,Nsamples=100,color='gray',marker='.',label='Random draw'):

    """
    Given getdist chain object,   set up dictionary for adding markers to chain plot
    by randomly selecting approximately Nsamples samples, with probability of selection
    proprotional to posterior
    """
    if gdchain is None:
        return None

    N = gdchain.samples.shape[0] # total number of samples in chain.
    thin = N/Nsamples/3

    # get random samples chosen with probability proportiona to their weight
    # (note that getdist has max_scatter_points set to 2000)
    outsamples = gdchain.makeSingleSamples(single_thin = thin)

    parameters = gdchain.paramNames.list()
    pinds = {p:gdchain.paramNames.numberOfName(p) for p in parameters}
    outdict = {p:outsamples[:,pinds[p]] for p in parameters}

    outdict['c'] = color
    basesize = rcParams['lines.markersize'] ** 2 #default markersize
    outdict['markerkwargs'] = {'marker':marker,'alpha':0.4,'lw':0,'s':0.3*basesize}
    outdict['dolines'] = False # don't plot lines on 1d plots (too busy)
    outdict['label'] = label

    return outdict

#------------------------------------------------------------

def shift_chain(gdchain,centeron_dict={},centerstat='peak',shiftranges=True):
    """
    Given getdist chain mcsamples object, shift values of 
    parameters to center contours on some specified values. 

    Note that this will not do any consistency checking of
    derived parameters, it's intended ot be used
    just for plotting. 

    Arguments:
    gdchain - getdist mcsamples chin object
    centeron_dict - dictionary of parameter values
        that you want to shift chain for. Keys are 
        parameter names, values are desired center value
        Only parameters included in this list will be shifted
    centerstat - can be 'peak' 'mean', or 'chain_map'
        peak will use 1d marginalied posterior max
        mean will use mean of 1d marg posts
        chain_map will use the chain's highest posterior point
    shiftranges - bool, if true, will also offset the 
       parameter ranges by the same amount (getdist will
       clip samples shifted out of prior range, this will 
       avoid that)

    returns a gettdist chain object with 
    shifted parameter values
    """
    # make a copy (uses deepcopy)
    ch = gdchain
    outch = gdchain.copy()

    origvals = get_centerstat_dict(gdchain,centeron_dict.keys(),centerstat)
    # loop through params we want to shift

    for p in centeron_dict.keys():
        origval = origvals[p]
        newval = centeron_dict[p]
        if origval is None: # parameter isn't in the chain
            continue 
        shift = newval - origval
        #print(">>",p,shift)
        pind = ch.paramNames.numberOfName(p)
        if shiftranges:
            origlower = ch.ranges.getLower(p)
            origupper = ch.ranges.getUpper(p)
            if origlower is None: # None ranges stay None
                newlower = None
                newupper = None
            else:
                newlower = origlower+shift
                newupper = origupper+shift
            #print("    ",newlower,newupper)
            outch.ranges.setRange(p,(newlower,newupper))
        origsamples = ch.samples[:,pind]
        newsamples = origsamples+shift
        outch.samples[:,pind] = newsamples
        

    return outch


def get_centerstat_dict(gdchain,paramlist,centerstat='peak'):
    """
    Helper function for shift_chain: given a chain,
    get the 1d marg peaks, means, or chain_map for 
    each parameter. 

    WARNING: this will do no consisteny checking between
    parameters it just finds the stats individually

    returns a dictionary with parameter names as keys
    center stat as values
    """
    ch = gdchain
    outdict = {}
    
    if centerstat == 'chain_map': # get index of max post
        postlabel = 'post'
        postind = ch.paramNames.numberOfName(postlabel)
        postvals = ch.samples[:,postind]
        sortinds = np.argsort(postvals)[::-1]
        mapdict = ch.getParamSampleDict(sortinds[0])
    for p in paramlist:
        # get relevant value from original chain
        pind = ch.paramNames.numberOfName(p)
        if pind==-1: #parameter not in chain
            origval = None
        else:
            if centerstat=='peak':
                # find 1d marg post peak
                margdens = ch.get1DDensity(p)
                guess = margdens.x[np.argmax(margdens.P)]
                probfunc = lambda x: -1*margdens.Prob(x)
                minval =sp.optimize.fmin(probfunc,guess,disp=False)[0]
                origval = minval
            elif centerstat=='mean':
                margdens = ch.get1DDensity(p)
                mean = (margdens.x*margdens.P).sum()/margdens.P.sum()
                origval = mean
            elif centerstat=='chain_map':
                origval = mapdict[p]
                
            else:
                raise ValueError("In get_centerstat_dict, centerstat arg {} not recognized".format(centerstat))

        outdict[p] = origval

    return outdict

    

