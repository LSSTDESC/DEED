"""
Script demonstrating how to use functions in getdist_chainplot_utils.py
to make chain plots. 
"""

import sys, os
import numpy as np
import scipy as sp


import getdist_chainplot_utils as cplot
## if your plotting script or notebook is in a different directory do this instead:
#chainplotdir = os.environ['Y6METHODSDIR']+"/y6_fiducial/plot_chains"
#sys.path.insert(0,chainplotdir)
#cplot = __import__('getdist_chainplot_utils')

#==================================================================
def main():
    FTYPE='png'
    KDESMOOTH=0.5
    # ^ getdist optimized is -1, 0.5 pretty similar but not exact
    # changing to 0.2 or -0.6 will give noisier contours but can be good for diagnostics

    # where to find chains
    chaindir = os.path.join(os.environ['HOME'],'workspace/des-repos/y3-3x2pt/code/chains_fromnersc')
    # where to put plots
    plotdir =os.path.join(os.environ['Y6METHODSDIR'],'plot_chains/')

    # dictionary translating cosmosis parameter labels to latex labels
    paramlabels = cplot.DEFAULT_PLABELS

    # --------------------------------------------------------------------------------
    # read in the chain files that we want to make plots for
    # dictionary structures not necessariy, but helps keep track of chains
    # without re-reading them in for every single plot
    # dictionary of chain objects 
    chaindict = {}

    # assoicating short tags with filenames, chains, labels
    # filenames for chains. 'oops' is there to show that
    # the plotting script will just skip files that can't be read in
    # but will keep colors and labels correctly associated
    filedict = {\
                '3x2pt':'chain_3x2pt_lcdm_SR_maglim.txt',\
                '3x2pt_noSR':'chain_3x2pt_lcdm_noSR_maglim.txt',\
                'shear':'chain_1x2pt_lcdm_SR_maglim.txt',\
                'shear_noSR':'chain_1x2pt_lcdm_noSR_maglim.txt',\
                'planck':'chain_p-TTTEEE-lowE_lcdm.txt',\
                'oops':'notarealfile.txt',\
                }


    # What will be in plot legends
    labeldict = {\
                 'shear':r"$1\times2$pt+SR",\
                 'shear_noSR':r"$1\times2$pt",\
                 '3x2pt':r"$3\times2$pt+SR",\
                 '3x2pt_noSR':r"$3\times2$pt",\
                 'planck':r"Planck",\
                 'oops':"Nonexistant file",\
                }
   
    for k in filedict.keys():
        # dictionary will hold getdist MCSamples objects;
        # if there was an error in reading in file, entry will be None
        fname = filedict[k]
        chaindict[k]= cplot.prep_chain(fname,labeldict[k],kdesmooth=KDESMOOTH, paramlabels = paramlabels,chaindir=chaindir)

    print(chaindict)

    coldict ={\
              '3x2pt':cplot.midblue,\
              '3x2pt_noSR':cplot.lightblue,\
              'shear':cplot.lightred,\
              'shear_noSR':cplot.pink,\
              'planck':'k',\
              'oops':'gray',\
    }

    # --------------------------------------------------------------------------------
    # Make a 1D posterior plot of how different choices impact S8 posteriors
    # --------------------------------------------------------------------------------
    if 0:
        tags = ['oops','shear_noSR','shear','3x2pt_noSR','3x2pt']
        
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        labels = [labeldict[t] for t in tags]
        colors = [coldict[t] for t in tags] # colors to use to plot them
        linestyles=[':','-','--','-','--']
        
        outname = "test_S8_1d" #output plot name
        params = ['S8'] # what parameter to plot; with only 1 parameter does 1d posterior
        vlines = [0.8] # plot a vertical line (default is empty list),
        # also if savefracs=True (default), make_chainplot will save
        # fractions of posterior above or below each line
        
        
        # use this if you want to shade a region of the plot in the range given by 'x'
        filldict = {\
                    'x': (0.82,0.85),\
                    'kwargs':{'ec':cplot.beige1,'fc':cplot.beige2,'hatch':'//'}
                    }
        
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE, \
                             linestyles=linestyles,\
                             colors=colors, figwidth=3.5,legtitle='DES Y3',\
                             param_limits = (0.6,0.86), outdir=plotdir,\
                             ylim_multiplier=1.2,legloc='upper left',legfontsize=10,\
                             vlines=vlines, singleplot_filldict=filldict,\
                             )
        # ylim_multiplier lifts upper bound of plot above 1, adjust to make room for legends

    # --------------------------------------------------------------------------------
    # make a single panel Omega_m-S8 plot
    # --------------------------------------------------------------------------------
    if 0:
        outname = "test_Om-S8_singlepanel" #output plot name
        
        tags = ['oops','shear','3x2pt','planck']
        labels = [labeldict[t] for t in tags]
        chainlist = [chaindict[t] for t in tags] # list of chain objects
        colors = [coldict[t] for t in tags] # colors to use to plot them
        shades = [True, True, False, True]
        
        params = ['cosmological_parameters--omega_m','S8']
        fidlines = {'cosmological_parameters--omega_m':0.3,'S8':0.8} #put dashed lines here
        # can also add lines using vlines, hlines arguments
        
        # set plot ranges (if not passed, will just let getdist set this as a default)
        ommin= 0.15
        ommax= 0.45
        s8min= 0.7
        s8max= 1.0
        uselims = [ ommin,ommax, s8min, s8max]
        
        #singleplot_filldict - want to add a hatched region to show a non-standard prior?
        # (if not, don't pass this argument, or just pass singleplot_filldict=None
        fillx = np.linspace(0.25,0.3,10)
        filldict = {\
                    'x': fillx,\
                    'ymax':0.9+10*(fillx-.3)**2,\
                    'ymin':0.73*np.ones_like(fillx),
                    'kwargs':{'ec':"lightgray",'fc':'None','hatch':'/'}
                    }
        
        watermark = 'watermark'
        cplot.make_chainplot(chainlist, params, outname, labels,ftype=FTYPE,\
                             linestyles=linestyles,shade=shades,colors=colors,\
                             figwidth=3.5,legtitle='DES Y3',singleplot=True,\
                             intruthvals=fidlines,showtruth=True,\
                             singleplot_filldict=filldict, param_limits = uselims,\
                             outdir=plotdir,legfontsize=10,\
                             lab_fontsize=18,axes_fontsize=12,\
                             graytext=watermark)
        
        # ^ playing with font sizes (those options are optional)
        
        # --------------------------------------------------------------------------------
        # same thing but without singleplot, so 2 parameter triangle
        # for this, parameter limits are a dictionary
        outname = "test_Om-S8_triangle" #output plot name
        uselims = {'cosmological_parameters--omega_m':(.15,.45),'S8':(0.7,0.9)}
        
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                             colors=colors, linestyles=linestyles,shade=shades,\
                             figwidth=3.5,legtitle='DES Y3', intruthvals=fidlines,\
                             param_limits = uselims, outdir=plotdir,
                             legfontsize=10,graytext=watermark)
    

    # --------------------------------------------------------------------------------
    # rectangular array of plots showing how S8 covaries with a few other parameters
    # --------------------------------------------------------------------------------
    if 0:
        outname = 'test_S8_grid'
        params = [['cosmological_parameters--omega_m','asx1.e9','summnu','intrinsic_alignment_parameters--a1','intrinsic_alignment_parameters--alpha1','cosmological_parameters--n_s'],['S8','cosmological_parameters--sigma_8']]
        
        tags = ['shear_noSR','shear','3x2pt_noSR','3x2pt']
        labels = [labeldict[t] for t in tags]
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        colors = [coldict[t] for t in tags] # colors to use to plot them
        colors[1] = cplot.red # let's make this line darker
        colors[0] = cplot.beige3
        shades = [True, False,True,False]
        linestyles= ['-','-','-',':']
        
        # start by setting plotting range equal to prior range
        uselims = cplot.rangedict_from_gdchain(chainlist[0])
        # ^ parameters without hard priors will ahve range set 
        #  automatically based on first chain's range
        # adjust some things
        uselims['S8'] = (0.65,.85)
        uselims['cosmological_parameters--omega_m'] = (.15,.5)
        uselims['cosmological_parameters--sigma_8'] = (.4,1.1)
        #uselims['asx1.e9'] = (1,3)
        uselims['intrinsic_alignment_parameters--a1'] = (-3.5,2)
        #uselims[''] = ()
        
        # set up watermark text
        watermark = 'watermark'
        notedict={'fontsize':40,'yloc':0.7,'rotation':10} # this dictionary can be used to format the watermark
        
        # adjust where lgend goes
        legaxinds=(1,4) #subplot index where we want to put the legend
        
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='',legaxinds=legaxinds,\
                         legloc='lower right',legfontsize=8,\
                         param_limits = uselims, outdir=plotdir,
                         rectplot=True,rectratio=1.3,\
                         lab_fontsize=14,graytext=watermark,notedict = notedict,\
                         legncol=4,\
    )
    # --------------------------------------------------------------------------------
    # Triangle plot example showing two sets of reference lines
    # --------------------------------------------------------------------------------
    if 0:
        outname = 'test_triangle'
        params = ['S8','cosmological_parameters--omega_m','asx1.e9','summnu','intrinsic_alignment_parameters--a1','intrinsic_alignment_parameters--alpha1','cosmological_parameters--n_s']
        
        tags = ['shear_noSR','shear','3x2pt_noSR','3x2pt','planck']
        labels = [labeldict[t] for t in tags]
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        colors = [coldict[t] for t in tags] # colors to use to plot them
        colors[1] = cplot.red # let's make this line darker
        colors[0] = cplot.beige2 # make this lighter
        shades = [True, False,True,False,False]
        linestyles= ['-','-','-',':','-']
        

        # start by setting plotting range equal to prior range
        uselims = cplot.rangedict_from_gdchain(chainlist[0])
        # ^ parameters without hard priors will ahve range set 
        #  automatically based on first chain's range
        uselims['S8'] = (0.65,.9)
        uselims['cosmological_parameters--omega_m'] =(.15,.5)
        #uselims['asx1.e9'] = (1,3)
        uselims['intrinsic_alignment_parameters--a1'] =  (-2,2)
        #uselims[''] = ()
    
        # set up watermark text
        watermark = 'watermark'
        
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                         lab_fontsize=12,graytext=watermark)


    # --------------------------------------------------------------------------------
    # pseudo profile likeihood plot. 
    # --------------------------------------------------------------------------------
    if 0:
        outname = 'test_profilelike'
        params = ['S8','cosmological_parameters--omega_m']
        
        tags = ['shear_noSR','shear']#,'3x2pt_noSR','3x2pt','planck']
        # ^ different normalizations makes it hard to comapre 1x2pt to 3x2pt here
        labels = [labeldict[t] for t in tags]
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        colors = [coldict[t] for t in tags] # colors to use to plot them
        colors[1] = cplot.red # let's make this line darker
        colors[0] = cplot.beige2 # make this lighter
        linestyles= ['-','--','-.',':','-']
   
        
        # set up watermark text
        watermark = 'watermark'
    
        cplot.plot_profile_likelihoods(chainlist,labels,params,\
                                   colors=colors,\
                                   linestyles=linestyles,\
                                   outdir=plotdir, graytext=watermark,\
                                   outnamebase=outname)


    # --------------------------------------------------------------------------------
    # triangle plot showing impact of KDE parameter
    # --------------------------------------------------------------------------------
    if 0:
        basetag = '3x2pt'
        for basetag in ['3x2pt','shear']:
            kdevals = [0.5,-1,-0.6,.2]#,1]
            kdelabels = ["KDE param: {0:+0.1f}".format(k) for k in kdevals]
            kdetags = [basetag+'_kde{}'.format(k) for k in kdevals]
            colors = ['lightgray','k',cplot.red,cplot.midblue]
            shade = [True, False, False, False]
            legtitle = 'DES Y3: '+labeldict[basetag]
            
            for i,k in enumerate(kdetags): # read in chain with different KDE assumptions
                chaindict[k] = cplot.prep_chain(filedict[basetag],kdelabels[i],kdesmooth=kdevals[i],paramlabels=paramlabels,chaindir=chaindir)


            outname = 'test_kde_y3_'+basetag
            params = ['S8','cosmological_parameters--omega_m','asx1.e9','summnu','intrinsic_alignment_parameters--a1','intrinsic_alignment_parameters--alpha1','cosmological_parameters--n_s']
            
            tags = kdetags
            labels = kdelabels
            chainlist = [chaindict[t] for t in tags] # list of chain objets
            
            
            # start by setting plotting range equal to prior range
            uselims = cplot.rangedict_from_gdchain(chainlist[0])
            # ^ parameters without hard priors will ahve range set 
            #  automatically based on first chain's range
            uselims['S8'] = (0.65,.9)
            uselims['cosmological_parameters--omega_m'] =(.15,.5)
            #uselims['asx1.e9'] = (1,3)
            uselims['intrinsic_alignment_parameters--a1'] =  (-2,2)
            #uselims[''] = ()
    
            # set up watermark text
            watermark = 'KDE test'
            
            cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                                 colors=colors,\
                                 shade=shade,\
                                 figwidth=7,legtitle=legtitle,\
                                 legloc='upper right',legfontsize=12,\
                                 param_limits = uselims, outdir=plotdir,
                                 lab_fontsize=12,graytext=watermark)

    # --------------------------------------------------------------------------------
    # Triangle plot adding profile likelihoods to diagonals
    # --------------------------------------------------------------------------------
    if 0: 
        outname = 'test_triangle_profilelike_diagonals'
        params = ['S8','cosmological_parameters--omega_m','asx1.e9','summnu','intrinsic_alignment_parameters--a1','intrinsic_alignment_parameters--alpha1','cosmological_parameters--n_s']
        
        tags = ['shear','3x2pt']
        labels = [labeldict[t] for t in tags]
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        colors = [coldict[t] for t in tags] # colors to use to plot them
        shades = [True, False]
        linestyles= ['-','-']
        

        # start by setting plotting range equal to prior range
        uselims = cplot.rangedict_from_gdchain(chainlist[0])
        # ^ parameters without hard priors will ahve range set 
        #  automatically based on first chain's range
        uselims['S8'] = (0.65,.9)
        uselims['cosmological_parameters--omega_m'] =(.15,.5)
        #uselims['asx1.e9'] = (1,3)
        uselims['intrinsic_alignment_parameters--a1'] =  (-2,2)
        #uselims[''] = ()
    
        # set up watermark text
        watermark = 'testing profile likelihoods'
        
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                             lab_fontsize=12,graytext=watermark,showprofilelike=True)


    # ------------------------------------------------------------------
    # Testing contour shifting utility shift_chain
    # ------------------------------------------------------------------

    if 10: 
        outname = 'test_chainshifts_peak'
        # parameters to plot
        params = ['S8','cosmological_parameters--omega_m','cosmological_parameters--sigma_8']#,'summnu']
        # parameters to move
        shiftparams = ['S8','cosmological_parameters--omega_m','cosmological_parameters--sigma_8']
        # ^ beware, some wonky behavior may happen if you do shifts with
        # prior bounded parameters. so e.g. leaving summnu unshifted
        # could make sense
        
        tags = ['shear','3x2pt']
        labels = [labeldict[t] for t in tags]
        colors = [coldict[t] for t in tags] # colors to use to plot them
        mcoldict = coldict
        shades = [False, False]
        linestyles= ['-','-']

        #original chains
        chainlist = [chaindict[t] for t in tags] # list of chain objets
        peaks32 = cplot.get_centerstat_dict(chaindict['3x2pt'],shiftparams,'peak')
        #print(peaks32)
        means32 = cplot.get_centerstat_dict(chaindict['3x2pt'],shiftparams,'mean')
        maps32 = cplot.get_centerstat_dict(chaindict['3x2pt'],shiftparams,'chain_map')

        peakshiftlist = [cplot.shift_chain(ch,peaks32,'peak') for ch in chainlist]
        meanshiftlist = [cplot.shift_chain(ch,means32,'mean') for ch in chainlist]
        mapshiftlist = [cplot.shift_chain(ch,maps32,'chain_map') for ch in chainlist]
        
        # just use default behavior for plot ranges
        uselims = None

        # use the original chains
        outname = 'test_chainshifts_orig'
        watermark = 'original y3 chains'
        mapvals = cplot.get_maxpost_gdchain(chainlist) # will be list of dicts, one per chain of map values
        mapdicts = []
        for i,t in  enumerate(tags):
            mapd = mapvals[i] # this will be a dictionary with parameter (column) labels as keys
            # floats or arrays of floats as values
            label = ''#'MAP for '+labeldict[t]+' (post={0:0.2f})'.format(mapd['post'])
            mapd['label'] = label
            mapd['c'] = mcoldict[t]
            mapd['markerkwargs'] = {'marker':'+'}
            mapd['linekwargs'] = {'lw':1,'ls':'-'}
            mapdicts.append(mapd)
        cplot.make_chainplot(chainlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                             lab_fontsize=12,graytext=watermark,pointlist=mapdicts)

        # shift based on peaks
        outname = 'test_chainshifts_peak'
        watermark = 'shifting to match 1d post peaks'
        mapvals = cplot.get_maxpost_gdchain(peakshiftlist) # will be list of dicts, one per chain of map values
        mapdicts = []
        for i,t in  enumerate(tags):
            mapd = mapvals[i] # this will be a dictionary with parameter (column) labels as keys
            # floats or arrays of floats as values
            label = ''#'MAP for '+labeldict[t]+' (post={0:0.2f})'.format(mapd['post'])
            mapd['label'] = label
            mapd['c'] = mcoldict[t]
            mapd['markerkwargs'] = {'marker':'+'}
            mapd['linekwargs'] = {'lw':1,'ls':'-'}
            mapdicts.append(mapd)
        cplot.make_chainplot(peakshiftlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                             lab_fontsize=12,graytext=watermark,pointlist=mapdicts)

        # # shift based on means
        outname = 'test_chainshifts_mean'
        watermark = 'shifting to match means'
        mapvals = cplot.get_maxpost_gdchain(meanshiftlist) # will be list of dicts, one per chain of map values
        mapdicts = []
        for i,t in  enumerate(tags):
            mapd = mapvals[i] # this will be a dictionary with parameter (column) labels as keys
            # floats or arrays of floats as values
            label = ''#'MAP for '+labeldict[t]+' (post={0:0.2f})'.format(mapd['post'])
            mapd['label'] = label
            mapd['c'] = mcoldict[t]
            mapd['markerkwargs'] = {'marker':'+'}
            mapd['linekwargs'] = {'lw':1,'ls':'-'}
            mapdicts.append(mapd)
        cplot.make_chainplot(meanshiftlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                             lab_fontsize=12,graytext=watermark,pointlist=mapdicts)

        # shift based on chain_map
        outname = 'test_chainshifts_chainmap'
        watermark = 'shifting to match chain MAP ests'
        mapvals = cplot.get_maxpost_gdchain(mapshiftlist) # will be list of dicts, one per chain of map values
        mapdicts = []
        for i,t in  enumerate(tags):
            mapd = mapvals[i] # this will be a dictionary with parameter (column) labels as keys
            # floats or arrays of floats as values
            label = ''#'MAP for '+labeldict[t]+' (post={0:0.2f})'.format(mapd['post'])
            mapd['label'] = label
            mapd['c'] = mcoldict[t]
            mapd['markerkwargs'] = {'marker':'+'}
            mapd['linekwargs'] = {'lw':1,'ls':'-'}
            mapdicts.append(mapd)
        cplot.make_chainplot(mapshiftlist, params, outname, labels, ftype=FTYPE,\
                         colors=colors, linestyles=linestyles, shade=shades,\
                         figwidth=7,legtitle='DES Y3',\
                         legloc='upper right',legfontsize=12,\
                         param_limits = uselims, outdir=plotdir,
                             lab_fontsize=12,graytext=watermark,pointlist=mapdicts)


#==================================================================
if __name__=="__main__":
    main()
