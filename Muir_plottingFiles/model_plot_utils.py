"""
This script will provide functions which can be used
to make plots of how changing different parameters affect different
compontents of or modeling predictions. The functions will be split into
two categories:
  1. Those used to call cosmosis and do calculations at various 
     parameter values, saving output to text files. 
  2. Those used to read in those text files and make plots.

Note that to run functions in category 1, you'll need to have initialized
your cosmosis environment already. 

While I'll include some examples of how these functions can be used down
in the main() function, this script is mainly with the assumption that
it will be imported as a module into other scripts which will generate data 
and make plots for specific purposes. 

This was initially developed by Jessie Muir for the Y3 extensions project,
is being adapted to function in this workspace. 

"""
import os, sys, subprocess
import numpy as np
import scipy as sp
import matplotlib 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
import time
from astropy.io import fits

from matplotlib import rcParams
rcParams["figure.dpi"] = 200 # 100 is default
rcParams['font.family'] = 'serif'
plt.rcParams['font.size']=10
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
from matplotlib import rc
rc('text',usetex=True)

rad_to_arcmin = 60*180/np.pi

Y6METHODSDIR = os.environ['Y6METHODSDIR']
CSLDIR = os.environ['CSL_DIR']

# these autogen files will have some fucntions for generating sh filenames
autogen_utils_path = os.path.join(Y6METHODSDIR,'autogen_scripts')
sys.path.insert(0,autogen_utils_path)
autogen_utils = __import__('autogen_utils')

default_twopt_sections = ['shear_xi_plus', 'shear_xi_minus', 'galaxy_xi', 'galaxy_shear_xi']
default_pk_sections = ['matter_power_lin','matter_power_nl']
default_growth_sections = ['fsigma_8','sigma_8']
default_cmb_sections = ['tt','te','ee'] #can also include bb
default_distance_sections = ['d_a','h'] # can also include d_l, d_m, mu
default_lss_cl_sections = ['galaxy_cl','galaxy_intrinsic_cl','galaxy_magnification_cl','galaxy_shear_cl','magnification_cl','magnification_intrinsic_cl','magnification_shear_cl','shear_cl','galaxy_cl_gg','galaxy_shear_cl_gg','shear_cl_gg']
default_nz_samples = ['lens','source']

#############################################################################
# helper functions for getting lists of parameters, finding sh filenames, etc
#############################################################################
def get_cosmosis_iniobj(inifile, callfrom='', shfile=None):
    """ 
    parse ini file using cosmosis tools
    expects  file patsh relative to callfrom
    """
    from cosmosis.runtime import Inifile
    origpath = os.getcwd()

    if shfile is not None:
        get_bashvars_fromshfile(os.path.expandvars(shfile),callfrom)
        
    callfrom = os.path.expandvars(callfrom)
    inifile = os.path.expandvars(inifile)
    
    os.chdir(callfrom)
    result = Inifile(inifile, print_include_messages=True)
    os.chdir(origpath)
    
    return result

def get_run_dict_from_yml(ymlfile, runname, callfrom='', shfile=None):
    """
    parse yaml file using cosmosis tools

    expects yml file path relative to callfrom
    """
    from cosmosis.campaign import parse_yaml_run_file
    # all paths in yaml and ini will be relative to callfrom, so need to move there
    origpath = os.getcwd()

    callfrom = os.path.expandvars(callfrom)
    if shfile is not None:
        get_bashvars_fromshfile(os.path.expandvars(shfile),callfrom)
    callfrom = os.path.expandvars(callfrom)

    print("CALLING FROM",callfrom)

    os.chdir(callfrom)
    rundict,componentdict = parse_yaml_run_file(os.path.expandvars(ymlfile)) #dictionary containing all runs in yml file
    run = rundict[runname] # this is a dictionary represetting cosmosis setting

    # go back to original working dir
    os.chdir(origpath)
    
    return run

def get_pdictlist_vary1param(variedparam,otherparams=None,Nvals=10,\
                             minval=None, maxval=None, rangedict=None, \
                             fidvaldict = None, altfidval = None):
    """
    Given a parameter, return a  list of dictionaries with that parameter set to various 
    values. That list of dicitonaries can then be used in to call cosmosis using 
    funcitons in datavec_plot_utils. 

    Arguments:
    variedparam - cosmosis parameter name in section--param format for 
                  the parameter you want to vary
    otherparams - optional additioanl dictionary of other parameter values
             that you want to set along with the varied parameter. 
    Nvals - integer identifying how many parameter values you want to output. 
            Note that if the fiducial value isn't in this array, it'll be added
            so you will actually plot Nvals+1 lines.
    minval and maxval - set min and max value of array
    rangedict - if minval and maxval aren't passed, range will get taken from this
            dictionary mapping parameter values names to the range 
    fidvaldict - dictionary mapping parameter names to fiducial values
    altfidval - in case you want to override the value in fidvaldict with another number
    """

    if Nvals>1:
        if (maxval is not None) and (minval is not None):
            pmin = minval
            pmax = maxval
        else:
            pmin,pmax = rangedict[variedparam]

            
        if altfidval is None:
            pfid = fidvaldict[variedparam]
        else:
            pfid = altfidval
        pvals = np.linspace(pmin,pmax,Nvals)
        if pfid not in pvals:
            pvals =np.sort(np.append(pvals,pfid))
    else:
        if altfidval is  None:
            pfid = fidvaldict[variedparam]
        else:
            pfid = altfidval
        pvals= np.array([pfid])
            
    if otherparams is None:
        otherparams={}
    return [{variedparam:val,**otherparams} for val in pvals],pvals,pfid


# ---------------    
def get_values_from_yml(ymlfile,runname,callfrom='',shfile=None, rangedict=None,fiddict=None):
    """
    Given yml file and run name, figure out fid values and parameter ranges
    all paths defined relative to callfrom

    ymlfile: path to the relevant yml fle
    runname: run name within that yml file to use
    callfrom: path that cosmosis would get called from
    """
    run = get_run_dict_from_yml(ymlfile, runname, callfrom, shfile)
    values = run["values"] # this is a cosmosis Inifile object

    rangedict, fiddict = get_paramranges_from_iniobj(values, rangedict, fiddict)
    
    return rangedict, fiddict

# ---------------    
def get_priors_from_yml(ymlfile,runname,callfrom='',shfile=None, meandict=None,stddict=None):
    """
    Given yml file and run name, figure out fid values and parameter ranges
    all paths defined relative to callfrom

    ymlfile: path to the relevant yml fle
    runname: run name within that yml file to use
    callfrom: path that cosmosis would get called from
    """
    run = get_run_dict_from_yml(ymlfile, runname, callfrom, shfile)
    priors = run["priors"] # this is a cosmosis Inifile object

    meandict, stddict = get_priors_from_iniobj(priors, meandict, stddict)
    
    return  meandict, stddict

#-------------------------    
def get_values_from_ini(valfile,callfrom='',rangedict=None,fiddict=None,shfile=None):
    """
    read in values ini files to get parameter ranges and fid values
    all paths defined relative to callfrom

    valfile: path to values file to read in
    callfrom: path that cosmosis would get called from, for interpreting %include statements
    """
    # if rangedict is None:
    #     rangedict={}
    # if fiddict is None:
    #     fiddict = {}
    
    # if shfile is not None:
    #     print('rangedict',rangedict.keys())
    #     get_bashvars_fromshfile(shfile,callfrom)

    # valfile = os.path.expandvars(valfile)
    # callfrom = os.path.expandvars(callfrom)

    values = get_cosmosis_iniobj(valfile, callfrom, shfile)
    
    rangedict, fiddict = get_paramranges_from_iniobj(values, rangedict, fiddict)
    return rangedict, fiddict

#-------------------------
def get_priors_from_ini(priorfile,callfrom='',meandict=None,stddict=None,shfile=None):
    """
    read in values ini files to get prior means and standard deviations
    all paths defined relative to callfrom

    ymlfile: path to values file to read in
    callfrom: path that cosmosis would get called from, for interpreting %include statements
    """

    priors = get_cosmosis_iniobj(priorfile, callfrom, shfile)
    
    meandict, stddict = get_priors_from_iniobj(priorss, meandict, stddict)
    return meandict, stddict 

# #-------------------------
# def get_values_from_ini(valfile,callfrom=Y6METHODSDIR,rangedict={},fiddict={}):
#     """
#     read in values ini files to get parameter ranges and fid values

#     valfile: path to values file to read in
#     callfrom: path that cosmosis would get called from, for interpreting %include statements
#     """
#     valfile = os.path.expandvars(valfile)
#     if callfrom in valfile:
#         f = open(valfile)
#     else:
#         f = open(os.path.join(callfrom,valfile))
#     for line in f:
#         line = line.strip()
#         if not line: #empty line
#             continue
#         if line.startswith('%include'): # include line
#             # call recursively to read in values in the same way cosmosis would do
#             incfile = os.path.join(callfrom,line.replace('%include ',''))
#             get_values_from_ini(incfile,callfrom,rangedict,fiddict)
#         elif line.startswith(';') or line.startswith('#'): #comment
#             continue
#         elif line.startswith('['): #section header
#             section = line.replace('[','').replace(']','').lower()
#         else: # a parameter line
#             items = line.replace('=','').split()
#             var = items[0].lower()
#             key = section+'--'+var
#             if len(items)==4: # varied parameter
#                 minval = float(items[1])
#                 fidval = float(items[2])
#                 maxval = float(items[3])
#                 rangedict[key]=(minval,maxval)
#                 fiddict[key]=fidval
#             elif len(items)==2: #fixed parameter
#                 fidval = float(items[1])
#                 fiddict[key]=fidval

#     f.close()
    
#     return rangedict, fiddict
    
#-------------------------
def get_paramranges_from_iniobj(ini,rangedict=None,fiddict=None):
    """
    Given cosmosis ini file object, get dictionary reporting 
    parameter ranges and fiducial values
    """
    if rangedict is None:
        rangedict={}
    if fiddict is None:
        fiddict = {}
        
    for entry in ini:
        section,var = entry[0]
        key = section+'--'+var
        vlist = entry[1].split()
        if len(vlist)==3: #varied prameter
            minval = float(vlist[0])
            fidval = float(vlist[1])
            maxval = float(vlist[2])
            rangedict[key]=(minval,maxval)
            fiddict[key]=fidval
        elif len(vlist)==1: #fixed parameter
            fidval = float(vlist[0])
            fiddict[key]=fidval

    return rangedict, fiddict

#-------------------------
def get_priors_from_iniobj(ini,meandict=None,stddict=None):
    """
    Given cosmosis ini file object, get dictionary reporting 
    parameter ranges and fiducial values
    """
    if meandict is None:
        meandict={}
    if stddict is None:
        stddict = {}
        
    for entry in ini:
        section,var = entry[0]
        key = section+'--'+var
        priortype,vlist = entry[1].split(' ',1)
        # here I'm just going to assume we're working with gaussian priors
        # with mean and then stdev being the two numbers
        vlist = [float(v) for v in  vlist.split()]
        meandict[key] = vlist[0]
        stddict[key] = vlist[1]

    return meandict, stddict


#############################################################################
# FUNCTIONS FOR RUNNING COSMOSIS AND SAVING DATA
# (A lot of this has been adapted from the 2pt blinding script)
#############################################################################
# def parse_scut_ini(fname):
#     """
#     read in ini file containing scale cuts
#     output dict with 2pt names as keys, values equal to 
#      array with columns: bin1, bin2, anglemin, anglemax
#     (matching what plotting script expects)
#     """
#     output = {'xip':[],'xim':[],'gammat':[],'wtheta':[]}
    
#     f = open(fname)
#     for line in f:
#         if line.startswith('angle_range_'):
#             entries = line.split()
#             corrtype = entries[0].split('_')[2] #2pt name
#             b1,b2 = entries[0].split('_')[-2:] # bin numbers
#             b1=int(b1)
#             b2=int(b2)
#             angmin = float(entries[-2])
#             angmax = float(entries[-1])
#             output[corrtype].append([b1,b2,angmin,angmax])
#     f.close()
#     # switch these from lists to arrays
#     for k in output.keys():
#         output[k] = np.array(output[k])

#     return output


#-------------------------
def scutdict_from_yml(ymlfile,runname,callfrom,twopt_section='2pt_like',shfile=None,checkmodules=False,scutdict={}):
    """
    given cosmosis campaign yml file, run name,  and optional sh file, read in scale cuts
    If shfile has already been read in and added to environment varabiles
    don't ahve to do this again here. 
    """
    run = get_run_dict_from_yml(ymlfile, runname, callfrom, shfile)
    iniobj = run["params"] # this is a cosmosis Inifile object
    scutdict = get_scutdict_from_iniobj(iniobj,twopt_section,checkmodules,scutdict)
    return scutdict

#-------------------------
def scutdict_fromini(paramsini,callfrom='',shfile=None,twopt_section='2pt_like',checkmodules=False,scutdict={}):
    """
    given params ini file and sh ini file, read in scale cuts
    Has very similar function to read_scut_ini, but checks section and navigates %includes
    paths defined relative to callfrom
    If shfile has already been read in and added to environment varabiles
    don't ahve to do this again here. 
    """
    iniobj = get_cosmosis_iniobj(paramsini, callfrom, shfile)
    scutdict = get_scutdict_from_iniobj(iniobj,twopt_section,checkmodules,scutdict)
    return scutdict

    # if shfile is not None:
    #     get_bashvars_fromshfile(shfile,callfrom)
    # paramsini = os.path.expandvars(paramsini)
    # callfrom = os.path.expandvars(callfrom)
    # # check that twopt_section here is actually being used in module list
    # if checkmodules:
    #     module_list = find_modules_fromini(paramsini,shfile=None,callfrom=callfrom).split(' ')
    #     if not (twopt_section in module_list):
    #         raise ValueError("two pt likelihood {} is not in module list for file {}".format(twopt_section,paramsini))

    
            
    # f = open(os.path.join(callfrom,paramsini))

    # section=None
    # for line in f:
    #     line = line.strip()
    #     if not line: #empty line
    #         continue
    #     if line.startswith('%include'): # include line
    #         # call recursively to read in values in the same way cosmosis would do
    #         incfile = line.replace('%include ','').split(';')[0].strip()
    #         scutdict_fromini(incfile,callfrom=callfrom,twopt_section=twopt_section,checkmodules=False,scutdict=scutdict)
    #     elif line.startswith(';') or line.startswith('#'): #comment
    #         continue
    #     elif line.startswith('['): #section header
    #         section = line.replace('[','').replace(']','').lower()
    #     elif section!=twopt_section: #we're just looking for a line in the pipeline section
    #         continue
    #     else: # non-comment, non-include line in 2pt_like section
    #         if line.startswith('angle_range_'): # scale cut line
    #             entries = line.split()
    #             corrtype = entries[0].split('_')[2] #2pt name
    #             if corrtype not in scutdict.keys():
    #                 scutdict[corrtype]={}
                    
    #             b1,b2 = entries[0].split('_')[-2:] # bin numbers
    #             b1=int(b1)
    #             b2=int(b2)
    #             angmin = float(entries[-2])
    #             angmax = float(entries[-1])
    #             scutdict[corrtype][(b1,b2)] = (angmin,angmax)
    # f.close()

    # return scutdict

#-------------------------
def get_scutdict_from_iniobj(iniobj,twopt_section='2pt_like',checkmodules=False,scutdict={}):

    # check that twopt_section here is actually being used in module list
    if checkmodules:
        module_list = iniobj['pipeline','modules'].split(' ')
        if not (twopt_section in module_list):
            raise ValueError("two pt likelihood {} is not in module list for file {}".format(twopt_section,paramsini))
    
    items = iniobj.items(section=twopt_section)
    # ^ list of tuples, (parameter, value)
    for item in items:
        if item[0].startswith('angle_range_'): # this is a scale cut setting!
            nameparts = item[0].split('_')
            corrtype = nameparts[2] #2pt name
            if corrtype not in scutdict.keys():
                scutdict[corrtype] = {}
            b1,b2 = nameparts[-2:] # bin numbers
            b1=int(b1)
            b2=int(b2)

            vals = item[1].split()
            angmin = float(vals[0])
            angmax = float(vals[1])
            scutdict[corrtype][(b1,b2)] = (angmin,angmax)
            #print("SC   ",corrtype,b1,b2,angmin,angmax)

        elif item[0].startswith('cut_'): # full bin combos are cut
            nameparts = item[0].split('_')
            corrtype = nameparts[1] #2pt name
            if corrtype not in scutdict.keys():
                scutdict[corrtype] = {}
            binpairs = item[1].split()
            for bp in binpairs:
                p = bp.split(',')
                b1= int(p[0])
                b2= int(p[1])
                scutdict[corrtype][(b1,b2)] = (999.0,999.0) # cut all measurements!

    return scutdict
            

def get_bashvars_fromshfile(shfile,force_testsampler=True,callfrom=Y6METHODSDIR):
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

def get_planck_module(inifile,planck_module=None,callfrom='',campaign_yaml=None,campaign_run=None):
    """
    Cosmosis gets finicky about planck modules getting initialized multiple times.
    This is part of a hack to get around that. 
    """
    startingdir = os.getcwd()
    os.chdir(callfrom)
    haveplanck = planck_module is not None
    if haveplanck:
        from cosmosis.runtime.config import Inifile
        from cosmosis.runtime.pipeline import LikelihoodPipeline
        if campaign_yaml is not None:
            doyaml=True
            from cosmosis.campaign import parse_yaml_run_file
            rundict = parse_yaml_run_file(campaign_yaml)[campaign_run]
            ini_v1 = rundict["params"]
        else:
            ini_v1 = Inifile(inifile)
        ini_v1.set("pipeline","modules",planck_module)
        planck_only_pipeline = LikelihoodPipeline(ini_v1)
        planck_module_obj = planck_only_pipeline.modules[0]
        print("PLANCK MODULE OBJ INITIALIZED")
    else:
        planck_module = None
        planck_module_obj = None

    
    os.chdir(startingdir)
    return planck_module_obj

def run_cosmosis_togendat(pdict={},inifile='params_template.ini',callfrom = '',\
                          do_twopt = True, do_pk=True, do_growth=True,\
                          do_dist=True, do_cmb=True, do_nz=True,\
                          twopt_sections = default_twopt_sections, \
                          pk_sections = default_pk_sections, pk_zvals = [0],\
                          growth_sections = default_growth_sections,\
                          distance_sections = default_distance_sections,\
                          cmb_sections = default_cmb_sections,\
                          nz_samples = default_nz_samples,\
                          shifts=False,\
                          planck_module = None,planck_module_obj=None,\
                          do_lss_cl = True, lss_cl_sections = default_lss_cl_sections,\
                          campaign_yaml = None, campaign_run = None,
):
    """ 
    Runs cosmosis pipeline to generate 2pt functions using the pipeline
    specfied in inifile. 

    If a parameter value is passed through the dictionary pdict...
        ...and SHIFTS = False, the param it will be set to that value.
        ...and SHIFTS = True, the param will be shifted by that amount 
           from the number in the values file
    Otherwise, it will be set to whatever value is in values ini file.
    
    Note that you'll need to source the cosmosis setup file before running this.

    callfrom: If cosmosis needs to be called from a specific location for the 
    ini file referenced to work, put the path to that 

    Getting complaints about planck module getting initialized multiple times, planck
    module wrangling is here to handle that

    campaign_yaml, campaign_run - set these if you want to use a cosmosis v3 
       campaign file instead of an ini file. If these aren't included
       we won't import cosmosis.campaign; hopefully that'll keep things
       back-compatible with cosmosis v2
    """
    # change working dir to call from, noting current dir before moving
    startingdir = os.getcwd()
    print("    Cosmosis will be called from path {0:s}".format(callfrom))
    os.chdir(callfrom)

    haveplanck = planck_module is not None
    
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline
    from cosmosis.datablock.cosmosis_py import block

    if campaign_yaml is not None:
        if campaign_run is None:
            raise ValueError("You set up a campaign_yaml file, but not campaign_run.")
        doyaml=True
        from cosmosis.campaign import parse_yaml_run_file
        #print(parse_yaml_run_file(campaign_yaml)[0].keys())
        rundict = parse_yaml_run_file(campaign_yaml)[0][campaign_run]
        ini = rundict["params"]
        valini = rundict["values"]
        priorini = rundict["priors"]
        print('     Cosmosis ini object initialized from campaign file {}, run {}'.format(campaign_yaml, campaign_run))

    else:
        doyaml=False

    #if haveplanck:
    #    planck_module_obj = get_planck_module(inifile,planck_module)
    
        ini=Inifile(inifile)
        print('     Cosmosis ini object initialized from',inifile)
    #print('do_twopt',do_twopt)

    if haveplanck: 
        # we do this because of an odd feature in the Planck likelihoods
        # that give errors if they get initialized twice
        mod_names = ini.get("pipeline","modules").split()
        planckindex = mod_names.index(planck_module)
        #print('mod_names before removing planck:',mod_names)
        mod_names.remove(planck_module)
        #print('mod_names AFTER removing planck:',mod_names)
        ini.set("pipeline","modules"," ".join(mod_names))
    
    # Supress output; we don't want to create output directories or files 
    if 'test' in ini.__dict__['_sections'].keys():
        ini.__dict__['_sections']['test']['save_dir']=''
    if 'output' in ini.__dict__['_sections'].keys():
        ini.__dict__['_sections']['output']['filename']=''
    ini.__dict__['_sections']['pipeline']['debug']='F'
    ini.__dict__['_sections']['pipeline']['quiet']='T'

    if doyaml:
        # if we don't pass the valini object, it'll just read in values
        # based on what's listed in the params files
        pipeline = LikelihoodPipeline(ini,values=valini,priors=priorini)
    else:
        pipeline = LikelihoodPipeline(ini)
    print('     Cosmosis pipeline object initialized')
    if haveplanck: #add in pre-initialized planck module
        pipeline.modules.insert(planckindex,planck_module_obj)


    # Go through and change parameter values as specfied in pdict
    haveshifted = [] #track which values have been changed
    for parameter in pipeline.parameters:
        key = str(parameter)
        if key in pdict.keys():
            matchtype = type(parameter.start)
            if shifts:
                parameter.start = parameter.start + matchtype(pdict[key])
            else:

                parameter.start = matchtype(pdict[key])
            haveshifted.append(key)

        # need to set all parameter to be fixed for  run_parameters([])
        #  to work. Doing this will effectively run things like the test sampler
        #  no matter what sampler is listed in the ini file
        pipeline.set_fixed(parameter.section, parameter.name, parameter.start)
        
    p = pipeline.start_vector()

    # Check whether all the parameter shifts requested were able to be done
    # This error will get raised if you spell a parameter name differently
    # that how cosmosis tracks it. 
    print("     Have shifted:",haveshifted)
    if len(haveshifted) != len(pdict.keys()):
        print("    pdict.keys=",len(pdict.keys()),pdict.keys())
        print("   haveshifted=",len(haveshifted),haveshifted)
        raise ValueError("Mismatch between number of parameter shifts requested and number of parameters found in cosmosis.")

    # run pipeline to generate 2pt functions for that cosmology
    print("about to RUN COSMOSIS FOR pdict=",pdict)
    data = pipeline.run_parameters([]) #data is a datablock object
    try:
        print("     RUNNING PIPELINE [{0:s}]".format(time.strftime("%Y-%m-%d %H:%M")))
        #data = pipeline.run_parameters(p) #data is a datablock object

        #clean up
        if haveplanck:
            #remove the Planck module before cleaning up, otherwise as we want to re-use it.
            pipeline.modules.remove(planck_module_obj)
        pipeline.cleanup()
    except:
        print("Cosmosis had a problem")
        return None

    
        
    #Pull out info we want from the datas block and put it in a dictionary
    outdat = {}
    if do_twopt:
        outdat = get_twoptdict_from_pipelinedata(data,outdat,twopt_sections)
        outdat = get_twopt_cuts_from_ini(ini,outdat,twopt_sections)
    if do_pk:
        outdat = get_pk_from_pipelinedata(data,outdat,pk_sections,pk_zvals)
    if do_growth:
        outdat = get_growth_from_pipelinedata(data,outdat,growth_sections)
    if do_dist:
        outdat = get_dist_from_pipelinedata(data,outdat,distance_sections)
    if do_cmb:
        outdat = get_cmbcl_from_pipelinedata(data,outdat,cmb_sections)
    if do_lss_cl:
        outdat = get_lss_cl_from_pipelinedata(data,outdat,lss_cl_sections)
    if do_nz:
        outdat = get_nz_from_pipelinedata(data,outdat,nz_samples)
    #change working dir back to where we started
    os.chdir(startingdir)

    return outdat


#-----------------------------------------------
# functions for getting power spec and growth from block
#-----------------------------------------------
def get_pk_from_pipelinedata(data,outdict=None,types=['matter_power_lin','matter_power_nl'],zvals=[0]):
    """
    Get matter power spectrum out of datablock. 
    If pk_zvals is None, just get all z values, returned dicitonary value will be a 2d grid of pk for all k and z
    If it is a list of z values, will separate 1d arrays with separate dict keys for each z. 

    If z value in list isn't in 2d array, will interpolate pk to that z value. 
    Will skip z values outside range in datablock.

    types can be 'matter_power_lin','matter_power_nl', 
    this should work also if you have wel using weyl_curvature_spectrum or weyl_curvature_spectrum_nl
    """
    if data is None: #this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}

    for category in types:
        try:
            kdat = data[category,'k_h']
            zdat = data[category,'z']
            pdat = data[category,'p_k'] #Nz x Nk array
        except:
            print("Unable to read in power spectrum data for section {0}".format(category))
            continue

        outdict[category+'_k_h'] = kdat
        if zvals is None:
            outdict[category+'_z'] = zdat
            outdict[category+'_p_k'] = pdat
        else:
            pk_interp = interp1d(zdat,pdat,axis=0,bounds_error=False)
            for z in zvals:
                if (z < np.min(zvals)) or (z > np.max(zvals)):
                    # skip points outside range
                    continue
                pkdat_z = pk_interp(z)
                pkey = category+'_p_k_z{0:g}'.format(z)
                zkey = category+'_z{0:g}'.format(z)
                outdict[pkey] = pkdat_z
                outdict[zkey] = z
    return outdict

def get_growth_from_pipelinedata(data,outdict=None,types=['fsigma_8','sigma_8']):
    """
    Expects growth functions as saved by CAMB module 
    options are d_z, da, f_ap, f_z, fsigma_8, h, rs_dv, sigma_8, z, a
    """
    if data is None: #this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}
    category = 'growth_parameters'
    
    savekeys = types.copy()
    savekeys.append('z')
    for k in savekeys:
        #print(block[category,k])
        try:
            kdat = data[category,k]
        except:
            print("Can't find category-key combo in block:",category,'-',k)
            continue
        else:
            savekey = category+"_"+k
            outdict[savekey]=kdat
    return outdict

def get_dist_from_pipelinedata(data,outdict=None,types=['d_a','h']):
    """
    Get distance functions out of datablock. 
    a, d_a, d_l, d_m, f_ap, h.txt,  mu.txt, rs_dv.txt, z
    """
    if data is None: #this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}
    category = 'distances'
    
    savekeys = types.copy()
    savekeys.append('z')
    savekeys.append('a')
    for k in savekeys:
        #print(block[category,k])
        try:
            kdat = data[category,k]
        except:
            print("Can't find category-key combo in block:",category,'-',k)
            continue
        else:
            savekey = category+"_"+k
            outdict[savekey]=kdat
    return outdict

def get_cmbcl_from_pipelinedata(data,outdict=None,types=['tt']):
    """
    Get cmb cl calcs from datablock. From camb module yaml, types are:
        tt: ell * (ell+1) C_ell^TT / 2 pi in mu K^2. Only if mode=cmb or all
        ee: ell * (ell+1) C_ell^EE / 2 pi in mu K^2. Only if mode=cmb or all
        bb: ell * (ell+1) C_ell^BB / 2 pi in mu K^2. Only if mode=cmb or all
        te: ell * (ell+1) C_ell^TE / 2 pi in mu K^2. Only if mode=cmb or all
        pp: ell * (ell+1) C_ell^PhiPhi. (phi-phi lensing spectruM) Only if mode=cmb or all"
    """
    if data is None: #this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}
    category = 'cmb_cl'
    
    savekeys = types.copy()
    savekeys.append('ell')
    for k in savekeys:
        #print(block[category,k])
        try:
            kdat = data[category,k]
        except:
            print("Can't find category-key combo in block:",category,'-',k)
            continue
        else:
            savekey = category+"_"+k
            outdict[savekey]=kdat
    return outdict

def get_lss_cl_from_pipelinedata(data,outdict=None,types='galaxy_cl'):
    """
    Get various LSS Cls from the datablock. options are:
    'galaxy_cl','galaxy_intrinsic_cl','galaxy_magnification_cl','galaxy_shear_cl','magnification_cl','magnification_intrinsic_cl','magnification_shear_cl','shear_cl','galaxy_cl_gg','galaxy_shear_cl_gg','shear_cl_gg'

    _gg means all the pieces combined, so e.g. 
      C_ell (galaxy_cl) is: galaxy_cl_gg + magnification_cl + galaxy_magnification_cl

    Honestly probably the twopointdict function could probably also do this, 
    but I'm setting it up myself here since the _gg names aren't in the typedict parsed by the twopoint
    cosmosis helper file's type table
    """
    if data is None:#this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}
    for category in types:
        try:
            basecat = category.replace('_gg','')
            # for derived cls, ell and bin number info not saved
            # refrence related calculation to get it
            nbin_a = data[basecat,'nbin_a']
            nbin_b = data[basecat,'nbin_b']
            auto_only = data[basecat,'auto_only']
            if auto_only:
                Ncorr = nbin_a
            else:
                Ncorr = nbin_a*nbin_b
            #sample_a = data[category]['sample_a']
            #sample_b = data[category]['sample_b']
            elldat = data[basecat,'ell']
            Nell = elldat.size
            Ndat = Ncorr*Nell
            x = np.tile(elldat,Ncorr)
            #print('x=',x)
            y = np.nan*np.ones(Ndat)
            if auto_only:
                bin1 = np.repeat(np.arange(1,nbin_a+1),Nell)
                bin2 = bin1
            else:
                bin1 = np.repeat(np.arange(1,nbin_a+1),Nell*nbin_b)
                tempbin2 = np.repeat(np.arange(1,nbin_b+1),Nell)
                bin2 = np.tile(tempbin2,nbin_a)
            # binformat will be bin_{0}_{1} or similar
            for i in range(nbin_a):
                b1=i+1
                for j in range(nbin_b):
                    if auto_only and j!=i:
                        continue
                    b2=j+1

                    if data.has_value(category,'bin_{}_{}'.format(b1,b2)):
                        yij = data[category,'bin_{}_{}'.format(b1,b2)]
                        y[np.logical_and(bin1==b1,bin2==b2)] = yij
                

            
                    
        except:
            print("Unable to read in LSS Cl  data for section {0}".format(category))
            continue

        ykey = category
        xkey = ykey+'_ell'
        outdict[xkey]=x
        outdict[ykey]=y
        outdict[ykey+'_bin1'] = bin1
        outdict[ykey+'_bin2'] = bin2
    return outdict


def get_nz_from_pipelinedata(data,outdict=None,samples=['lens','source']):
    """
    n(z) distributions from the datablock for some list of samples
    """
    if data is None: #this will happen if cosmosis failed
        return None
    if outdict is None:
        outdict = {}

    for s in samples:
        category = 'nz_'+s
        nbin = data[category,'nbin']
        nz = data[category,'nz']
        savekeys = ['bin_{}'.format(i) for i in range(1,nbin+1)]
        savekeys.append('z')
        savekeys.append('nbin')
        savekeys.append('nz')
        
        for k in savekeys:
            try:
                kdat = data[category,k]
            except:
                print("Can't find category-key combo in block:",category,'-',k)
                continue
            else:
                savekey = category+"_"+k
                outdict[savekey]=kdat
    return outdict

#-----------------------------------------------
# Functions for geting 2pt data from block. Note that
# this includes a number of helper files adapted from
# the cosmosis twopoint module. Most of these functions are
# copied nearly directly from blind_2pt_usingcosmosis.py
#-----------------------------------------------
def get_twoptdict_from_pipelinedata(data,outdict = None, get_sections = None):
    """
    Get 2pt funciton data vector info out of cosmosis datablock. 
    Uses type table that's part of the cosmosis twopoint module
    to identify names of twopoint functions to look for. 

    For auto spectra where z bin label order doesn't matter, both orders
    will be stored. (e.g. the same data will be there for bins 1-2 and 2-1).

    If outdict is passed as arg, adds data to that dictionary
    (if entries are already there with same keys, will overwrite them). 
    Otherwise creates and returns a dictionary with output data. 

    Section names can be (leaving out cmbkappas in this list, seee type file for more)
    shear_xi_plus, shear_xi_minus, galaxy_xi, galaxy_shear_xi
    galaxy_cl, shear_cl,  galaxy_shear_cl

    if get_sections is not None, only get info for the sections in that list.
    """
    if data is None: #this will happen if cosmosis failed
        return None

    if outdict is None:
        outdict = {}

    #print('get_sections',get_sections)
    
    # pull type table and some other from the cosmosis 2pt likelihood dir
    # This will only work if cosmosis environment has been set up
    if 'type_table' not in sys.modules: # don't bother if it's already imported
        csld = os.environ['CSL_DIR'] + '/'
        typefilepath = 'likelihood/2pt/twopoint_cosmosis.py'
        # get dir and file name
        dirname, filename = os.path.split(typefilepath)
        # split off .py
        impname, ext = os.path.splitext(filename)
        # add directory to path
        sys.path.insert(0, dirname)
        # import the library
        twopoint_cosmosis = __import__(impname)
        from twopoint_cosmosis import type_table
        #type_table #keys are (type1,type2), values are (section, x label, y label)
    
        
    # get list of possible sections
    for k in type_table.keys():
        section,  xlabel, binformat= type_table[k]
        #print(">>>",k,section)
        if get_sections is not None:
            if section not in get_sections:
                continue
        # xlabel is either ell or theta
        # binformat will be bin_{0}_{1} or similar
        if data.has_section(section):
            #print("   data has section!")
            types = k
            xkey,ykey = get_dictkey_for_2pttype(types[0],types[1],type_table)
            x,y,bin1, bin2, is_binavg, x_mins, x_maxs = spectrum_array_from_block(data,section,types, xlabel, binformat)

            outdict[xkey]=x
            outdict[ykey]=y
            outdict[ykey+'_bin1'] = bin1
            outdict[ykey+'_bin2'] = bin2
            outdict[ykey+'_binavg']=is_binavg
            if is_binavg:
                outdict[xkey+'_mins'] = x_mins
                outdict[xkey+'_maxs'] = x_maxs


    return outdict


def get_dictkey_for_2pttype(type1,type2,type_table):
    """
     Convert strings used in fits file to label spectra type in fits file to
     dictionary keys expected by this script's functions.
     (fits file 2pt table naming -> this script's naming)

    keys will be like:
    shear_xi_plus, shear_xi_plus_theta
    shear_xi_minus, shear_xi_minus_theta
    galaxy_xi, galaxy_xi_theta
    galaxy_shear_xi, galaxy_shear_xi_theta
   
    """
    firstdict = {'G':'galaxy', 'C':'cmb' }
    seconddict = {'P':'position','E':'shear_emode','B':'shear_bmode','+':'shear_plus','-':'shear_minus','K':'kappa'}
    thirddict = {'R':'real', 'F':'fourier' }
    
    if type1 in ("GPF","GEF","GBF","GPR","G+R","G-R","CKR","GPF","GEF","GBF"):
        #translate short codes into longer strings
        newtypes = ['_'.join([firstdict[t[0]],seconddict[t[1]],thirddict[t[2]]]) for t in [type1,type2]]
        type1 = newtypes[0]
        type2 = newtypes[1]           

    try:
        section, xlabel, ylabel = type_table[(type1,type2)]

        ykey = section
        xkey = section+'_'+xlabel
    except:
        raise ValueError("Spectra type not recognized: {0:s}, {1:s} ".format(type1,type2))

    return xkey,ykey

def spectrum_array_from_block(block, section_name, types, xlabel='theta', bin_format = 'bin_{0}_{1}'):
    """
    This script is nominally set up so that it could handle cases with and without
    angular bin averaging, but fair warning: I'm only really checking cases where bin averaging is on. 
    """
    if xlabel=='theta':
        is_binavg = block[section_name,"bin_avg"]
        # have theory calcs have been averaged over angular bins?
    else:
        is_binavg = False #no bin averaging in fourier space

    # for cross correlations we must save bin_ji as well as bin_ij.
    # but not for auto-correlations.
    is_auto = (types[0] == types[1])
    if block.has_value(section_name, "nbin"):
        nbin_a = block[section_name, "nbin"]
    else:
        nbin_a = block[section_name, "nbin_a"]
        nbin_b = block[section_name, "nbin_b"]


    #This is the ell/theta values that have been calculated by cosmosis,
    # if bin averaging, these should match what's in the fits files (up to rounding)
    # if interpolating, angles will be more densely sampled than values in fits files.
    theory_angle = block[section_name,xlabel]
    n_angle = len(theory_angle) #whatevers in the block, length of array
    if is_binavg:
        theory_angle_edges = block[section_name,xlabel+'_edges'] #angle bin edges

    #The fits format stores all the measurements
    #as one long vector.  So we build that up here from the various
    #bins that we will load in.  These are the different columns
    value = [] # two point function values
    bin1 = [] # z bin 1 indices
    bin2 = [] # z bin 2 indices
    angles = []
    # these will only be used if averaging over angular bins
    angle_mins = []
    angle_maxs = []

    #Bin pairs. Varies depending on auto-correlation
    for i in range(nbin_a):
        if is_auto:
            jmax = i+1
        else:
            jmax = nbin_b
        for j in range(jmax):
            #Load and interpolate from the block
            binlabel = bin_format.format(i+1,j+1)
            if block.has_value(section_name, binlabel):
                cl = block[section_name, binlabel] #two pt function values
                bin1.append(np.repeat(i + 1, n_angle))
                bin2.append(np.repeat(j + 1, n_angle))
                angles.append(theory_angle)
                if is_binavg:
                    angle_mins.append(theory_angle_edges[:-1])
                    angle_maxs.append(theory_angle_edges[1:])
                value.append(cl)

                if is_auto and i!=j: #also store under flipped z bin labels
                    # this allows the script to work w fits files uing either convention
                    bin1.append(np.repeat(j + 1, n_angle))
                    bin2.append(np.repeat(i + 1, n_angle))
                    value.append(cl)
                    angles.append(theory_angle)
                    if is_binavg:
                        angle_mins.append(theory_angle_edges[:-1])
                        angle_maxs.append(theory_angle_edges[1:])
                    

    #Convert all the lists of vectors into long single vectors
    value = np.concatenate(value)
    bin1 = np.concatenate(bin1)
    bin2 = np.concatenate(bin2)
    angles = np.concatenate(angles)
    if is_binavg:
        angle_mins = np.concatenate(angle_mins)
        angle_maxs = np.concatenate(angle_maxs)
    else:
        angle_mins = None
        angle_maxs = None

    return angles, value, bin1, bin2, is_binavg, angle_mins, angle_maxs

def theoryname_from_section(section):
    """
    names of 2pt functions as passed in ini file for calc and scale cut file
    """
    if section=='shear_xi_plus':
        return 'xip'
    elif section=='shear_xi_minus':
        return 'xim'
    elif section=='galaxy_xi':
        return 'wtheta'
    elif section=='galaxy_shear_xi':
        return 'gammat'
    else:
        return section

def section_from_theoryname(theoryname):
    if theoryname=='xip':
        return 'shear_xi_plus'
    elif theoryname=='xim':
        return 'shear_xi_minus'
    elif theoryname=='wtheta':
        return 'galaxy_xi'
    elif theoryname=='gammat':
        return 'galaxy_shear_xi'
    else:
        return theoryname
    
def get_twopt_cuts_from_ini(ini,outdict=None, get_sections = None,twoptlike_secname='2pt_like'):
    """
    Parse 2pt function scale cut settings. For each kind of 2pt function, adds entry
    to outdict with key 'twopointtype_scuts'. The value will be a Ncuts x 4 array. 
    
    column 0 = bin 1 index
    column 1 = bin 2 index
    column 2 = min angle 
    column 3 = max angle 
    """
    if outdict is None:
        outdict = {}
    options = ini.__dict__['_sections'][twoptlike_secname]
    scutkeys = []
    for k in options.keys():
        if k.startswith('angle_range'):
            #print('k',k,options[k])
            xmin,xmax = [float(x) for x in options[k].split()]
            #print(anglerange,type(anglerange[0]))
            
            theoryname,b1,b2 = (k.replace('angle_range_','')).split("_")
            ykey = section_from_theoryname(theoryname)
            scutkey = ykey+'_scuts'
            if scutkey in outdict.keys():
                outdict[scutkey].append([int(b1),int(b2),xmin,xmax])
            else:
                scutkeys.append(scutkey)
                outdict[scutkey] = [[int(b1),int(b2),xmin,xmax]]

    for k in scutkeys:
        #print('scutkey=',k)
        outdict[k] = np.array(outdict[k])

    return outdict
            
    
    
#############################################################################
# FUNCTIONS FOR collecting and saving sets of pk, growth, etc data
#############################################################################

#-----------------
# filename helpers
#-----------------
def get_datfile_2pt(fbase,twopt_section,datdir='output_datavec_calcs'):
    """
    twopt_section can be shear_xi_plus, shear_xi_minus, galaxy_xi, galaxy_shear_xi
    galaxy_cl, shear_cl,  galaxy_shear_cl
    """
    
    fname= '.'.join([fbase,'2pt',twopt_section,'dat'])
    return os.path.join(datdir,fname)

def get_datfile_pk(fbase,z=0,pktype='matter_power_nl',datdir='output_datavec_calcs'):
    """
    pktype can be 'matter_power_lin','matter_power_nl'
    """
    fname= '.'.join([fbase,'pk',pktype,'z{0:g}'.format(z),'dat'])
    return os.path.join(datdir,fname)

def get_datfile_grow(fbase,gtype='fsigma_8',datdir='output_datavec_calcs'):
    """
    gtype can be d_z, da, f_ap, f_z, fsigma_8, rs_dv, sigma_8
    """
    fname= '.'.join([fbase,'gro',gtype,'dat'])
    return os.path.join(datdir,fname)

def get_datfile_dist(fbase,dtype='d_a',datdir='output_datavec_calcs'):
    """
    dtype can be d_a, d_l, d_m, h, mu
    """
    fname= '.'.join([fbase,'dist',dtype,'dat'])
    return os.path.join(datdir,fname)

def get_datfile_cmbcl(fbase,dtype='tt',datdir='output_datavec_calcs'):
    """
    dtype can be tt, te, ee, bb
    """
    fname= '.'.join([fbase,'cmb',dtype,'dat'])
    return os.path.join(datdir,fname)


def get_datfile_lss_cl(fbase,dtype='galaxy_cl',datdir='output_datavec_calcs'):
    """
    dtype can be 'galaxy_cl','galaxy_intrinsic_cl','galaxy_magnification_cl','galaxy_shear_cl','magnification_cl','magnification_intrinsic_cl','magnification_shear_cl','shear_cl','galaxy_cl_gg','galaxy_shear_cl_gg','shear_cl_gg'
    """
    fname= '.'.join([fbase,'lss_cl',dtype,'dat'])
    return os.path.join(datdir,fname)

def get_datfile_nz(fbase,sample='lens',datdir='output_datavec_calcs'):
    """
    dtype should be the name of a sample for which there is a 
    datablock section called nz_[sample]
    """
    fname = '.'.join([fbase,'nz_{}'.format(sample),'dat'])
    return os.path.join(datdir,fname)

#-----------------
# workhorse function that loops through, calls cosmosis, saves files
#-----------------
def gen_dataset_list(pdictlist,labels,outfbase='paramtest',\
                     outdir='output_datavec_calcs',\
                     inifile='params_template.ini',callfrom = '',\
                     do_twopt = True, do_pk=True, do_growth=True,\
                     do_dist=True, do_cmb=True, do_nz=True,\
                     twopt_sections = default_twopt_sections, \
                     pk_sections = default_pk_sections, pk_zvals = [0],\
                     growth_sections = default_growth_sections,\
                     distance_sections = default_distance_sections,\
                     cmb_sections = default_cmb_sections,\
                     nz_samples = default_nz_samples,\
                     shifts=False, planck_module=None,\
                     do_lss_cl = True, lss_cl_sections = default_lss_cl_sections,\
                     campaign_yaml=None,campaign_run=None,\
):
    """
    Given list of dictionaries of parameter values, for each parameter
    dictionary in that list, run cosmosis and pull out info for desired sections.

    Generate output files where rows are entries in arrays extracted from
    the datablock, and columns correspond to different parameter dictionaries. 
    
    for 2pt output:
      filenames will be like: outfbase.shear_xi_plus (one per entry in twopt_sections)
      col 0: theta values (not bothering with theta min and max for now)
      col 1: bin 1 values
      col 2: bin 2 values
      col 3+: 2pt function y values for each entry in pdict list
    
    for pk output: assumes pk_zvals is not None, so separate files saved for each z value
      filenames will be like: outfbase.matter_power_lin.z0 (one combo from in pk_sections and pk_zvals)
      col 0: k values
      con 1+: P(k,zval) for each entry in pdict list

    for growth output:
      filenames will be like outfbase.sigma8_z (one per entry in growth_sections)
      col 0: z values
      col 1+: relevant growth function's for each entry in pdict list

    for nz output:
      filenames will be like: outfbase.nz_sample (one per sample)
      col 0: bin number
      col 1: z vals
      col 2+: n(z) for each entry in pdict list
      

    planck_module = string name of planck module; sometimes cosmosis complains
       if you initialize the plik likelihood twice, this gets around that 

    campaing_yaml, campaing_run - if these are passed, assumes you're using
     cosmosis v3 campaign setup instead of ini file
    """
    # check that cosmosis is initialized
    if 'COSMOSIS_SRC_DIR' not in os.environ:
        raise OSError("Cosmosis not initialized.")

    #from cosmosis.runtime.config import Inifile
    #from cosmosis.runtime.pipeline import LikelihoodPipeline
    #from cosmosis.datablock.cosmosis_py import block
    
    
    Ncosm = len(pdictlist)
    if do_twopt:
        twoptdat = [None for item in twopt_sections]
        twopt_scuts = [None for item in twopt_sections]
        # one entry per item in twopt_sections
        # each entry will be a Npoint x (Ncosm +5) array (first cols are theta,thetamin,thetamax bin1, bin2)
    if do_pk:
        pkdat = [None for item in pk_sections]
        # one entry per item in pk_sections
        # each entry is a Nz x Nk x (Ncosm+1) array (first col is k)
    if do_growth:
        growdat = [None for item in growth_sections]
        # one entry per item in growth_sections
        # each item is a Nz x (Ncosm + 1) array (first col is z)
    if do_dist:
        distdat = [None for item in distance_sections]
        # one entry per item in distance_sections
        # each item is a Nz x (Ncosm + 2) array (first cols are z and a)
    if do_cmb:
        cmbdat = [None for item in cmb_sections]
        # one entry per item in cmb_sections
        # each item is a Nz x (Ncosm + 1) array (first col is z)
    if do_lss_cl:
        cldat = [None for item in lss_cl_sections]
        # one entry per item in lss_cl-sections
        # each item is a Ndata X Ncosm + 3 array (first cols are ell, bin1, and bin2)
    if do_nz:
        nzdat = [None for item in nz_samples]
        # one entry per nz sample
        # each item will be a Ndata x Ncosm + 2 array (first cols are bin number, z)
        #  Ndata will be nbin*nz

    # this will be None if we don't have planck module
    planck_module_obj = get_planck_module(inifile,planck_module,callfrom,campaign_yaml=campaign_yaml,campaign_run=campaign_run)
    
    # if we have planck, get planck module object
    for i,pdict in enumerate(pdictlist):
        label = labels[i]
        print("=====================\n ON {0} \n=====================".format(label))
        
        datdict = run_cosmosis_togendat(pdict,inifile,callfrom,do_twopt,do_pk,\
                                        do_growth,do_dist, do_cmb, do_nz,\
                                        twopt_sections,pk_sections,\
                                        pk_zvals, growth_sections,distance_sections,\
                                        cmb_sections, nz_samples, shifts, \
                                        planck_module = planck_module, \
                                        planck_module_obj = planck_module_obj ,\
                                        do_lss_cl=do_lss_cl, lss_cl_sections = lss_cl_sections,\
                                        campaign_yaml=campaign_yaml,campaign_run=campaign_run,\
        )


        if datdict is None: #something went wrong for cosmosis 
            continue
        #print(datdict.keys())
        if do_twopt:
            print("  Getting 2pt data from datablock")
            for j,sec in enumerate(twopt_sections):
                ykey = sec
                #print('twoptdat[j] is None:',twoptdat[j] is None)
                if twoptdat[j] is None:
                    # create arrays to hold data 
                    # grab theta, bin info for first entry
                    if '_cl' in sec:
                        xkey = ykey+"_ell"
                    else:
                        xkey = ykey+"_theta"
                    b1key = ykey+'_bin1'
                    b2key = ykey+'_bin2'

                    skipsec=False
                    for k in [xkey,b1key,b2key,ykey]:
                        if k not in datdict.keys():
                            print("  can't find key",k)
                            skipsec=True
                            break
                        
                    if skipsec:
                        # don't have info to replace None entry with array
                        continue #go to next section
                    else:
                        #initialize array for section
                        ydat = datdict[ykey]
                        secarray = np.nan*np.ones((ydat.size,Ncosm+3))
                        secarray[:,0] = datdict[xkey]
                        secarray[:,1] = datdict[b1key]
                        secarray[:,2] = datdict[b2key]
                        secarray[:,i+3] = ydat
                        twoptdat[j] = secarray
                        
                        # look for scale cuts
                        scutkey = ykey+'_scuts'
                        #print(datdict.keys())
                        if scutkey in datdict.keys():
                            twopt_scuts[j] = datdict[scutkey]
                        else:
                            pass
                else:
                    secarray = twoptdat[j]
                    ydat = datdict[ykey]
                    secarray[:,i+3] = ydat
                    
        if do_pk:
            Nz = len(pk_zvals)
            #print("  Getting Pk data from datblock, Nzvals=",Nz)

            for j,sec in enumerate(pk_sections):
                if pkdat[j] is None: #initialize data array, get k alues
                    xkey = sec+'_k_h'
                    skipsec = False
                    if xkey not in datdict.keys():
                        print("  can't find key",k)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        kdat = datdict[xkey]
                        secarray = np.nan*np.ones((Nz,kdat.size,Ncosm+1))
                        for zi in range(Nz):
                            secarray[zi,:,0]=kdat
                        pkdat[j] = secarray
                secarray = pkdat[j]
                for zi,z in enumerate(pk_zvals):
                    ykey = sec+'_p_k_z{0:g}'.format(z)
                    if ykey in datdict.keys():
                        ydat = datdict[ykey]
                        #print(pdict)
                        #print('test1',ykey,ydat.shape)
                        secarray[zi,:,i+1] = ydat
                    
        if do_growth:
            growthcat = 'growth_parameters'
            for j,sec in enumerate(growth_sections):
                #print('gro test',j,sec)
                if growdat[j] is None: #intialize array and get z data
                    print('   initializing attempt')
                    zkey = growthcat+'_z'
                    skipsec = False
                    if zkey not in datdict.keys():
                        print("  can't find key",k)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        zdat = datdict[zkey]
                        Nz = zdat.size
                        secarray = np.nan*np.ones((Nz,Ncosm+1))
                        secarray[:,0] = zdat
                        growdat[j] = secarray
                secarray = growdat[j]
                ykey = growthcat+"_"+sec
                ydat = datdict[ykey]
                secarray[:,i+1] = ydat

        if do_dist:
            cat = 'distances'
            for j,sec in enumerate(distance_sections):
                if distdat[j] is None: #intialize array and get z data
                    zkey = cat+'_z'
                    akey = cat+'_a'
                    skipsec = False
                    if zkey not in datdict.keys():
                        print("  can't find key",zkey)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        zdat = datdict[zkey]
                        adat = datdict[akey]
                        Nz = zdat.size
                        secarray = np.nan*np.ones((Nz,Ncosm+2))
                        secarray[:,0] = zdat
                        secarray[:,1] = adat
                        distdat[j] = secarray
                secarray = distdat[j]
                ykey = cat+"_"+sec
                ydat = datdict[ykey]
                secarray[:,i+2] = ydat

        if do_cmb:
            cat = 'cmb_cl'
            for j,sec in enumerate(cmb_sections):
                if cmbdat[j] is None: #intialize array and get z data
                    ellkey = cat+'_ell'
                    skipsec = False
                    if ellkey not in datdict.keys():
                        print("  can't find key",k)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        elldat = datdict[ellkey]
                        Nell = elldat.size
                        secarray = np.nan*np.ones((Nell,Ncosm+1))
                        secarray[:,0] = elldat
                        cmbdat[j] = secarray
                secarray = cmbdat[j]
                ykey = cat+"_"+sec
                ydat = datdict[ykey]
                secarray[:,i+1] = ydat

        if  do_lss_cl:
            for j, sec in enumerate(lss_cl_sections):
                if cldat[j] is None: #intialize array and get ell and bin data
                    ellkey = sec+'_ell'
                    bin1key = sec+'_bin1'
                    bin2key = sec+'_bin2'
                    skipsec = False
                    if ellkey not in datdict.keys():
                        print("  can't find key",ellkey)
                        skipsec=True
                        break
                    if bin1key not in datdict.keys():
                        print("  can't find key",bin1key)
                        skipsec=True
                        break
                    if bin2key not in datdict.keys():
                        print("  can't find key",bin2key)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        elldat = datdict[ellkey]
                        bin1dat = datdict[bin1key]
                        bin2dat = datdict[bin2key]
                        Nell = elldat.size
                        secarray = np.nan*np.ones((Nell,Ncosm+3))
                        secarray[:,0] = elldat
                        secarray[:,1] = bin1dat
                        secarray[:,2] = bin2dat
                        cldat[j] = secarray
                secarray = cldat[j]
                ykey = sec
                ydat = datdict[ykey]
                secarray[:,i+3] = ydat

        if do_nz:
            for j, sample in enumerate(nz_samples):
                sec = 'nz_{}'.format(sample)
                nzkey = sec+'_nz'
                nbinkey = sec+'_nbin'
                zkey = sec+'_z'
                if nzdat[j] is None: #initialize array and get bin and z data
                    skipsec=False
                    if nbinkey not in datdict.keys():
                        print("  can't find key",nbinkey)
                        skipsec=True
                        break
                    if nzkey not in datdict.keys():
                        print("  can't find key",nzkey)
                        skipsec=True
                        break
                    if zkey not in datdict.keys():
                        print("  can't find key",zkey)
                        skipsec=True
                        break
                    if skipsec: #don't have info to replace None entry w array
                        continue # go to next section
                    else: #initialize array
                        nz = datdict[nzkey]
                        nbin = datdict[nbinkey]
                        zvals = datdict[zkey]
                        secarray = np.nan*np.ones((nz*nbin,Ncosm+2))
                        for n in range(nbin):
                            minind = n*nz
                            maxind = (n+1)*nz
                            secarray[minind:maxind,0] = n+1
                            secarray[minind:maxind,1] = zvals
                            nzdat[j] = secarray
                secarray = nzdat[j]
                nz = datdict[nzkey]
                nbin = datdict[nbinkey]
                for n in range(nbin):
                    minind = n*nz
                    maxind = (n+1)*nz
                    ykey = sec+'_bin_{}'.format(n+1)
                    ydat = datdict[ykey]
                    secarray[minind:maxind,i+2] = ydat
                    

    # now save that data to files
    # there may nan values if e.g. pipeline failed for some parameter values
    if do_twopt:
        for j,sec in enumerate(twopt_sections):
            if twoptdat[j] is None:
                print("No 2pt data for",sec)
                continue
            outf = get_datfile_2pt(outfbase,sec,outdir)
            if '_cl' in sec:
                xlabel = "ell"
            else:
                xlabel = "theta"
            header = sec+'\n'+'{0:s} bin1 bin2 '.format(xlabel)+' '.join(labels)
            print("Writing 2pt data to",outf)
            np.savetxt(outf,twoptdat[j],header=header,fmt='%0.6e')

            # save scale cut info if it is there
            if twopt_scuts[j] is None:
                print("No 2pt scale cuts for",sec)
            else:
                outf = get_datfile_2pt(outfbase,sec+"_scuts",outdir)
                header = sec+'\nbin1 bin2 anglemin anglemax'
                print("Writing 2pt scale cut data to",outf)
                np.savetxt(outf,twopt_scuts[j],header=header,fmt='%0.6e')
    if do_pk:
        for j,sec in enumerate(pk_sections):
            if pkdat[j] is None:
                print("No Pk data for",sec)
                continue
            for zi, z in enumerate(pk_zvals):
                outf = get_datfile_pk(outfbase,z,sec,outdir)
                
                header = sec+' at z={0:g}\n'.format(z)+'k[h/Mpc] '+' '.join(labels)
                print("Writing P(k,z) data to",outf)
                np.savetxt(outf,pkdat[j][zi,:,:],header=header,fmt='%0.6e')


    if do_growth:
        for j, sec in enumerate(growth_sections):
            if growdat[j] is None:
                print("No growth data for ",sec)
                continue
            outf = get_datfile_grow(outfbase,sec,outdir)
            header = sec+'\nz '+' '.join(labels)
            print("Writing grow data to",outf)
            np.savetxt(outf,growdat[j],header=header,fmt='%0.6e')


    if do_dist:
        for j, sec in enumerate(distance_sections):
            if distdat[j] is None:
                print("No distance data for ",sec)
                continue
            outf = get_datfile_dist(outfbase,sec,outdir)
            header = sec+'\nz a '+' '.join(labels)
            print("Writing distance  data to",outf)
            np.savetxt(outf,distdat[j],header=header,fmt='%0.6e')
    
    if do_cmb:
        for j, sec in enumerate(cmb_sections):
            if cmbdat[j] is None:
                print("No CMB Cl data for ",sec)
                continue
            outf = get_datfile_cmbcl(outfbase,sec,outdir)
            header = sec+'\nell '+' '.join(labels)
            print("Writing CMB Cl  data to",outf)
            np.savetxt(outf,cmbdat[j],header=header,fmt='%0.6e')

    if do_lss_cl:
        for j, sec in enumerate(lss_cl_sections):
            if cldat[j] is None:
                print("No LSS Cl data for ",sec)
                continue
            outf = get_datfile_lss_cl(outfbase,sec,outdir)
            header = sec+'\nell bin1 bin2 '+' '.join(labels)
            print("Writing LSS Cl data to", outf)
            np.savetxt(outf,cldat[j],header=header,fmt='%0.6e')

    if do_nz:
        for j, sample in enumerate(nz_samples):
            sec = 'nz_{}'.format(sample)
            if nzdat[j] is None:
                print("No nz data for ",sec)
                continue
            outf = get_datfile_nz(outfbase,sample,outdir)
            header = sec+'\nbin z '+' '.join(labels)
            print("Writing nz data to",outf)
            np.savetxt(outf,nzdat[j],header=header,fmt='%0.6e')
            
    return 0
    

#############################################################################
# FUNCTIONS FOR PLOTTING
#############################################################################
def addlines(ax,x,ygrid,dolab=False,lslist=None,collist=None,labels=None,fidind=None,fidlabel="",diffratio=False,lwlist=None):
    """
    Helper for various plotting scripts.
    """
    Nlines = ygrid.shape[1]

    if diffratio:
        fidy=ygrid[:,fidind]
        
    for i in range(Nlines):
        if np.all(ygrid[:,i]==np.nan):
            print("all nans for ",i)
            continue
        if lslist is None:
            ls = '-'
        else:
            ls = lslist[i]
        if lwlist is None:
            lw = 1.5
        else:
            lw= lwlist[i]
        if collist is not None:
            c = collist[i]
        else:
            c = None
        if dolab:
            label=labels[i]
            if i==fidind:
                label=label+' ('+fidlabel+')'
        else:
            label=""
        if diffratio:
            ax.plot(x,ygrid[:,i]/fidy -1.,color=c,ls=ls,lw=lw,label=label)
        else:
            ax.plot(x,ygrid[:,i],color=c,ls=ls,lw=lw,label=label)

    if not dolab: #plot fid line again on top in black
        if diffratio:
            ax.axhline(0,color='k',label=fidlabel,lw=1,ls='--')
        else:
            if fidind is not None:
                ax.plot(x,ygrid[:,fidind],color='k',ls='--',label=fidlabel,lw=1)
        
#------
def plot_func(xvals,ygrid,plottype='pk',labels=None,collist=None,lslist=None,\
              colvals=None,collabel=None,vmin=None,vmax=None,usecmap=None,\
              z=0,outftag='pktest',outdir='plots_datavecs',ftype='png',\
              justratio=False,noratio=False,typenote="",fidind=0,fidlabel='fid',\
              ylabel=None,yratiolabel=None,xlabel=None,\
              ymin=None,ymax=None,yrmin=None,yrmax = None, \
              xmin = None, xmax = None, dology = None, dologx=None,\
              save=True,legtitle='',hlines=[],vlines=[],lwlist=None):
    """
    Plot the variations in a function for different parameter values.
    plottype indicates what kind of function is being plotted. 
      pk - power spectrum 
      pklin - linear power spectrum (just adds subscript to y axis label)
      sig8 - sigma_8(z)
      fsig8 - fsigma8(z)
      h - H(z) (computeed as H/c, so multipy by c to get km/s/Mpc units)
      d_a - D_A(z)
      cmb_cl_tt, cmb_cl_ee ... - CMB Cl (can be tt, ee, te, bb)
      other - something else, will look for use ylabel,yratiolabel,xlabel for labels
      

    k - single array of k values in Mpc/h to use for all power spectra
    ygrid - Nk x Nlines array of power spectra or whatever function we're dealing with

    There are two ways to label the lines: legend labels, or a color scale.

    legend label approach: used if labels is not None
        labels - list of string labels corresponding to each line plotted. 
        collist - optional, list of colors to use for the lines

    line color approach: color of line set by where numerical value falls on color scale
        colvals - Ncosm-entry array with associated number for each line
        collabel - string to use to label color scale
        vmin - bottom value for colorbar; default is min of colvals
        vmax - top value for colorbar; default is max of colvals
        usecmap - optional matplotlib.cm object specifying color map to use

    lslist = list of linestyles for each cosm. If None, will set all to solid.
    

    if justratio: single panel plot just showing ratio between fid and others. 
                  otherwise shows both power spectrum and ratio. 
    if noratio: single panel plot just showing main funciton, no ratio panel

    
    typenote - string to use if you want to include an annotatio 
          about type of power spectrum (e.g. to distinguish linear vs non)
    z = z value that power spectra were computed at. If not None
         this info will go in the legend title. 

    fidind = index of cosm to use as ratio denominator
    fidlabel = short label for fid cosm to use in axis label

    outftag - output filename will be outdir/plottype.outftag.ftype
    outdir - directory where to put the plot
    ftype  - file type of plot    
    """
    ckmpers=299792.
    dolab = labels is not None
    if dology is None:
        dology = True
    if dologx is None:
        dologx  = True

    if plottype=='h':
        ygrid = ckmpers*ygrid.copy()
    # find fid y, compute diffratios
    yfid = ygrid[:,fidind]
    yfidgrid = np.tile(yfid,(ygrid.shape[1],1)).T
    ydiffratio = ygrid/yfidgrid -1
    whereinf = np.logical_not(np.isfinite(ydiffratio))
    ydiffratio[whereinf] = np.nan

    if not np.sum(np.isfinite(ygrid)):
        print("No finite y values, skipping plot for",plottype,outftag,typenote)

    # . . . . . . . . . . . . . . . . . . . . . .
    # Set up colors
    # . . . . . . . . . . . . . . . . . . . . . .
    if not dolab: #set up colormap
        # perceptually uniform sequential options: viridis, inferno, cividis
        # diverging: coolwarm, Spectral, RdBu
        if vmin is None:
            vmin = np.min(colvals)
        if vmax is None:
            vmax = np.max(colvals)
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        if usecmap is None:
            usecmap = matplotlib.cm.RdBu
        cmap = matplotlib.cm.ScalarMappable(norm=norm,cmap=usecmap)
        cmap.set_array([])
        collist = cmap.to_rgba(colvals)

    # . . . . . . . . . . . . . . . . . . . . . .
    # Set up axis labels and plot-type specific x ranges
    # . . . . . . . . . . . . . . . . . . . . . .
    outtypetag = plottype
    if plottype in ['pk','pklin','matter_power_nl','matter_power_lin','weyl_curvature_spectrum','weyl_curvature_spectrum_nl']:

        if plottype in ('pklin','matter_power_lin'):
            outtypetag = 'pk.matter_power_lin'
            substr = '_{{\\rm lin}}'
        elif plottype in ('pk','matter_power_nl'):
            outtypetag = 'pk.matter_power_nl'
            substr = ''
        elif plottype=='weyl_curvature_spectrum':
            outtypetag = 'pk.weyl_power_lin'
            substr = '_{{\Phi{{\\rm,lin}}}}'
        elif plottype=='weyl_curvature_spectrum_nl':
            outtypetag = 'pk.weyl_power_nl'
            substr = '_{{\Phi}}'
            
        if z is not None:
            ylabel=r'$P{1:s}(k,z={0:0g})$'.format(z,substr)
        else:
            ylabel = r'$P(k)$'
        yratiolabel=r'$\Delta P{1:s}(k)/P{1:s}^{{\rm {0:s}}}(k)$'.format(fidlabel,substr)
        xlabel = r'$k$ [Mpc/$h$]$^{-1}$'
        if xmax is None:
            xmax=1.
        if xmin is None:
            xmin=1.e-3
        
    elif plottype in ['sig8','fsig8','d_z','f_z','fsigma_8','sigma_8','d_a','h']:
        #functions of z
        xlabel =r'$z$'
        dology = False
        if xmin is None:
            xmin=max(np.nanmin(xvals),0.1)
        outtypetag = 'gro.'+plottype
        z=None
        
        if plottype in ('sig8','sigma_8'):
            outtypetag = 'gro.sigma_8'
            ylabel = r'$\sigma_8(z)$'
            yratiolabel=r'$\Delta\sigma_8/\sigma_8^{{\rm {0:s}}}$'.format(fidlabel)
        elif plottype in ('fsig8', 'fsigma_8'):
            outtypetag = 'gro.fsigma_8'
            ylabel = r'$f(z)\sigma_8(z)$'
            yratiolabel=r'$\Delta[f\sigma_8]/[f\sigma_8]^{{\rm {0:s} }}$'.format(fidlabel)
        elif plottype in ('d_z'):
            ylabel = r'$D(z)$'
            yratiolabel=r'$\Delta D/D^{{\rm {0:s} }}$'.format(fidlabel)
        elif plottype in ('f_z'):
            ylabel = r'$f(z)$'
            yratiolabel=r'$\Delta f/f^{{\rm {0:s} }}$'.format(fidlabel)
        elif plottype in ('sigma2_vdelta_8_z'):
            ylabel = r'$\sigma^2_{(v \delta)8}$'
            yratiolabel=r'$\sigma^2_{{(v \delta)8}}/[\sigma^2_{{(v \delta)8}}]^{{\rm {0:s} }}$'.format(fidlabel)
        elif plottype in ('d_a'):
            outtypetag = 'dist.d_a_z'
            ylabel = r'$D_A(z)$ [Mpc/$h$]'
            yratiolabel=r'$\Delta D_A/D_A^{{\rm {0:s}}}$'.format(fidlabel)
        elif plottype in ('h'):
            outtypetag = 'dist.h_z'
            ylabel = r'$H(z)$ [km/s/Mpc]'
            yratiolabel=r'$\Delta H/H^{{\rm {0:s}}}$'.format(fidlabel)

            
    elif plottype in ['cmb_cl_tt','cmb_cl_te','cmb_cl_ee','cmb_cl_bb','cmb_cl_bb','cmb_cl_pp']:
        #cmb fucntions
        xlabel = r'$\ell$'
        dology = False
        end= plottype[-2:]
        unitstr = r' [$\mu$K$^2$]'
        twopistr = r'$(2\pi)^{-1}'
        z=None
        if end=='tt':
            texlabel =r'TT'
        elif end=='ee':
            texlabel =r'EE'
        elif end=='te':
            texlabel =r'TE'
        elif end=='bb':
            texlabel =r'BB'
        elif end=='pp':
            texlabel =r'\Phi\Phi'
            unitstr = ''
            twopistr='$'
        outtypetag='cmb.'+end
        ylabel = twopistr+r'\ell(\ell+1)C_{{\ell}}^{{{0:s}}}$'.format(texlabel)+unitstr
        yratiolabel=r'$\Delta C_{{\ell}}^{{{0:s}}}/[C_{{\ell}}^{{{0:s}}}]^{{\rm {1:s}}}$'.format(texlabel,fidlabel)
        

    # . . . . . . . . . . . . . . . . . . . . . .
    # sort out plot bounds
    # . . . . . . . . . . . . . . . . . . . . . .
    if xmin is None:
        xmin = np.nanmin(xvals)
    if xmax is None:
        xmax = np.nanmax(xvals)

    usexinds = (xvals>=xmin)*(xvals<=xmax)
    if not justratio:
        if ymax is None:
            if dology:
                ymax = 2*np.nanmax(ygrid[usexinds,:])
            else:
                yrange = np.nanmax(ygrid[usexinds,:]) - np.nanmin(ygrid[usexinds,:])
                ymax = np.nanmax(ygrid[usexinds,:]) + 0.1*yrange
        if ymin is None:
            if dology:
                ymin = 0.5*np.nanmin(ygrid[usexinds,:])
            else:
                yrange = np.nanmax(ygrid[usexinds,:]) - np.nanmin(ygrid[usexinds,:])
                ymin = np.nanmin(ygrid[usexinds,:]) - 0.1*yrange

    if yrmax is None:
        yrmax = np.nanmax(ydiffratio[usexinds,:])
        if yrmax>0:
            yrmax = max(1.1*yrmax,0.1)
        else:
            yrmax = .8*yrmax
    if yrmin is None:
        yrmin = np.nanmin(ydiffratio[usexinds,:])
        if yrmin>0:
            yrmin = .8*yrmin
        else:
            yrmin = min(1.2*yrmin,-0.1)
     
    #. . . . . . . . . . . . . . . . . . . .
    # Prep annotation with whatever we want written on the plot
    # . . . . . . . . . . . . . . . . . . . . . .
    
    if typenote or (z is not None):
        if z is not None:
            zstr = '$z={0:g}$'.format(z)
        else:
            zstr=''
        if zstr and typenote:
            bothstr = '\n '
        else:
            bothstr = ''
        infonote = typenote+bothstr+zstr
    else:
        infonote = ""

    #. . . . . . . . . . . . . . . . . . . .
    # Make the plot
    # . . . . . . . . . . . . . . . . . . . . . .
    if justratio or noratio: #one panel plot setup
        if dolab: 
            fig,ax = plt.subplots(1,1,figsize=(5,3))
        else: #need to make room for colorbar
            fig,ax = plt.subplots(1,1,figsize=(5,3))
        if dologx:
            ax.set_xscale("log", nonpositive='clip')
    else: # 2 panel plot
        if dolab:
            fig,axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]},sharex=True,figsize=(5,4))
        else: #make room for colorabar
            fig,axes = plt.subplots(2,1,gridspec_kw={'height_ratios':[2,1]},sharex=True,figsize=(5,4))
            #fig = plt.figure(constrained_layout=True)
            #gridspec = fig.add_gridspec(2,2,height_ratios=[2,1],width_ratios=[10,1])
            #axes = [fig.add_subplot(gridspec[0,0]),fig.add_subplot(gridspec[1,0])]
            #cbarax = fig.add_subplot(gridspec[:,1])
        ax = axes[0] # power spec log log plot

    if not justratio:
        if dologx:
            ax.set_xscale("log", nonpositive='clip')
        if dology:
            ax.set_yscale("log", nonpositive='clip')
        ax.set_ylim((ymin,ymax))
        ax.set_xlim((xmin,xmax))
        ax.set_ylabel(ylabel)
        plt.subplots_adjust(hspace=0)
        ax.tick_params(axis='x',which='both',bottom=True,top=True,direction="in")
        for hl in hlines:
            ax.axhline(hl,ls='-',color='darkgray',lw=0.5)
        for vl in vlines:
            ax.axvline(vl,ls='-',color='darkgray',lw=0.5)

        addlines(ax,xvals,ygrid,dolab,lslist,collist,labels,fidind,fidlabel,lwlist=lwlist)

        ax.legend(title=legtitle)
        if infonote:
            ax.annotate(infonote,(.05,.05),xycoords = 'axes fraction',\
                        color='gray', ha='left', va='bottom',fontsize=8)
        #if not dolab: #add in colorbar
        #    fig.colorbar(cmap,label=collabel)
        if not noratio:
            ax = axes[1]

    #. . . . . . . . . . . . . . . . . . . .
    # plot ratios
    if not noratio:
        ax.tick_params(axis='x',which='both',bottom=True,top=True,direction="in")
        ax.set_ylim((yrmin,yrmax))
        ax.set_ylabel(yratiolabel)
        ax.set_xlabel(xlabel)

        addlines(ax,xvals,ydiffratio,dolab,lslist,collist,labels,fidind,fidlabel,lwlist=lwlist)#,diffratio=True)

        if justratio:
            ax.legend(title=legtitle)

        if infonote and justratio:
            ax.annotate(infonote,(.05,.05),xycoords = 'axes fraction',\
                    color='gray', ha='left', va='bottom',fontsize=8)

    #. . . . . . . . . . . . . . . . . . . .
    # tight layout and color bar if needed
    #. . . . . . . . . . . . . . . . . . . .
    if dolab:
        plt.tight_layout(h_pad=0,w_pad=0)
    else:  #add in colorbar (need to place after tight layout)
        plt.tight_layout(h_pad=0,w_pad=0,rect=[0,0,.97,1])
        fidcol=colvals[fidind]
        if justratio or noratio:
            cbar=fig.colorbar(cmap,label=collabel,ax=ax,pad=0,aspect=20)
        else:
            cbar=fig.colorbar(cmap,label=collabel,ax=axes,pad=0,aspect=20)
        cbar.ax.axhline(fidcol,ls='--',color='k',lw=1)

    if save:
        outf = '.'.join([outftag,outtypetag,ftype])
        saveto = os.path.join(outdir,outf)
        print("Saving plot to",saveto)
        plt.savefig(saveto)
        plt.clf()
    plt.close(fig)

#------
def plot_2pt_grid(xvals=None,bin1=None,bin2=None,ygrid=None, plottype='galaxy_xi',\
                  scaleby=None, sym=None, autoonly=None,\
                  doratio=False, fidind=0, fidlabel='fid',showfid=True,\
                  labels=None,collist=None,lslist=None,lwlist=None,\
                  colvals=None,collabel=None,vmin=None,vmax=None,usecmap=None,\
                  outftag='twopttest',outdir='plots_datavecs',ftype='png',\
                  note="",xlabel=None,firstxlabel = None, ylabel=None, ylabeltag=None,\
                  angles_in_radians=True,scutgrid=None,\
                  save=True,maxdiff=1.5,ylim_input=None,\
                  isratio=False, ratio_label = None,fitsfile='',fitslabel='',\
                  scalefitsby = None, addtofits = 0, centerfitsat = None,\
                  morefits=[],morefitscols=[],morefitslabels=[],morefitserrors=False,\
                  justfits=False,marksize=2,legtitle='',hlines=[],vlines=[],\
                  doratio_errordenom=False,doratio_nodiff=False,\
                  hidefits=False,hidelegend=False):

    """
    Plot 2pt grid. Color scales and labels work the same way as plot_func. 
    xvals - can be either theta or ell. 1d array with Npoint entries
    bin 1 and bin 2 are 1d arryays of z bin labels, length is Npoint
    ygrid  - array of 2pt values, Npoint x Nline (assume all entries have same x and bin arrays)
    ^ can skip these if justfits=True (just plotting data from fits file)

    sym - bool, True if bin1 and bin2 tracers are the same so we just want 
         lower triangle of plot grid rather than whole rectangle
    autoonly - if True, just plots cases where bin1=bin2

    scaleby - set automatically but can optionally pass
            Npoint sized array to multiply by y values for plotting
            (used to show e.g. w(theta)*theta rather than just w(theta))

    doratio - if true, use fidind to plot fractional difference with fid
    doratio_nodiff - if True, use fid ind to plot ratio with fid
    doratio_errordenom - only available if fits file is being plotted. if True, 
               plot fractional difference, denominator set to error bar size
               rather than fiducial model prediction
    showfid - used only if doratio is False; if True, puts dashed black
             line for fid entry. 

    angles_in_radians - by default, cosmosis angles will be in radius. 
         if this is True, will convert xvals to armin before plotting
          (will only do so if string 'theta' is in the xlabel)

    May eventually want to add ablity to plot errors and scale cuts,
     but not set up to do that yet. 

    if scutgrid is not None, expect array with columns: bin1, bin2, min angle, maxangle
      if passed, will shade regions outside scale cuts. Assumes angles are already
       in arcmin units. 

    maxdiff - most y scaling will be done automatically, but to protect against cases
        when ploting diffs where denominator is close to zero, use this parameter to set 
        the largest difference to be shown
    ylim_input - (min,max) tuple for nonratio plots; specifying ymin and max. 
        will use narrowest combo of these input limits and the default limits from the plot

    if isratio, you're telling the function that you're alreayd passing in a ratio.
    This means the various scalings won't be applied. Plot axes will be labeled
    with the optional ratio_label. If it isn't given, will just do e.g. 'xi ratio'

    if fitsfile is passed, add datapoints from it in black.
    currently fitslabel does nothing, but e.g. could be used for making a legend later
    morefits - if we want points from other fits files, pass them here as a list
             NOTE: will assume bins and angle values match those of first fits file
    morefitscols - colors for other datapoints from morefits (will default to gray)
    morefitserrors -bool, if True puts error bars on extra datapoints, if not, doesn't
    morefitslabels - labels for addtional datapoints
    justfits = set to True if we have no theory DV, just want plot fits

    if we're using doratio=True (e.g. passing in ratios of various theory predictions), 
    we can optionally use these variables to add error bars from the fits file
       scalefitsby = array of values in same format as ygrid 2pt functs, will multiply fits 
                      datapoints and error bars
       addtofits = constant to add to scaled fits values
       centerfitsat = where to center fits values; None uses values in fits file, 
                   could also set e.g. to zero or 1
       i.e. if these are used, fits data points will be at
              y_fits*scalefitsby + addtofits   or at centerfitsat with just error bars scaled
       hidefits - used only with doratio_errordenom; if we want to use the fits file
             just to grab error bars, but don't want to plot datapoints, set this to True

    """
    
    # . . . . . . . . . . . . . . . . . . . . . .
    # set up labels and some default settings for known 2pt functions
    # . . . . . . . . . . . . . . . . . . . . . .
    if doratio_errordenom or doratio_nodiff:
        # make sure scalings etc are right
        doratio=True
    outbase='twopt'
    dolab = labels is not None
    doxlog= True
    doylog=False
    if plottype in ('galaxy_xi','wtheta'):
        sym = True
        if autoonly is None:
            autoonly=True

        if isratio:
            if ratio_label is None:
                ylabel = r'$w(\theta)$ ratio'
                    
            else:
                ylabel = ratio_label
        elif doratio:
            ylabel = r'$\Delta w/w^{{\rm {0:s}}}$'.format(fidlabel)
            if doratio_nodiff:
                ylabel = r'$w/w^{{\rm {0:s}}}$'.format(fidlabel)
            if doratio_errordenom:
                ylabel = r'$(w-w^{{\rm {0:s}}})/\sigma_w$'.format(fidlabel)
        else:
            ylabel = r'$\theta w$ [arcmin]'
        xlabel = r'$\theta$'
        firstxlabel = r'$\theta$ [arcmin]'
        if not justfits:
            if angles_in_radians:
                xvals = xvals*rad_to_arcmin
            if isratio or doratio:
                scaleby=1.
            else:
                scaleby = xvals

        
    elif plottype in ('shear_xi_plus','xip','shear_xi_minus','xim'):
        sym = True
        if autoonly is None:
            autoonly=False
        if isratio:
            if ratio_label is None:
                if plottype in ('shear_xi_plus','xip'):
                    ylabel = r'$\xi_+$ ratio'
                elif plottype in ('shear_xi_minus','xim'):
                    ylabel = r'$\xi_-$ ratio'
            else:
                ylabel = ratio_label
        elif not doratio:
            if plottype in ('shear_xi_plus','xip'):
                ylabel = r'$\theta \xi_+$  [$10^{-4}$ arcmin]'
                
            elif plottype in ('shear_xi_minus','xim'):
                ylabel = r'$\theta \xi_-$ [$10^{-4}$ arcmin]'
                
        else:
            if plottype in ('shear_xi_plus','xip'):
                ylabel = r'$\Delta \xi_+/\xi_+^{{\rm {0:s}}}$'.format(fidlabel)
                if doratio_nodiff:
                    ylabel =r'$\xi_+/\xi_+^{{\rm {0:s}}}$'.format(fidlabel)
                if  doratio_errordenom:
                    ylabel = r'$(\xi_+-\xi_+^{{\rm {0:s}}})/\sigma_{{\xi_+}}$'.format(fidlabel)
            elif plottype in ('shear_xi_minus','xim'):
                ylabel = r'$\Delta \xi_-/\xi_-^{{\rm {0:s}}}$'.format(fidlabel)
                if doratio_nodiff:
                    ylabel = r'$\xi_-/\xi_-^{{\rm {0:s}}}$'.format(fidlabel)
                if  doratio_errordenom:
                    ylabel = r'$(\xi_--\xi_-^{{\rm {0:s}}})/\sigma_{{\xi_-}}$'.format(fidlabel)
        xlabel = r'$\theta$'
        firstxlabel = r'$\theta$ [arcmin]'
        if not justfits:
            if angles_in_radians:
                xvals = xvals*rad_to_arcmin
            if isratio or doratio:
                scaleby=1.
            else:
                scaleby=xvals*1.e4
            
    elif plottype in ('galaxy_shear_xi','gammat'):
        sym = False
        if autoonly is None:
            autoonly=False
        xlabel = r'$\theta$'
        firstxlabel = r'$\theta$ [arcmin]'
        if not justfits:
            if angles_in_radians:
                xvals = xvals*rad_to_arcmin
            if isratio or doratio:
                scaleby = 1
            else:
                scaleby=xvals*1.e2
        if isratio:
            if ratio_label is None:
                ylabel = r'$\gamma_t$ ratio'
            else:
                ylabel = ratio_label
        elif doratio:
            ylabel = r'$\Delta \gamma_t/\gamma_t^{{\rm {0:s}}}$'.format(fidlabel)
            if doratio_nodiff:
                ylabel = r'$ \gamma_t/\gamma_t^{{\rm {0:s}}}$'.format(fidlabel)
            if  doratio_errordenom:
                ylabel = r'$(\gamma_t-\gamma_t^{{\rm {0:s}}})/\sigma_{{\gamma_t}}$'.format(fidlabel)
        else:
            ylabel = r'$\theta \gamma_t$ [$10^{-2}$ arcmin]'
    elif '_cl' in plottype:
        outbase='lsscl'
        scaleby=xvals
        fitsfile="" #can't do datapoint plots here
        if autoonly is None:
            autoonly = plottype in ['galaxy_cl','galaxy_cl_gg']
        sym = plottype in ['shear_cl','shear_cl_gg','magnification_cl']
        xlabel = r'$\ell$'
        texdict={'galaxy_cl':r'GG','galaxy_intrinsic_cl':r'GI','galaxy_magnification_cl':r'GM','galaxy_shear_cl':r'G\kappa','magnification_cl':r'MM','magnification_intrinsic_cl':r'MI','magnification_shear_cl':r'M\kappa','shear_cl':r'\kappa\kappa','galaxy_cl_gg':r'GG{\rm (full)}','galaxy_shear_cl_gg':r'G\kappa{\rm(full)}','shear_cl_gg':r'\kappa\kappa{\rm (full)}'}
        
        if isratio:
            if ratio_label is None:
                ylabel = r'$C_{{\ell}}^{{{0:s}}}$ ratio'.format(texdict[plottype])
            else:
                ylabel = ratio_label
        elif doratio:
            ylabel = r'$\Delta C_{{\ell}}^{{{0:s}}}/[C_{{\ell}}^{{{0:s}}}]^{{\rm {1:s}}}$'.format(texdict[plottype],fidlabel)
            if doratio_nodiff:
                ylabel = r'$C_{{\ell}}^{{{0:s}}}/[C_{{\ell}}^{{{0:s}}}]^{{\rm {1:s}}}$'.format(texdict[plottype],fidlabel)
        else:
            doylog=True
            ylabel = r'$\ell |C_{{\ell}}^{{{0:s}}}|$'.format(texdict[plottype])
            ygrid = np.fabs(ygrid)
            
    else:
        if not justfits:
            if 'theta' in xlabel and angles_in_radians:
                xvals = xvals*rad_to_arcmin
        
    if firstxlabel is None:
        firstxlabel = xlabel

    if ylabeltag is not None:
        ylabel = ylabel + ": "+ylabeltag

    # . . . . . . . . . . . . . . . . . . . . . .
    # take ratios or multiply in scaling factors as appropraite
    # . . . . . . . . . . . . . . . . . . . . . .
    if not justfits:
        if isratio:
            ploty = ygrid
        elif doratio:
            yfid= ygrid[:,fidind]
            yfidgrid = np.tile(yfid,(ygrid.shape[1],1)).T
            if doratio_errordenom:
                # to get scaling, need to get errors from fits file to scale
                # we'll do this plot by plot, not here
                ploty=(ygrid-yfidgrid)
            elif doratio_nodiff:
                ploty = ygrid/yfidgrid
            else:
                ploty = ygrid/yfidgrid-1
        else:
            scalebygrid = np.tile(scaleby,(ygrid.shape[1],1)).T
            ploty = ygrid*scalebygrid

    # . . . . . . . . . . . . . . . . . . . . . .
    # Get points out of the fits file
    # . . . . . . . . . . . . . . . . . . . . . .
    if fitsfile:# and not isratio:
        if isratio and (scalefitsby is None) :
            # scalefitsby will  match theory calc format
            # centerfitsat: None will use values in fits file, could also specify
            #  constant values like zero or 1 if we just want errors for reference
            print("Skipping plot because we can't show fits points for an input ratio without information about how to scale it.")
            return
        # fits files use slightly different labeling than datablock
        labeltranslate = {'galaxy_shear_xi':'gammat','shear_xi_plus':'xip','shear_xi_minus':'xim','galaxy_xi':'wtheta'}
        
        if plottype in labeltranslate.keys():
            fitstype = labeltranslate[plottype]
        else:
            fitstype = plottype
        fitsdat = fits.open(fitsfile)
        tab = fitsdat[fitstype]
        tab_bin1 = tab.data['BIN1']
        tab_bin2 = tab.data['BIN2']
        tab_ang = tab.data['ANG'] #will be in arcmin
        tab_y = tab.data['VALUE']

        if morefits: #we'll assume all the fits files ahve same bin and angle arrays
            more_tab_y = [(fits.open(f)[fitstype]).data['VALUE'] for f in morefits]

        # do needed scaling of datapoints
        # (won't just use scaleby from above because angles saved may be different)
        if isratio or doratio:
            fitscale=1.
        elif fitstype=='wtheta':
            fitscale = tab_ang
        elif fitstype=='gammat':
            fitscale = tab_ang*1.e2
        elif fitstype in ['xip','xim']:
            fitscale = tab_ang*1.e4

        tab_y = tab_y * fitscale
        if morefits:
            more_tab_y = [y*fitscale for y in more_tab_y]

        # let's also get diagonal cov entries for error bars
        covtab = fitsdat['COVMAT']
        for i in range(4):
            trystr = 'NAME_{}'.format(i)
            if trystr in covtab.header:
                if covtab.header[trystr]==fitstype:
                    startind=covtab.header['STRT_{}'.format(i)]
                    endind = None
                    if 'STRT_{}'.format(i+1) in covtab.header:
                        endind = covtab.header['STRT_{}'.format(i+1)]
                    break
        if endind is None:
            fitserr = np.sqrt(np.diag(covtab.data[startind:,startind:]))*fitscale
        else:
            fitserr = np.sqrt(np.diag(covtab.data[startind:endind,startind:endind]))*fitscale
            # this will be just sqrt of diagonal entries of covmat corresponding to the
            #  2pcf we're looking at now
            # assuming these are in the same order as in tab_y (for angles and z bins)

    # . . . . . . . . . . . . . . . . . . . . . .
    # Set up colors
    # . . . . . . . . . . . . . . . . . . . . . .
    if (not dolab) and (not justfits): #set up colormap
        # perceptually uniform sequential options: viridis, inferno, cividis
        # diverging: coolwarm, Spectral, RdBu
        if vmin is None:
            vmin = np.min(colvals)
        if vmax is None:
            vmax = np.max(colvals)
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        if usecmap is None:
            usecmap = matplotlib.cm.RdBu
        cmap = matplotlib.cm.ScalarMappable(norm=norm,cmap=usecmap)
        cmap.set_array([])
        collist = cmap.to_rgba(colvals)
        #print('colvals.size',colvals.shape)
        #print('collist.size',collist.shape)
        
    # figure out bin numbers and structures
    if not justfits:
        Nbin1 = np.max(bin1) #bin numbers start at 1 not 0
        Nbin2 = np.max(bin2)
    else:
        Nbin1 = np.max(tab_bin1) #bin numbers start at 1 not 0
        Nbin2 = np.max(tab_bin2)
    if autoonly:
        Nrow=1
    else:
        Nrow = Nbin1
    Ncol = Nbin2

    fig,axes = plt.subplots(Nrow,Ncol,figsize=(7,.8*(1+Nrow)),sharex=True,sharey=True)
    plt.subplots_adjust(hspace=0,wspace=0)
    

    for row in range(Nrow):
        for col in range(Ncol):
            if Nrow==1 and Ncol==1:
                ax = axes
            elif Nrow==1:
                ax = axes[col]
            elif Ncol==1:
                ax = axes[row]
            else:
                ax = axes[row,col]
            ax.minorticks_on()
            if doylog:
                ax.set_yscale("log", nonpositive='clip')
            if autoonly:
                b1 = col+1
                b2 = col+1
            else:
                b1 = row+1
                b2 = col+1

            if sym and b2>b1:
                # don't plot anything
                ax.axis('off')
                continue
            else:
                ax.tick_params(axis='x',which='both',bottom=True,top=True,direction="in")
                ax.tick_params(axis='y',which='both',left=True,right=True,direction="in")
                for hl in hlines:
                    ax.axhline(hl,ls='-',color='darkgray',lw=0.5)
                for vl in vlines:
                    ax.axvline(vl,ls='-',color='darkgray',lw=0.5)
            #print("===========>>>>>>>",b1,b2,plottype)
            if row == Nrow-1: #want x label
                if col==0:
                    ax.set_xlabel(firstxlabel)
                else:
                    ax.set_xlabel(xlabel)
            if doxlog:
                ax.set_xscale("log", nonpositive='clip')
            # get mask for bin indices

            #axmask = np.where((bin1==b1)*(bin2==b2))[0]
            axmask = (bin1==b1)*(bin2==b2)

            # set up some fits file read-in if we have one:
            if fitsfile:# and not isratio:

                #fits_axmask = np.where((tab_bin1==b1)*(tab_bin2==b2))[0]
                fits_axmask = (tab_bin1==b1)*(tab_bin2==b2)
                #print(fits_axmask)
                if fitstype in ['wtheta','xip','xim']:
                    if not np.any(fits_axmask):
                        # for symmetric 2pts, if bins don't match anywhere, try flipped order
                        #fits_axmask = np.where((tab_bin1==b2)*(tab_bin2==b1))[0]
                        fits_axmask = (tab_bin1==b2)*(tab_bin2==b1)

                plot_tab_x = tab_ang[fits_axmask]
                plot_tab_y = tab_y[fits_axmask]
                plot_tab_yerr = fitserr[fits_axmask]
                if morefits:
                    plot_more_tab_y = [y[fits_axmask] for y in more_tab_y]

                 # if we're not showing ratios, just plot datapoint
                 # otherewise need to do some scaling

                if  (scalefitsby is not None) or (centerfitsat is not None) or (addtofits is not None): # scale fits info by something
                    if scalefitsby is not None:
                        if np.isscalar(scalefitsby):
                            scalefitsby = scalefitsby*np.ones_like(xvals)
                        scaley_interp = interp1d(xvals[axmask],scalefitsby[axmask],bounds_error=False,fill_value='extrapolate')
                        compscaley = scaley_interp(plot_tab_x)
                    else:
                        compscaley = 1.

                    plot_tab_yerr = plot_tab_yerr*compscaley

                    if centerfitsat is not None:
                        plot_tab_y = centerfitsat*np.ones_like(plot_tab_y)
                        # doesn't really make sense to plot multple fits here
                    else:
                        plot_tab_y = plot_tab_y*compscaley +addtofits

                        if morefits:
                            plot_more_tab_y = [y*compscaley+addtofits for y in plot_more_tab_y]

                if  doratio: #plotting ratios,  we need to take the ratios here
                    if not justfits: # need to interpolate fits stuff to compare to theory
                        # need fidy at x values of fits for differncing
                        fidy_interp = interp1d(xvals[axmask],yfid[axmask],bounds_error=False,fill_value='extrapolate')
                        compfidy= fidy_interp(plot_tab_x)
                        if doratio_errordenom: # interpolate errorbars
                            yerr_tab_interp = interp1d(plot_tab_x,plot_tab_yerr,bounds_error=False,fill_value='extrapolate')
                            # scale the theory y as needed
                            compyerr= yerr_tab_interp(xvals[axmask])

                            yerrgrid = np.tile(compyerr,(ploty.shape[1],1)).T

                            ploty[axmask,:] = ploty[axmask,:]/yerrgrid

                            if centerfitsat is not None:
                                plot_tab_y = centerfitsat*np.ones_like(plot_tab_y)
                            else:
                                plot_tab_y = (plot_tab_y-compfidy)/plot_tab_yerr
                            if morefits:
                                plot_more_tab_y = [(y-compfidy)/plot_tab_yerr for y in plot_more_tab_y]

                            plot_tab_yerr = np.ones_like(plot_tab_yerr)
                        elif doratio_nodiff: #plotting ratio wrt fid theory
                            if centerfitsat is not None:
                                plot_tab_y = centerfitsat*np.ones_like(plot_tab_y)
                            else:
                                plot_tab_y = plot_tab_y/compfidy
                            plot_tab_yerr = plot_tab_yerr/np.fabs(compfidy)
                            if morefits:
                                plot_more_tab_y = [y/compfidy for y in plot_more_tab_y]
                        else: # plotting ratio of diff from fid wrt fid theory
                            if centerfitsat is not None:
                                plot_tab_y = centerfitsat*np.ones_like(plot_tab_y)
                            else:
                                plot_tab_y = (plot_tab_y-compfidy)/compfidy 
                            plot_tab_yerr = plot_tab_yerr/np.fabs(compfidy)
                            if morefits:
                                plot_more_tab_y = [y/compfidy-1. for y in plot_more_tab_y]
                    else: #if just fits, ratio will be wrt first fits file
                        fidtab_y = plot_tab_y
                        tab_yerr = plot_tab_yerr
                        if doratio_errordenom: #ratio is yerr
                            plot_tab_y = (fidtab_y - fidtab_y)/tab_yerr
                            plot_more_tab_y = [(y-fidtab_y)/tab_yerr for y in plot_more_tab_y]
                            plot_tab_yerr = np.ones_like(plot_tab_yerr)
                        elif doratio_nodiff:
                            plot_tab_y = fidtab_y/fidtab_y 
                            plot_tab_yerr = plot_tab_yerr/np.fabs(fidtab_y)
                            plot_more_tab_y = [y/fidtab_y  for y in plot_more_tab_y]
                        else: # denom is fid y
                            plot_tab_y = fidtab_y/fidtab_y -1.
                            plot_tab_yerr = plot_tab_yerr/np.fabs(fidtab_y)
                            plot_more_tab_y = [y/fidtab_y -1. for y in plot_more_tab_y]


            # plot lines
            if not justfits: #plot theorylines
                # plot lines for appropriate y values
                if showfid:
                    addlines(ax,xvals[axmask],ploty[axmask,:],dolab,lslist,collist,labels,fidind,fidlabel,lwlist=lwlist)
                else:
                    addlines(ax,xvals[axmask],ploty[axmask,:],dolab,lslist,collist,labels,None,lwlist=lwlist)

            binstr="{0:d}{1:d}".format(b1,b2)

            if fitsfile and not hidefits: # plot the fits file lines
                if not doratio_errordenom:
                    # not sure why, but errorbar plotting alters ranges in
                    # weird ways, using some workarounds
                    if justfits:
                        # this is clunky, but this seems to be needed to
                        # get plot range to behave
                        ax.plot(plot_tab_x,plot_tab_y,color='k',label=fitslabel,marker='.',markersize=0,lw=0,ls='none',zorder=100)

                    xmin,xmax = ax.get_xlim()
                    ax.errorbar(plot_tab_x,plot_tab_y,yerr=plot_tab_yerr,color='k',label=fitslabel,capsize=1,marker='.',markersize=marksize,lw=1,ls='none',zorder=100)
                    ax.set_xlim((xmin,xmax))
                else:
                    ax.scatter(plot_tab_x,plot_tab_y,color='k',label=fitslabel,marker='.',s=marksize,lw=1,ls='-',zorder=100)

                if morefits:
                    for morei,y in enumerate(plot_more_tab_y):
                        if not morefitscols:
                            mcol = 'gray'
                        else:
                            mcol = morefitscols[morei]
                        if morefitslabels:
                            mlab = morefitslabels[morei]
                        else:
                            mlab =''
                        if morefitserrors:
                            if not justfits:
                                xmin,xmax = ax.get_xlim()
                            # this is still using errorbars from initial fits file
                            # should eventually change to get them from
                            # each fits file individually
                            ax.errorbar(plot_tab_x,y,plot_tab_yerr,color=mcol,label=mlab,capsize=1,marker='.',markersize=marksize,lw=1,ls='none',zorder=101)
                            if not justfits:
                                ax.set_xlim((xmin,xmax))
                        else:
                            if doratio_errordenom:
                                ax.scatter(plot_tab_x,y,color=mcol,label=mlab,marker='.',s=marksize,lw=1,ls='-',zorder=101)
                            else:
                                ax.errorbar(plot_tab_x,y,color=mcol,label=mlab,capsize=1,marker='.',markersize=marksize,lw=1,ls='none',zorder=101)

            ax.annotate(binstr,(.06,.94),xycoords = 'axes fraction',\
                 color='k', ha='left', va='top',fontsize=8,\
                     bbox={'facecolor':'white','lw':0,'alpha':0.8,'pad':2})

            if scutgrid is not None: # add shaded regions for scale cuts

                #print('b1,b2',b1,b2)
                #print(scutgrid[:,0],type(scutgrid[:,0]),scutgrid[:,0]==b2)
                search = (scutgrid[:,0]==b1)*(scutgrid[:,1]==b2)
                if sym and not np.sum(search): #try the flipped order
                    search = (scutgrid[:,0]==b2)*(scutgrid[:,1]==b1)
                if np.sum(search):
                    cutmin,cutmax= scutgrid[np.where(search)[0][0],2:]
                    #print(plottype,b1,b2,cutmin,cutmax)
                    #ax.axvline(cutmin,ls=':',color='k')
                    #ax.axvline(cutmax,ls='--',color='k')
                    xmin,xmax = ax.get_xlim()

                    if cutmin>xmin:
                        ax.axvspan(xmin,cutmin,facecolor='lightgray',alpha=.5)
                    if cutmax<xmax:
                        ax.axvspan(cutmax,xmax,facecolor='lightgray',alpha=.5)
                    ax.set_xlim((xmin,xmax))
                    #print("    ",ax.get_xlim())




    # after all subplots are made, check in ylims
    if doratio:
        ymin,ymax = ax.get_ylim()

        if ymin<0:
            ymin = max(ymin,-1*maxdiff)
        if ymax>0:
            ymax = min(ymax,maxdiff)
        ax.set_ylim((ymin,ymax))
    if ylim_input is not None:
        # can use ylim_input to show narrower range than what
        # would be default based on lines
        ymin_in, ymax_in = ylim_input
        ymin,ymax = ax.get_ylim()
        ymin_out = max(ymin,ymin_in)
        ymax_out = min(ymax,ymax_in)
        ax.set_ylim((ymin_out,ymax_out))
                
    # show y label once along left side of figure
    fig.text(0.01,0.5,ylabel,va='center',rotation='vertical')
    
            
    #. . . . . . . . . . . . . . . . . . . .
    # tight layout and color bar if needed
    #. . . . . . . . . . . . . . . . . . . .
    haveleg=False
    plt.tight_layout(h_pad=0,w_pad=0,rect=[0.02,0,1,1])
    if (dolab or justfits):
        if not hidelegend:
            handles,labels = ax.get_legend_handles_labels()
            if labels:
                fig.legend(handles,labels,loc='upper right',title=legtitle)
                haveleg=True
    else:  #add in colorbar (need to place after tight layout)
        cbar=fig.colorbar(cmap,ax=axes,pad=0,aspect=6*Nrow,fraction=.1)
        cbar.set_label(collabel)
        if showfid:
            fidcol=colvals[fidind]
            cbar.ax.axhline(fidcol,ls='--',color='k',lw=1)

    if note: #add info note
        if justfits:
            notecol = 'k'
        else:
            notecol = 'darkgray'
        if haveleg: #put note to right of first column
            if autoonly:
                cornerax = axes[1]
            else:
                cornerax = axes[0,1]
            axpos = cornerax.get_position().get_points()#[[x0, y0], [x1, y1]]
            axheight = np.fabs(axpos[1,1] - axpos[0,1])
            axwidth = np.fabs(axpos[0,0] - axpos[1,0])
            notepos = (axpos[0,0]+0.1*axwidth, axpos[0,1] +.1*axheight)

            ax.annotate(note,notepos,xycoords = 'figure fraction',\
                        color=notecol, ha='left', va='bottom',fontsize=10)
        else: #put note in upper right of figure
            if autoonly:
                cornerax = axes[Ncol-1]
            else:
                cornerax = axes[0,Ncol-1]
            axpos = cornerax.get_position().get_points()#[[x0, y0], [x1, y1]]
            axheight = np.fabs(axpos[1,1] - axpos[0,1])
            axwidth = np.fabs(axpos[0,0] - axpos[1,0])
           

            if autoonly:
                notepos = (axpos[1,0]-.1*axwidth, axpos[1,1] + .1*axheight)
                ax.annotate(note,notepos,xycoords = 'figure fraction',\
                            color=notecol, ha='right', va='top',fontsize=8)
            else:
                notepos = (axpos[1,0]-.1*axwidth, axpos[1,1] + .1*axheight)
                ax.annotate(note,notepos,xycoords = 'figure fraction',\
                            color=notecol, ha='right', va='top',fontsize=8)

            
    if save:
        if doratio_nodiff:
            ratiotag = '-ratio'
        elif doratio_errordenom:
            ratiotag = '-diff-vssig'
        elif doratio:
            ratiotag = '-diff'
        else:
            ratiotag = ''

        outf = '.'.join([outftag,outbase+ratiotag,plottype,ftype])
        saveto = os.path.join(outdir,outf)
        print("Saving plot to",saveto)
        plt.savefig(saveto)
        plt.clf()
        plt.close(fig)


#------
def plot_nz_grid(xvals,binvals,ygrid,plottype='nz',sample='lens',\
                 labels=None,collist=None,lslist=None,\
              colvals=None,collabel=None,vmin=None,vmax=None,usecmap=None,\
              outftag='nztest',outdir='plots_datavecs',ftype='png',\
              justdiff=False,nodiff=False,typenote="",fidind=0,fidlabel='fid',\
              ymin=None,ymax=None,ydmin=None,ydmax = None, \
              xmin = None, xmax = None, dology = None, dologx=None,\
              save=True,legtitle='',hlines=[],vlines=[],lwlist=None):
    """
    
    Plot the variations in a set of for different parameter values.
    plottype indicates what kind of function is being plotted. 
      nz - redshift distributions for a given sample (one panel per z bin)

    sample - for nz plots, what is the sample label used in datablock and for plot labels
      
    # for n(z) plots, expects arrays with Nrows=nz*bins
    xvals - z values; will likely be same array of size nz repeated nbins times
    ygrid - Nrows x Nlines array of power spectra or whatever function we're dealing with

    There are two ways to label the lines: legend labels, or a color scale.

    legend label approach: used if labels is not None
        labels - list of string labels corresponding to each line plotted. 
        collist - optional, list of colors to use for the lines

    line color approach: color of line set by where numerical value falls on color scale
        colvals - Ncosm-entry array with associated number for each line
        collabel - string to use to label color scale
        vmin - bottom value for colorbar; default is min of colvals
        vmax - top value for colorbar; default is max of colvals
        usecmap - optional matplotlib.cm object specifying color map to use

    lslist = list of linestyles for each cosm. If None, will set all to solid.
    

    if justdiff: plots just show diff between fid and others calcs. 
                  otherwise shows both power spectrum and diff. 
    if nodiff: single panel plot just showing main funciton, no diff panel

    
    typenote - string to use if you want to include an annotation 
          about type of power spectrum (e.g. to distinguish linear vs non)

    fidind = index of cosm to use as diff denominator
    fidlabel = short label for fid cosm to use in axis label

    outftag - output filename will be outdir/plottype.outftag.ftype
    outdir - directory where to put the plot
    ftype  - file type of plot    
    """
    dolab = labels is not None
    if dology is None:
        dology = False
    if dologx is None:
        dologx  = False

    # find fid y, compute diffdiffs
    yfid = ygrid[:,fidind]
    yfidgrid = np.tile(yfid,(ygrid.shape[1],1)).T
    ydiff = ygrid-yfidgrid #-1
    whereinf = np.logical_not(np.isfinite(ydiff))
    ydiff[whereinf] = np.nan

    if not np.sum(np.isfinite(ygrid)):
        print("No finite y values, skipping plot for",plottype,outftag,typenote)

    # . . . . . . . . . . . . . . . . . . . . . .
    # Set up colors
    # . . . . . . . . . . . . . . . . . . . . . .
    if not dolab: #set up colormap
        # perceptually uniform sequential options: viridis, inferno, cividis
        # diverging: coolwarm, Spectral, RdBu
        if vmin is None:
            vmin = np.min(colvals)
        if vmax is None:
            vmax = np.max(colvals)
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        if usecmap is None:
            usecmap = matplotlib.cm.RdBu
        cmap = matplotlib.cm.ScalarMappable(norm=norm,cmap=usecmap)
        cmap.set_array([])
        collist = cmap.to_rgba(colvals)

    # . . . . . . . . . . . . . . . . . . . . . .
    # Set up axis labels and plot-type specific x ranges
    # . . . . . . . . . . . . . . . . . . . . . .
    outtypetag = plottype
    if plottype in ['nz']:
        outtypetag = 'nz'
        xlabel = r'$z$'
        #samplestr = '^{{\rm {}}}'.format(sample)
            
        #ydifflabel=r'$\Delta n{1:s}(z)/n{1:s}^{{\rm {0:s}}}(z)$'.format(fidlabel,samplestr)
        

    # . . . . . . . . . . . . . . . . . . . . . .
    # sort out plot bounds
    # . . . . . . . . . . . . . . . . . . . . . .

    if xmax is None:
        xmax=np.nanmax(xvals)
    if xmin is None:
        xmin=np.nanmin(xvals)

    usexinds = (xvals>=xmin)*(xvals<=xmax)
    if not justdiff:
        if ymax is None:
            if dology:
                ymax = 2*np.nanmax(ygrid[usexinds,:])
            else:
                yrange = np.nanmax(ygrid[usexinds,:]) - np.nanmin(ygrid[usexinds,:])
                ymax = np.nanmax(ygrid[usexinds,:]) + 0.02*yrange
        if ymin is None:
            if dology:
                ymin = 0.5*np.nanmin(ygrid[usexinds,:])
            else:
                yrange = np.nanmax(ygrid[usexinds,:]) - np.nanmin(ygrid[usexinds,:])
                ymin = np.nanmin(ygrid[usexinds,:]) - 0.02*yrange
    #print('ymaxmin',ymax,ymin)
    
    if ydmax is None:
        ydmax = np.nanmax(ydiff[usexinds,:])
    else:
        ydmax = min(ydmax,np.nanmax(ydiff[usexinds,:]))
    if ydmax>0:
        ydmax = max(1.1*ydmax,0.1)
    else:
        ydmax = .8*ydmax
    if ydmin is None:
        ydmin = np.nanmin(ydiff[usexinds,:])
    else:
        ydmin = max(ydmin,np.nanmin(ydiff[usexinds,:]))
        
    if ydmin>0:
        ydmin = .8*ydmin
    else:
        ydmin = min(1.2*ydmin,-0.1)

    #. . . . . . . . . . . . . . . . . . . .
    # Prep annotation with whatever we want written on the plot
    # . . . . . . . . . . . . . . . . . . . . . .
    
    if typenote:
        infonote = typenote
    else:
        infonote = ""

    #. . . . . . . . . . . . . . . . . . . .
    # Make the plot
    # . . . . . . . . . . . . . . . . . . . . . .
    nbin = np.max(binvals) # binval numbering starts at 1, so max is # of bins

    # make subplots, vertically stacked, one row per bin.
    # if we want to see diffs, make 2nd column
    if justdiff or nodiff: #one columnplot setup
        fig,axes = plt.subplots(nbin,1,figsize=(5,1+nbin),sharex=True,sharey=True)
        #Ncol =1
    else: # 2 column plot
        fig,axes = plt.subplots(nbin,2,sharex=True,sharey='col',figsize=(8,1+nbin))
        #Ncol=2
    plt.subplots_adjust(hspace=0)#,wspace=0)
        
    # set axis ranges
    if nodiff:
        ax = axes[0]
    elif not justdiff:
        ax = axes[0,0]
        
    if not justdiff: # set up n(z) nondiff plots
        for n in range(nbin):
            if nodiff:
                ax = axes[n]
            else:
                ax = axes[n,0]
                
            if dologx:
                ax.set_xscale("log", nonpositive='clip')
            if dology:
                ax.set_yscale("log", nonpositive='clip')
            ax.set_ylim((ymin,ymax))
            ax.set_xlim((xmin,xmax))
            if n==nbin-1: # set xlabel for the bottom row
                ax.set_xlabel(xlabel)

            ylabel = r"$n^{{\rm {}}}_{}(z)$".format(sample,n+1)
            ax.set_ylabel(ylabel)
            
            ax.tick_params(axis='x',which='both',bottom=True,top=True,direction="in")
            for hl in hlines:
                ax.axhline(hl,ls='-',color='darkgray',lw=0.5)
            for vl in vlines:
                ax.axvline(vl,ls='-',color='darkgray',lw=0.5)


            binmask = binvals==(n+1)
            addlines(ax,xvals[binmask],ygrid[binmask,:],dolab,lslist,collist,labels,fidind,fidlabel,lwlist=lwlist)

            if n==0:
                ax.legend(title=legtitle)
            if n==1:
                if infonote:
                    ax.annotate(infonote,(.95,.95),xycoords = 'axes fraction',\
                                color='gray', ha='right', va='top',fontsize=8)

    if not nodiff: # plot the diffs
        for n in range(nbin):
            if justdiff:
                ax = axes[n]
            else:
                ax = axes[n,1]

            ax.set_ylim((ydmin,ydmax))
            ydifflabel = r"$[n - n_{{ \rm {2:s}}}]^{{\rm {0:s}}}_{1:d}$".format(sample,n+1,fidlabel)

            ax.set_ylabel(ydifflabel)
            if n==nbin-1: # only add xlable for bottom row
                ax.set_xlabel(xlabel)
            binmask = binvals==(n+1)
            addlines(ax,xvals[binmask],ydiff[binmask,:],dolab,lslist,collist,labels,fidind,fidlabel,lwlist=lwlist)

            if justdiff and n==0: # need plot labels
                ax.legend(title=legtitle)
            if justdiff and n==1:
                if infonote:
                    ax.annotate(infonote,(.95,.95),xycoords = 'axes fraction',\
                                color='gray', ha='right', va='top',fontsize=8)

    #. . . . . . . . . . . . . . . . . . . .
    # tight layout and color bar if needed
    #. . . . . . . . . . . . . . . . . . . .
    plt.subplots_adjust(hspace=0)#,wspace=0)
    plt.tight_layout(h_pad=0,w_pad=0.5,rect=[0.02,0,1,1])
    if dolab:
        #plte.tight_layout(h_pad=0,w_pad=0)
        pass
    else:  #add in colorbar (need to place after tight layout)
        #plt.tight_layout(h_pad=0,w_pad=0.0,rect=[0,0,.97,1])
        fidcol=colvals[fidind]
        cbar=fig.colorbar(cmap,label=collabel,ax=axes,pad=0,aspect=6*nbin,fraction=0.1)
        cbar.ax.axhline(fidcol,ls='--',color='k',lw=1)

    if save:
        if nodiff:
            difftag='-nodiff'
        elif justdiff:
            difftag='-justdiff'
        else:
            difftag=''
        outf = '.'.join([outftag,outtypetag+difftag,'nz_'+sample,ftype])
        saveto = os.path.join(outdir,outf)
        print("Saving plot to",saveto)
        plt.savefig(saveto)
        plt.clf()
    plt.close(fig)
                                


#==================================================================
def make_onepanel_sets(outfbase,bashfile=None,colvals=None,fidval=None,collabel="",\
                       gendata=False, dopk=True, dogro=True,\
                       do2pt=True,do2pt_diffratio =True, do2pt_diffvssig=True,
                       dodist=True, docmb=True,dolsscl=True, donz=True,\
                       callfrom=Y6METHODSDIR,datdir='output_dat',\
                       plotdir = 'plots',infostr="",plist=None,\
                       twoptlist=None, pklist=None,cmblist = None,\
                       cllist=None,nzlist=None,nz_zmin=None,nz_zmax=None,nzdiffmax=1.5,\
                        planck_module=None,\
                       fitsfile='',fitslabel='',morefits=[],morefitscols=[],morefitslabels=[],max2ptdiffvssig=2.,\
                       inistr='${INI_PATH}',campaign_yaml=None,campaign_run=None\
):
    """
    Generate batch of Pk, sigma8z, fsigma8 plots for one set of 
    generated DVs. 


    outfbase: start of output filenames
    bashfile - path to sh job script
    colvalues - values  for each line to associate with line colors
    collabel - string to label color scales
    gendata - bool, if true runs cosmosis to make data, if false just does lots
    dopk, dogor, do2pt, donz, etc  - generate/plot different components of DV
    callfrom - path where cosmosis should be called from
    datdir - where to put output data from calculations, read in for plots
    plotdir - where to put plots
    plist  - list of dictionaries containing different parameter
             choices; one dictionary per line to plot
    twoptlist, pklist, cmblist - list of strings specifying which
           sections of each data type to compute/plot
    nzlist - list of sample lables for which we want to plot (nz)
    planck_module - planck likelihood module name
    fitsfile - name of fits file for which we want to plot datapoints
    fitslabel - label for fits file datapoints
    morefits - list of other fits files to make other datapoints for
    morefitscols - colors for the additional datapoints
    morefitslabels - labels for additional datapoints
    inistr - path to params ini file. will be sorted automatically
             given run_sh files, but leaving this option here
             in case you want to use a file in a different setup
    max2ptdiffvssig - if you're plotting 2pt difference vs errorbar, can set 
             max plot range with this
    campaign_yaml - path to cosmosis campaign yaml file to use
    campaign_run - run name for pipeline you want to run in that campaign file
    """
    if twoptlist is None:
        twoptlist = default_twopt_sections
    if pklist is None:
        pklist = default_pk_sections
    if cmblist is None:
        cmblist = default_cmb_sections
    if cllist is None:
        cllist = default_lss_cl_sections
    if nzlist is None:
        nzlist = default_nz_samples

    if gendata:
        if bashfile is not None:
            get_bashvars_fromshfile(bashfile)
        inifile_params = os.path.expandvars(inistr)

        plabels=['{0:g}'.format(val) for val in colvals]
        gen_dataset_list(plist,plabels,outfbase,datdir,inifile_params,callfrom,\
                             do2pt+do2pt_diffratio+do2pt_diffvssig, dopk, dogro, dodist, docmb, donz, twopt_sections=twoptlist, pk_sections = pklist, planck_module=planck_module,do_lss_cl = dolsscl, lss_cl_sections = cllist, nz_samples = nzlist, campaign_yaml=campaign_yaml,campaign_run=campaign_run)

    # print('dopk',dopk)
    # print('dogro',dogro)
    # print('do2pt',do2pt)
    # print('dodist',dodist)
    # print('docmb',docmb)
    # print('dolsscl',dolsscl)

    fidind = np.where(colvals==fidval)[0][0]

    if dopk: #pk  plots 
        #read in data
        #NL power spec
        for ptype in pklist:
            pkfile = get_datfile_pk(outfbase,0,ptype,datdir)
            pkdat = np.loadtxt(pkfile)
            plot_func(pkdat[:,0],pkdat[:,1:],plottype=ptype,\
                          colvals=colvals,collabel=collabel,\
                          outftag=outfbase,outdir=plotdir,fidind=fidind,\
                          typenote=infostr)


    if dogro: #growth plots
        fsigfile = get_datfile_grow(outfbase,'fsigma_8',datdir)
        sigfile = get_datfile_grow(outfbase,'sigma_8',datdir)
        
        fsigdat = np.loadtxt(fsigfile)
        plot_func(fsigdat[:,0],fsigdat[:,1:],plottype='fsig8',\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                          typenote=infostr)
        
        sigdat = np.loadtxt(sigfile)
        plot_func(sigdat[:,0],sigdat[:,1:],plottype='sig8',\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                      typenote=infostr)


    if do2pt or do2pt_diffratio or do2pt_diffvssig:
        
        for plottype in twoptlist:
            fname = get_datfile_2pt(outfbase,plottype,datdir)
            dat = np.loadtxt(fname)
            theta = dat[:,0]
            bin1 = dat[:,1].astype('int')
            bin2 = dat[:,2].astype('int')
            twopt = dat[:,3:]

            #print('>> twopt.shape',twopt.shape)
            scutf =  get_datfile_2pt(outfbase,plottype+"_scuts",datdir)
            if os.path.isfile(scutf):
                scutdat = np.loadtxt(scutf)
                #print('scutdat.shape',scutdat.shape)
            else:
                scutdat = None
                
            # non-ratio
            if do2pt:
                plot_2pt_grid(theta,bin1,bin2,twopt,plottype,doratio=False,\
                              colvals=colvals,collabel=collabel,\
                              outftag=outfbase,outdir=plotdir,fidind=fidind,\
                              note=infostr, scutgrid=scutdat,\
                              fitsfile=fitsfile,fitslabel=fitslabel,\
                              morefits=morefits,morefitscols=morefitscols,\
                              morefitslabels=morefitslabels,\
                )
            
            # diff ratio
            if do2pt_diffratio:
                plot_2pt_grid(theta,bin1,bin2,twopt,plottype,doratio=True,\
                              colvals=colvals,collabel=collabel,\
                              outftag=outfbase,outdir=plotdir,fidind=fidind,\
                              note=infostr, scutgrid=scutdat,\
                              fitsfile=fitsfile,fitslabel=fitslabel,\
                              morefits=morefits,morefitscols=morefitscols,\
                              morefitslabels=morefitslabels)

            # diff over cov entries (if we have fits file)
            if fitsfile and do2pt_diffvssig:
                #if max2ptdiffvssig>1:
                #    hlines = [-1,0,1]
                #else:
                hlines=[0]
                    
                plot_2pt_grid(theta,bin1,bin2,twopt,plottype,doratio=False,\
                              colvals=colvals,collabel=collabel,\
                              outftag=outfbase,outdir=plotdir,fidind=fidind,\
                              note=infostr, scutgrid=scutdat,\
                              fitsfile=fitsfile,fitslabel=fitslabel,\
                              morefits=morefits,morefitscols=morefitscols,\
                              morefitslabels=morefitslabels,\
                              centerfitsat=0.,hlines=hlines,maxdiff=max2ptdiffvssig,\
                              doratio_errordenom=True,hidefits=True)

    if dodist:
        dafile = get_datfile_dist(outfbase,'d_a',datdir)
        hfile = get_datfile_dist(outfbase,'h',datdir)
        
        dadat = np.loadtxt(dafile)
        plot_func(dadat[:,0],dadat[:,2:],plottype='d_a',\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                          typenote=infostr)
        
        hdat = np.loadtxt(hfile)
        plot_func(hdat[:,0],hdat[:,2:],plottype='h',\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                      typenote=infostr)

    if docmb:
        
        for plottype in cmblist:
            clfile = get_datfile_cmbcl(outfbase,plottype,datdir)
            cldat = np.loadtxt(clfile)
            plot_func(cldat[:,0],cldat[:,1:],plottype='cmb_cl_'+plottype,\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                          typenote=infostr)

    
    if dolsscl:
        for plottype in cllist:
            clfile = get_datfile_lss_cl(outfbase,plottype,datdir)
            dat = np.loadtxt(clfile)
            ell = dat[:,0]
            bin1 = dat[:,1].astype('int')
            bin2 = dat[:,2].astype('int')
            cl = dat[:,3:]
            # non-ratio
            plot_2pt_grid(ell,bin1,bin2,cl,plottype,doratio=False,\
                      colvals=colvals,collabel=collabel,\
                      outftag=outfbase,outdir=plotdir,fidind=fidind,\
                              note=infostr)
            # diff ratio
            plot_2pt_grid(ell,bin1,bin2,cl,plottype,doratio=True,\
                              colvals=colvals,collabel=collabel,\
                              outftag=outfbase,outdir=plotdir,fidind=fidind,\
                              note=infostr)

    if donz:
        for sample in nzlist:
            nzfile = get_datfile_nz(outfbase,sample,datdir)
            dat = np.loadtxt(nzfile)
            bins = dat[:,0].astype('int')
            zvals = dat[:,1]
            nz = dat[:,2:]

            # # no ratio (maybe skip this but leave in for testing)
            # plot_nz_grid(zvals,bins,nz,plottype='nz',sample=sample,\
            #              nodiff=True, colvals = colvals, collabel = collabel,\
            #              outftag = outfbase, outdir = plotdir, fidind = fidind,\
            #              typenote=infostr, xmin=nz_zmin,xmax=nz_zmax)

            # with ratio
            plot_nz_grid(zvals,bins,nz,plottype='nz',sample=sample,\
                         nodiff=False, colvals = colvals, collabel = collabel,\
                         outftag = outfbase, outdir = plotdir, fidind = fidind,\
                         typenote=infostr, xmin=nz_zmin,xmax=nz_zmax,\
                         ydmax=nzdiffmax,ydmin=-1*nzdiffmax)
    

#==================================================================
def main():
    """
    Putting some code here to test that script is working, give an example for how to use it.
    Should generally import this script into other files to use when doing 
    bigger batches of test plots
    """
    gendata=10
    
    Y6METHODSDIR = os.environ['Y6METHODSDIR']
    callfrom = Y6METHODSDIR
    outdir='output_dat'
    fitsfile= os.path.join(Y6METHODSDIR,'data_vectors/sim_d-gen2pt_simy6_l-desy3-hm20-tatt_ml4_sy3-covG.fits')
    fitslabel='simy6-atdy3-covG'

    if 0: #example varying T_AGN for baryon feedback
        # need to set some bash variables to use ini file
        bashvar_file = os.path.join(callfrom,'run_sh/d_l/d_l-hm20f-tatt-bwide_simfid.polychord-test.sh')
        get_bashvars_fromshfile(bashvar_file)
        bashbase = os.path.split(bashvar_file)[-1]

        inistring = '${INI_PATH}'
        inifile_params = os.path.expandvars(inistring)
        inifile_values = inifile_params.replace('params','values')
        
        rangedict, fiddict = get_values_from_ini(inifile_values)
        varstr = 'halo_model_parameters--logt_agn'
        vartex = r'$\log T_{\rm AGN}$'
        pdictlist,pvals,pfid = get_pdictlist_vary1param(varstr,Nvals=6,rangedict=rangedict,fidvaldict=fiddict)
    
        fbase = 'plottest'
        fitsf = ''#,Y6METHODSDIR+'/data_vectors/y3-2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'
        make_onepanel_sets(fbase,bashvar_file,pvals,pfid,vartex, \
                           gendata=gendata, dopk=1, dogro=1, do2pt=1,\
                           dodist=1,docmb=0,dolsscl=1,\
                           plist=pdictlist,fitsfile=fitsf,fitslabel=fitslabel,\
                           infostr=bashbase)

    if 1: # example varying S8 with NL bias
        # need to set some bash variables to use ini file
        bashvar_file = os.path.join(callfrom,'run_sh/d_l/d_l-s8-hm20-tatt-nlbias_simy6-atdy3-covG.polychord-test.sh')
        get_bashvars_fromshfile(bashvar_file)
        bashbase = os.path.split(bashvar_file)[-1]

        inistring = '${INI_PATH}'
        inifile_params = os.path.expandvars(inistring)
        inifile_values = inifile_params.replace('params','values')
        
        rangedict, fiddict = get_values_from_ini(inifile_values)
        print(rangedict.keys())
        varstr = 'cosmological_parameters--s_8_input'
        vartex = r'$S_8$'
        pdictlist,pvals,pfid = get_pdictlist_vary1param(varstr,Nvals=2,rangedict=rangedict,fidvaldict=fiddict)
    
        fbase = 'plottest-s8-nlbias'
        fitsf = ''#,Y6METHODSDIR+'/data_vectors/y3-2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'
        make_onepanel_sets(fbase,bashvar_file,pvals,pfid,vartex, \
                           gendata=gendata, dopk=1, dogro=1, do2pt=1,\
                           dodist=1,docmb=0,dolsscl=1,\
                           plist=pdictlist,fitsfile=fitsf,fitslabel=fitslabel,\
                           infostr=bashbase)

    
                       
    
#==================================================================
if __name__=="__main__":
    main()
