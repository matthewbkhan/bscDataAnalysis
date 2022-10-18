#!/usr/bin/env python

import scipy.interpolate as interpolate
from scipy import integrate
import numpy as np

""" Element density dictionary """
densities = {'Ag':10.5,'Al':2.2,'Au':19.3,'C':2.27,'Cu':8.96,'Fe':4.54,'Mn':7.21,'Mo':10.22,'Mylar':1.38,
             'Pb':11.35,'Polystyrene':1.06,'Sn':7.31,'Ta':16.65,'Teflon':2.2,'Ti':4.506,'W':19.25,'Zn':7.14}

def no_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind):
    mu_rho = mu_rho**-1.#inverting the data to help the interp
    interp_func = interpolate.interp1d(photon_MeV, mu_rho, kind='cubic')
    interp = interp_func(E)
    mu_rho = mu_rho**-1.
    interp = interp**-1.
    return interp

def one_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind):
    mu_rho = mu_rho**-1.#inverting the data to help the interp
    edge_1 = interp_edge[1]
    booles_1 = (E<np.max(photon_MeV[0:edge_1]))
    inds_1   = np.where(booles_1)
    E_1      = E[inds_1]
    interp_func_1 = interpolate.interp1d(photon_MeV[0:edge_1], mu_rho[0:edge_1], kind=interp_kind[1])
    interp_1 = interp_func_1(E_1)

    booles_2 = (E>np.min(photon_MeV[edge_1:]))
    inds_2   = np.where(booles_2)
    E_2      = E[inds_2]

    interp_func_2 = interpolate.interp1d(photon_MeV[edge_1:], mu_rho[edge_1:], kind='cubic')
    interp_2 = interp_func_2(E_2)

    interp = np.concatenate((interp_1,interp_2))

    mu_rho = mu_rho**-1.
    interp = interp**-1.

    return interp

def two_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind):
    mu_rho = mu_rho**-1.#inverting the data to help the interp
    edge_1 = interp_edge[1]
    booles_1 = (E<np.max(photon_MeV[0:edge_1]))
    inds_1   = np.where(booles_1)
    E_1      = E[inds_1]
    interp_func_1 = interpolate.interp1d(photon_MeV[0:edge_1], mu_rho[0:edge_1], kind=interp_kind[1])
    interp_1 = interp_func_1(E_1)

    edge_2 = interp_edge[2]
    booles_2 = np.logical_and(E>=np.max(photon_MeV[0:edge_1]), E<np.min(photon_MeV[edge_2:]))
    inds_2   = np.where(booles_2)
    E_2      = E[inds_2]
    interp_func_2 = interpolate.interp1d(photon_MeV[edge_1:edge_2], mu_rho[edge_1:edge_2], kind=interp_kind[2])
    interp_2 = interp_func_2(E_2)

    booles_3 = (E>np.min(photon_MeV[edge_2:]))
    inds_3   = np.where(booles_3)
    E_3      = E[inds_3]

    interp_func_3 = interpolate.interp1d(photon_MeV[edge_2:], mu_rho[edge_2:], kind='cubic')
    interp_3 = interp_func_3(E_3)

    interp = np.concatenate((interp_1,interp_2))
    interp = np.concatenate((interp,interp_3))

    mu_rho = mu_rho**-1.
    interp = interp**-1.

    return interp

def four_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind):
    mu_rho = mu_rho**-1.#inverting the data to help the interp
    edge_1 = interp_edge[1]
    booles_1 = (E<np.max(photon_MeV[0:edge_1]))
    inds_1   = np.where(booles_1)
    E_1      = E[inds_1]
    interp_func_1 = interpolate.interp1d(photon_MeV[0:edge_1], mu_rho[0:edge_1], kind=interp_kind[1])
    interp_1 = interp_func_1(E_1)

    edge_2 = interp_edge[2]
    booles_2 = np.logical_and(E>=np.max(photon_MeV[0:edge_1]), E<np.min(photon_MeV[edge_2:]))
    inds_2   = np.where(booles_2)
    E_2      = E[inds_2]
    interp_func_2 = interpolate.interp1d(photon_MeV[edge_1:edge_2], mu_rho[edge_1:edge_2], kind=interp_kind[2])
    interp_2 = interp_func_2(E_2)

    edge_3 = interp_edge[3]
    booles_3 = np.logical_and(E>=np.max(photon_MeV[edge_1:edge_2]), E<np.min(photon_MeV[edge_3:]))
    inds_3   = np.where(booles_3)
    E_3      = E[inds_3]
    interp_func_3 = interpolate.interp1d(photon_MeV[edge_2:edge_3], mu_rho[edge_2:edge_3], kind=interp_kind[3])
    interp_3 = interp_func_3(E_3)

    edge_4 = interp_edge[4]
    booles_4 = np.logical_and(E>=np.max(photon_MeV[edge_2:edge_3]), E<np.min(photon_MeV[edge_4:]))
    inds_4   = np.where(booles_4)
    E_4      = E[inds_4]
    interp_func_4 = interpolate.interp1d(photon_MeV[edge_3:edge_4], mu_rho[edge_3:edge_4], kind=interp_kind[4])
    interp_4 = interp_func_4(E_4)

    booles_5 = (E>np.min(photon_MeV[edge_4:]))
    inds_5   = np.where(booles_5)
    E_5      = E[inds_5]

    interp_func_5 = interpolate.interp1d(photon_MeV[edge_4:], mu_rho[edge_4:], kind='cubic')
    interp_5 = interp_func_5(E_5)

    interp = np.concatenate((interp_1,interp_2))
    interp = np.concatenate((interp,interp_3))
    interp = np.concatenate((interp,interp_4))
    interp = np.concatenate((interp,interp_5))

    mu_rho = mu_rho**-1.
    interp = interp**-1.

    return interp

# Attenuation = 1- absorption
def attenuation(E,element,density,thickness): # E in MeV, element as string, density in g/cc, thickness in mm

    attenuation_data_path = './Attenuation_Data/'
    thickness            *= 0.1 # convert mm to cm
    mass_thickness        = thickness * density # mass tickness in g/cm2
    attenData             = np.loadtxt(attenuation_data_path+element+'.dat',comments="#") # Load in attenuation data
    photon_MeV, mu_rho    = attenData[:,0], attenData[:,1]

    filename = element+'_interp.dat'
    file = open(attenuation_data_path+filename, 'r')
    interp_edge = []
    interp_kind = []
    for line in file:
        p = line.split()
        interp_edge.append(int(p[0]))
        interp_kind.append(p[1])
    file.close()

    if(interp_edge[0]==0):
        interp = no_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
    elif(interp_edge[0]==1):
        interp = one_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
    elif(interp_edge[0]==2):
        interp = two_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
    elif(interp_edge[0]==4):
        interp = four_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)

    attenuation_factor = np.exp(-(interp*mass_thickness))

    return attenuation_factor

########################
# RAW ABSORPTION MODEL #
########################
def layer_attenuation(E,layer_thickness,layer_density,element_names,ratios,atomic_weights): # layer_thickness in microns, density in g/cc
    attenuation_data_path = './Attenuation_Data/'
    total_weight = np.sum(np.array(ratios)*np.array(atomic_weights))
    layer_thickness *= 1e-4 # convert from microns to cm
    class element:
        global attenuation_data_path
        global total_weight
        global layer_density
        global mu_en_rho
        def __init__(self, name, ratio, atomic_weight):
            self.name = name
            self.ratio = ratio
            self.atomic_weight = atomic_weight
            attenData          = np.loadtxt(attenuation_data_path+name+'.dat',comments="#")
            photon_MeV, mu_rho = attenData[:,0], attenData[:,1]
            filename = name+'_interp.dat'
            file = open(attenuation_data_path+filename, 'r')
            interp_edge = []
            interp_kind = []
            for line in file:
                p = line.split()
                interp_edge.append(int(p[0]))
                interp_kind.append(p[1])
            file.close()
            if(interp_edge[0]==0):
                self.interp = no_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
            elif(interp_edge[0]==1):
                self.interp = one_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
            elif(interp_edge[0]==2):
                self.interp = two_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)
            elif(interp_edge[0]==4):
                self.interp = four_edge(E,photon_MeV,mu_rho,interp_edge,interp_kind)

            self.density = ratio * (atomic_weight/total_weight) * layer_density
            self.mass_thickness = layer_thickness * self.density
            self.attenuation_factor = np.exp(-(self.interp*self.mass_thickness))
    elements = list()
    for element_name,ratio,atomic_weight in zip(element_names,ratios,atomic_weights):
        elements.append(element(element_name,ratio,atomic_weight))
    attenuation_factor = 1.
    for element in elements:
        attenuation_factor *= element.attenuation_factor
    return attenuation_factor

#########
#FITTING#
#########
# Fitting in needed to match the raw absoprtion model to the experimental data
def fitting_function(m,E,b):
    return m*E + b

def scaling(E,type):
    #Type decides between Meadowcroft (True) and Maddox (False)
    E *= 1e6
    booles_E_I = (E<(5.98880e3))
    inds_E_I   = np.where(booles_E_I)
    E_I        = E[inds_E_I]
    booles_E_II = np.logical_and(E>=5.98880e3,E<13.4737e3)
    inds_E_II   = np.where(booles_E_II)
    E_II        = E[inds_E_II]
    booles_E_III = np.logical_and(E>=13.4737e3,E<37.4406e3)
    inds_E_III   = np.where(booles_E_III)
    E_III        = E[inds_E_III]
    booles_E_IV = (E>=(37.4406e3))
    inds_E_IV   = np.where(booles_E_IV)
    E_IV        = E[inds_E_IV]
    if(type):
        #Meadowcroft
        MR_Region_I_m = 1.07*1e-3
        MR_Region_I_b = -0.51
        MR_Region_II_m = 0.61*1e-3
        MR_Region_II_b = 1.92
        MR_Region_III_m = 0.67*1e-3
        MR_Region_III_b = 0.75
        MR_Region_IV_m = 0.67*1e-3
        MR_Region_IV_b = -9.10
        Sensitivity_MR_scaling = np.concatenate((fitting_function(MR_Region_I_m, E_I, MR_Region_I_b),fitting_function(MR_Region_II_m, E_II, MR_Region_II_b)))
        Sensitivity_MR_scaling = np.concatenate((Sensitivity_MR_scaling,fitting_function(MR_Region_III_m, E_III, MR_Region_III_b)))
        Sensitivity_MR_scaling = np.concatenate((Sensitivity_MR_scaling,fitting_function(MR_Region_IV_m, E_IV, MR_Region_IV_b)))
    else:
        #Maddox
        MR_Region_II_m = 7.098e-4
        MR_Region_II_b = -3.789
        MR_Region_III_m = 4.446e-4
        MR_Region_III_b = 0.3745
        MR_Region_IV_m = 5.658e-4
        MR_Region_IV_b = -12.05
        Sensitivity_MR_scaling = np.concatenate((fitting_function(MR_Region_II_m, E_II, MR_Region_II_b),fitting_function(MR_Region_III_m, E_III, MR_Region_III_b)))
        Sensitivity_MR_scaling = np.concatenate((Sensitivity_MR_scaling,fitting_function(MR_Region_IV_m, E_IV, MR_Region_IV_b)))
    mPSL_scaling = Sensitivity_MR_scaling
    PSL_scaling  = mPSL_scaling*1e-3
    E *= 1e-6
    return PSL_scaling

def sensitivity(E,phosphor_width,phosphor_density,phosphor_elements,phosphor_ratios,phosphor_atomic_weights,type=True):
    attenuation    = layer_attenuation(E,phosphor_width,phosphor_density,phosphor_elements,phosphor_ratios,phosphor_atomic_weights)
    absorption     = 1. - attenuation
    scaling_factor = scaling(E,type)
    sens           = absorption * scaling_factor
    return sens

#######################
# IP LAYER DEFINITION #
#######################
class layer:
    global atomic_weights_dict
    atomic_weights_dict = {'Bromine':79.9,'Fluorine':19.0,'Barium':137.3,'Iodine':126.9,'Mn':54.9,'Zn':65.4,'Fe':55.8,'C':12.0,'H':1.0,'O':16.0,'Mylar':1.,'Polystyrene':1.}
    def __init__(self, E, name, width, density, elements, ratios, sens_type=True):
        self.E = E # Energy range
        self.name = name # Layer name as string
        self.width = width # in microns
        self.density = density # in g/cc
        self.elements = elements # Element names ans strings
        self.ratios = ratios # Array of element ratios
        self.sens_type = sens_type # Bool for sensitivity model; True for Meadowcroft, False for Maddox
        self.atomic_weights = [atomic_weights_dict[element] for element in self.elements]
        self.attenuation = layer_attenuation(E,self.width,self.density,self.elements,self.ratios,self.atomic_weights)
        if(self.name=='Phosphor'):
            self.sensitivity = sensitivity(E,self.width,self.density,self.elements,self.ratios,self.atomic_weights,self.sens_type)

##########################################################################################

""" Load in image plate data and calculate its properties; sensitivity and attenuation """
def loadIPdata(hybrid):
    global layer
    if(hybrid):
        IPdata        = np.loadtxt("IPsensitivity_PSLperPhoton.txt")
        E_MeV         = IPdata[:,0]
        E_keV         = E_MeV*1e3
        IPsensitivity = IPdata[:,1]
    else:
        E_keV = np.arange(6,3001,1)
        E_MeV = E_keV * 1e-3
    surface_layer  = layer(E_MeV,'Surface', 9.,  1.66,['Mylar'],[1.])
    phosphor_layer = layer(E_MeV,'Phosphor',115.,3.31,['Bromine','Fluorine','Barium','Iodine'],[0.85,1,1,0.15])
    back_layer     = layer(E_MeV,'Back',    190.,1.66,['Mylar'],[1.])
    ferrite_layer  = layer(E_MeV,'Ferrite', 160.,2.77,['Zn','Mn','Fe','O','H','C'],[1.,2.,5.,40.,15.,10.])
    layers         = [surface_layer,phosphor_layer,back_layer,ferrite_layer]
    imagePlateAttenuation = 1.
    for ipLayer in layers:
        imagePlateAttenuation *= ipLayer.attenuation
    # for layer in layers:
    #     plt.plot(E_keV,np.ones(len(layer.attenuation))-layer.attenuation,label=layer.name)
    # plt.xlim([0,20])
    # plt.legend(loc="best")
    # plt.show()
    if not(hybrid):
        IPsensitivity = phosphor_layer.sensitivity
    return E_MeV, IPsensitivity, imagePlateAttenuation

""" Load in filter data and calculate the IP response curves (combination of filter atten and IP response) """
def loadFilterData(fileName,cannonType,geant4IP=False,preFilter=False):
    """ Load in image plate data and calculate its properties; sensitivity and attenuation """
    E_MeV, IPsensitivity, imagePlateAttenuation = loadIPdata(geant4IP) #low res produces manageable file sizes
    filters = np.loadtxt('./responseFunctions/'+fileName+'/'+fileName+'_FILTERS.txt',dtype=str,delimiter=',')
    filterMaterials, filterThicknesses = filters[:,0], [float(i) for i in filters[:,1]]
    numChannels = len(filterMaterials)
    numFilters  = np.arange(1,numChannels+1,1)
    if(cannonType=='singleChannel'):
        responseFunctions = []
        totalAttenuation = 1.
        for filMat, filThick, filt in zip(filterMaterials, filterThicknesses, numFilters):
            filterAttenuation = attenuation(E_MeV,filMat,densities[filMat],filThick)
            totalAttenuation *= filterAttenuation
            if(preFilter and filt==1):continue
            IPresponse = totalAttenuation*IPsensitivity
            totalAttenuation *= imagePlateAttenuation
            # totalAttenuation = 1.
            responseFunctions.append(IPresponse)
    elif(cannonType=='multiChannel'):
        responseFunctions = []
        for filMat, filThick in zip(filterMaterials, filterThicknesses):
            filterAttenuation = attenuation(E_MeV,filMat,densities[filMat],filThick)
            IPresponse = filterAttenuation*IPsensitivity
            responseFunctions.append(IPresponse)
    else:
        print('Cannon design not specified!')
    if(cannonType=='singleChannel' and preFilter):
        ch = np.arange(1,numChannels,1)
    else:
        ch = np.arange(1,numChannels+1,1)
    return filterMaterials, filterThicknesses, E_MeV, responseFunctions, ch

""" Plotting for the IP response curves """
def plotResponseFunctions(E_MeV,responseFunctions,filterMaterials,filterThicknesses,figNum,fileName,figName,log=True,save=False):
    mpl.rcParams['font.size']=24
    plt.figure(figNum)
    colours = cm.rainbow(np.linspace(0, 1, len(filterMaterials)))
    for resFunc,filMat,filThick,colour in zip(responseFunctions,filterMaterials,filterThicknesses,colours):
        if(normed):
            plt.plot(E_MeV*1e3,resFunc*(1./np.max(resFunc[E_MeV<0.5])),label='%s %.1f'%(filMat,filThick),color=colour)
        else:
            plt.plot(E_MeV*1e3,resFunc,label='%s %.1f'%(filMat,filThick),color=colour)
    plt.legend(loc='best',prop={"size":12},ncol=2)
    plt.xlabel('Photon Energy (keV)')
    plt.ylabel('IP Response (PSL/photon)')
    # plt.xlim([0,250])
    plt.xlim([1,500])
    # plt.ylim([1e-5*max(responseFunctions[0]),5e0*max(responseFunctions[0])])
    plt.tight_layout()
    if(log):
        plt.yscale('log')
        # plt.ylim([1e-10*max(responseFunctions[0]),5e0*max(responseFunctions[0])])
    # plt.tight_layout()
    if(save):
        plt.savefig('./responseFunctions/'+fileName+'/'+figName)
    return figNum+1

""" Saving the response functions """
def saveResFuncs(E_MeV,ch,fileName,txtName,save=False):
    resFuncOutput = np.zeros((len(E_MeV),len(ch)+1))
    for resFunc,chan in zip(responseFunctions,ch):
        resFuncOutput[:,chan] = resFunc
    resFuncOutput[:,0] = E_MeV
    if(save):
        np.savetxt('./responseFunctions/'+fileName+'/'+txtName,resFuncOutput,delimiter=',',fmt='%.5e')
    return

""" Calculate the IP responses """
def imagePlateResponses(E_MeV,A,T_MeV,responseFunctions,ch):
    IPsignals = []
    for resFunc in responseFunctions:
        photonNumSpec = funcPhotonNumSpec(E_MeV,A,T_MeV)
        measuredSpectrum = resFunc*photonNumSpec
        IPsignal = integrate.trapz(measuredSpectrum, E_MeV)
        IPsignals.append(IPsignal)
    if(plotting):
        mpl.rcParams['font.size']=24
        plt.figure(figNum)
        plt.plot(ch,IPsignals,marker='+',linestyle=' ',markersize=12, color='red',zorder=14)
        plt.yscale('log')
        plt.xlabel('Channel Number')
        plt.ylabel('Channel Signal (mPSL)')
        plt.xticks((ch))
        # plt.show()
    return np.array(IPsignals)

""" Find the closest value in an array """
def find_closest(data, v):
	return (np.abs(np.array(data)-float(v))).argmin()

""" ############# """
""" Gen Res Funcs """
""" ############# """
def generateResponseFunctions(fileName):
    hybrid   = False
    normed   = False
    plotting = False

    # fileName = 'TAW_v2'
    figName  = fileName+'.png'
    if(hybrid):
        txtName = fileName+'_hybrid.txt'
    else:
        txtName = fileName+'.txt'
    saving = False
    cannonType = 'singleChannel'
    # cannonType = 'multiChannel'

    """ Load in filter data and calculate the IP response curves (combination of filter atten and IP response """
    filterMaterials, filterThicknesses, E_MeV, responseFunctions, ch = loadFilterData(fileName,cannonType,geant4IP=hybrid,preFilter=True)
    if(plotting):
        """ Plot the response curves for the different filter thicknesses """
        figNum = 1
        figNum = plotResponseFunctions(E_MeV,responseFunctions,filterMaterials,filterThicknesses,figNum,fileName,figName,log=False,save=saving)
        """ Saving the response functions """
        # saveResFuncs(E_MeV,ch,fileName,txtName,save=saving)
        """ Show the graphs """
        plt.show()
    return E_MeV, responseFunctions
