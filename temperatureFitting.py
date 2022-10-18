#!/usr/bin/env python

import matplotlib.gridspec as gridspec
import responseFunctionBuilder as rFB
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import integrate
from matplotlib import cm
import matplotlib as mpl
import pandas as pd
import numpy as np
import sys

# Photon flux distribution function
def photonSourceDist(E_keV,T_keV):
    """
    Returns the thick-target model for the bremsstrahlung photon emission in photons/(keV.mm^2)
    Equation (10.1063/1.4890537) is typically used for single element material, Zeff here is the effective nuclear charge
    E_keV - photon energy range in keV
    T_keV - photon/hot electron temperature in keV
    """
    Zc, Zh, Zcl    = 12., 1., 17.
    Nc, Nh, Ncl    = 2.,  2., 2.
    Zeff           = (Zc*Nc + Zh*Nh + Zcl*Ncl)/(Nc+Nh+Ncl) # Effective nuclear charge
    detectorRadius = 220. # convert from cm to mm
    photonSource   = (1./E_keV)*np.exp(1.-E_keV/T_keV)*(5e11/(4.*np.pi))*(Zeff/79.)*(1./(detectorRadius*detectorRadius))
    return photonSource

# IP fading correction
def fadingFunction(t_min):
    """
    Natural fading of the exposed IPs, this function corrects for that (10.1063/1.4936141).
    Returns the fading factor, 0 < factor < 1
    t_min - time between exposure and scan in minutes
    """
    A1, A2 = 0.334,   0.666
    B1, B2 = 107.320, 33974.
    return A1*np.exp(-t_min/B1) + A2*np.exp(-t_min/B2)

# Load in shot data and convert to PSL/mm^2
def loadInShotData(shotNum,BSCnum):
    """
    Extract the relevant scaning info
    Correct for fading
    Naming and retrieving of data to be sorted later
    """
    if(shotNum<10):fileName = "./shotData/shot_0"+str(shotNum)+"/IP_Signals_shot0"+str(shotNum)+"_BSC"+str(BSCnum)+".csv"
    else:          fileName = "./shotData/shot_"+str(shotNum)+"/IP_Signals_shot"+str(shotNum)+"_BSC"+str(BSCnum)+".csv"
    shotData = pd.read_csv(fileName)
    dataSignal = shotData["ipSignal"].values
    # dataSignal = shotData["ipPrefitSignal"].values
    dataStdDev = shotData["ipError"].values

    # scanInfo = "./shotData/TAW2018_s24/Shot_24_a50.inf"
    # with open(scanInfo) as scanInfoFile:
    #     lines = scanInfoFile.readlines()
    #     lines = [line.rstrip() for line in lines]
    # scanTime = datetime.strptime(lines[10],"%c").time()
    # scanMins = float(scanTime.hour * 60 + scanTime.minute)
    # holdTimeMins = float(shotMins-scanMins)
    holdTimeMins = 30.
    fadeFactor   = fadingFunction(holdTimeMins)
    dataSignal  /= fadeFactor
    dataStdDev   = 0.15*dataSignal
    return dataSignal, dataStdDev

# Main chi squared reduction loop
def mainLoop(tempArray,engArray,dataSignal,dataStdDev,usableChannels,E_MeV,responseFunctions):
    """
    Iterates of the temperatures and energies for the hot electrons.
    Minimises the value of the chi squared (sum of the residuals).
    """
    numTemps, numEngs = round(len(tempArray)), round(len(engArray))
    tempInts, engInts = range(int(numTemps)),  range(int(numEngs))
    usbaleChanInds    = [i-1 for i in usableChannels]
    numberDataPoints  = len(usbaleChanInds)
    chAbsUse    = dataSignal[usbaleChanInds]
    stdDevUse   = dataStdDev[usbaleChanInds]
    recipStd    = 1./stdDevUse
    chiSqBest   = 1e99
    chiSqMatrix = np.zeros((int(numEngs),int(numTemps)))
    """  Main loop """
    E_keV = E_MeV*1e3
    for T_keV,t in zip(tempArray,tempInts):
        photonSource      = photonSourceDist(E_keV,T_keV)
        signals           = photonSource*responseFunctions
        syntheticSignals  = integrate.trapz(signals[usbaleChanInds],E_keV)
        synthSignalMatrix = np.ones((int(numEngs),numberDataPoints))*syntheticSignals
        fracMatrix        = synthSignalMatrix*engArray[:,np.newaxis]
        chiSqArray        = np.sum(np.square((fracMatrix-chAbsUse)*recipStd),axis=1)
        chiSqMatrix[:,t]  = chiSqArray
    chiSqMin   = chiSqMatrix.min()
    minChSqInd = np.unravel_index(np.argmin(chiSqMatrix,axis=None),chiSqMatrix.shape)
    bestEng    = engArray[minChSqInd[0]]
    bestTemp   = tempArray[minChSqInd[1]]
    print('Main loop complete.')
    # print('Best temp. [keV] = %.1f'%bestTemp)
    # print('Best engergy [J] = %.3e'%bestEng)
    """ Finding the minimum chi squared value """
    chiSqMatrix /= (numberDataPoints-2.)
    if(1): #This section normalises the chi squareds to the minimum (setting it to 1) which helps with the errors
        chiSqMatrix /= chiSqMatrix.min()
    chiSqMin     = chiSqMatrix.min()
    minChSqInd   = np.unravel_index(np.argmin(chiSqMatrix,axis=None),chiSqMatrix.shape)
    bestEng, bestTemp = engArray[minChSqInd[0]], tempArray[minChSqInd[1]]
    """ Find the best fitting signals """
    fitPhotonSrc   = photonSourceDist(E_keV,bestTemp)
    bestFitSignals = fitPhotonSrc*responseFunctions
    bestFitSignals = integrate.trapz(bestFitSignals[range(len(dataStdDev))],E_keV)
    bestFitSignals = bestFitSignals*bestEng
    return chiSqMatrix, chiSqMin, minChSqInd, bestEng, bestTemp, bestFitSignals, usbaleChanInds

# Legacy name for the quadrant plot
def crazyFigure(chiSqMatrix,chiSqMin,minChSqInd,bestEng,bestTemp,engArray,tempArray,bestFitSignals,usbaleChanInds):
    """ Find the best fitting T for each value of n and plots the chi squared """
    chiSqVariedE, chiSqEng = [], []
    for i in range(len(chiSqMatrix[0,:])):
        chiSqCol = chiSqMatrix[:,i]
        chiSqVariedE.append(chiSqCol.min())
        chiSqEng.append(engArray[chiSqCol.argmin()])
    chiSqVariedT, chiSqTemp = [], []
    for i in range(len(chiSqMatrix[:,0])):
        chiSqCol = chiSqMatrix[i,:]
        chiSqVariedT.append(chiSqCol.min())
    """ Crazy figure (legacy name) """
    try:
        crazyFig = True
        """ Finding the errors """
        engMinusErr  =  engArray[rFB.find_closest(chiSqVariedT[:minChSqInd[0]],chiSqMin+2.3)]
        engPlusErr   =  engArray[rFB.find_closest(chiSqVariedT[minChSqInd[0]:],chiSqMin+2.3)+minChSqInd[0]]
        tempMinusErr = tempArray[rFB.find_closest(chiSqVariedE[:minChSqInd[1]],chiSqMin+2.3)]
        tempPlusErr  = tempArray[rFB.find_closest(chiSqVariedE[minChSqInd[1]:],chiSqMin+2.3)+minChSqInd[1]]
        engArray,engMinusErr,bestEng,engPlusErr = engArray,engMinusErr,bestEng,engPlusErr
        tempErr = ((tempPlusErr-bestTemp)+(bestTemp-tempMinusErr))/2.
        engErr  = ((engPlusErr-bestEng)+(bestEng-engMinusErr))/2.
        ylabel = "E$_{hot}$ (J)"#'Total Hot Electron Energy [J]'
        print('Best temp. [keV] = %.1f, +/- %.1f (+ %.1f, - %.1f)'%(bestTemp,0.5*(tempPlusErr-tempMinusErr),tempPlusErr-bestTemp,bestTemp-tempMinusErr))
        print('Best engergy [J] = %.2f, +/- %.2f (+ %.2f, - %.2f)'%(bestEng, 0.5*(engPlusErr-engMinusErr),  engPlusErr-bestEng,  bestEng-engMinusErr))
    except Exception as ExceptErr:
        crazyFig = False
        print(ExceptErr)
        print("Could not find minimum!")
    # crazyFig = False
    """ Crazy figure """
    if(crazyFig):
        mpl.rcParams['font.size']=18
        fig5    = plt.figure(1,figsize=(10,10))#,constrained_layout=True)
        widths  = [1, 4]
        heights = [4, 1]
        spec5   = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=widths,height_ratios=heights,figure=fig5)
        spec5.update(wspace=0.1, hspace=0.1)
        """ Top Left, energy chi squared """
        ax = fig5.add_subplot(spec5[0, 0])
        plt.plot(chiSqVariedT,engArray,linewidth=3)
        plt.plot([chiSqMin+2.3,chiSqMin+2.3,chiSqMin+2.3],np.array([engMinusErr,bestEng,engPlusErr]),linestyle='-',color='k',linewidth=3,marker='_',markersize=10,mew=2)
        plt.ylim([min(engArray),max(engArray)])
        # plt.xlim([0,5])
        plt.xscale('log')
        plt.ylabel('E$_{hot}$ (J)')
        plt.xlabel('$\widetilde{\chi}^{2}}$')
        # plt.yscale('log')
        # plt.axis('off')
        """ Top Right (centre), main image """
        ax   = fig5.add_subplot(spec5[0, 1])
        X, Y = np.meshgrid(tempArray,engArray)
        im   = plt.pcolormesh(X,Y,np.log(chiSqMatrix),cmap='viridis')
        bestFit, = plt.plot(bestTemp,bestEng,linestyle="",marker="x",mew=16,ms=3,color="white",label=r"%i\pm%i, %i\pm%i"%(bestTemp,tempErr,bestEng,engErr))
        # plt.plot(tempArray,chiSqEng, linewidth=2,linestyle='--',color='grey')
        # cbar = plt.colorbar()#(pad=0.)
        # cbar.set_label('log($\widetilde{\chi}^{2}}$)')
        contourSet1 = plt.contour(X,Y,chiSqMatrix,[chiSqMin+1.0],colors='yellow')
        contourSet2 = plt.contour(X,Y,chiSqMatrix,[chiSqMin+2.3],colors='orange')
        contourSet3 = plt.contour(X,Y,chiSqMatrix,[chiSqMin+4.6],colors='red')
        labels = ["$\widetilde{\chi}^{2}_{min} + 1.0$ (25%)", "$\widetilde{\chi}^{2}_{min} + 2.3$ (68%)","$\widetilde{\chi}^{2}_{min} + 4.6$ (95%)", "T$_{hot}$ = %i$\pm$%i keV\nE$_{hot}$ = %i$\pm$%i J"%(bestTemp,np.ceil(tempErr),bestEng,np.ceil(engErr))]
        h1,l1 = contourSet1.legend_elements()
        h2,l2 = contourSet2.legend_elements()
        h3,l3 = contourSet3.legend_elements()
        # h4,l4 = plt.gca().legend_elements(bestFit)
        plt.legend([h1[0], h2[0], h3[0], bestFit], [labels[0], labels[1], labels[2], labels[3]], loc="upper right")
        # plt.legend(loc="upper right")
        plt.tick_params(axis='both',which='both',labelbottom=False,labeltop=False,labelleft=False,labelright=False)#,left=False,right=False,bottom=False,top=False)
        # plt.tick_params(axes='y',which='both',left=False,right=False,labelleft=False,labelright=False)
        # plt.xlabel('T$_{hot}$ [keV]')
        # plt.ylabel(ylabel)
        # plt.yscale('log')
        cb_ax = fig5.add_axes([.91,.2925,.02,0.5875])
        cbar  = fig5.colorbar(im,orientation='vertical',cax=cb_ax)
        cbar.set_label('log($\widetilde{\chi}^{2}}$)')
        """ Bottom Left, blank """
        ax = fig5.add_subplot(spec5[1, 0])
        plt.axis('off')
        """ Bottom Right, temperature chi squared """
        ax = fig5.add_subplot(spec5[1, 1])
        # plt.axis('off')
        plt.plot(tempArray,chiSqVariedE,linewidth=3)
        plt.plot([tempMinusErr,bestTemp,tempPlusErr],[chiSqMin+2.3,chiSqMin+2.3,chiSqMin+2.3],linestyle='-',color='k',linewidth=3,marker='|',markersize=10,mew=2)
        # plt.ylim([0,5])
        plt.xlim([min(tempArray),max(tempArray)])
        plt.yscale('log')
        plt.xlabel('T$_{hot}$ (keV)')
        plt.ylabel('$\widetilde{\chi}^{2}}$',rotation=0)
        ax = plt.gca()
        ax.yaxis.set_label_coords(-0.1,0.3)
        # plt.savefig("./ukhk1_chiSquared.png",format='png',bbox_inches='tight',pad_inches=0.0)
        # plt.savefig("./ukhk1_chiSquared.eps",format='eps',bbox_inches='tight',pad_inches=0.0)
        # plt.show()
    plt.figure(2,figsize=(9,7))
    channels = np.arange(1,len(dataSignal)+1,1)
    plt.errorbar(channels,                dataSignal,                    yerr=dataStdDev,                        color="tab:blue",  label="data")
    plt.errorbar(channels[usbaleChanInds],bestFitSignals[usbaleChanInds],yerr=0.1*bestFitSignals[usbaleChanInds],color="tab:orange",label="fit: %i keV, %.2e J"%(bestTemp,bestEng))
    plt.errorbar(channels,                bestFitSignals,                yerr=0.1*bestFitSignals,                color="tab:orange",linestyle="--",alpha=0.9)
    plt.legend(loc="best")
    plt.xticks(channels)
    plt.xlabel('Channel Number')
    plt.ylabel('IP Signal (mPSL/mm$^2$)')
    plt.yscale('log')
    plt.tight_layout()
    return

shotNums = [4,5,8,9,10,12,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44]
BSCnum   = 1

resFuncDict = {4:"designB", 5:"designC", 6:"designC", 8:"designD", 9:"designD", 10:"designD",12:"designD",13:"designD",14:"designD",15:"designD",16:"designD",
               17:"designD",19:"designE",20:"designE",21:"designE",23:"designE",24:"designE",25:"designE",26:"designE",27:"designE",28:"designE",29:"designE",
               30:"designE",31:"designE",32:"designE",33:"designE",34:"designE",35:"designE",36:"designE",37:"designE",38:"designE",40:"designE",41:"designE",
               42:"designE",43:"designE",44:"designE",45:"designE"}
for index,shotNum in enumerate(shotNums):
    print('*****')
    print('Temperature fitting for shotNum = %i'% shotNum)
    print('*****\n')

    """ Load in shot data """
    dataSignal, dataStdDev = loadInShotData(shotNum,BSCnum)

    """ Load in the relevant response functions """
    filterType = resFuncDict[shotNum]
    E_MeV, responseFunctions = rFB.generateResponseFunctions(filterType)

    """ Inputs for scanning parameters """
    tempArray = np.arange(5.00, 30.0, 0.1)
    engArray  = np.arange(0.01, 1.0,  0.005)
    numIPs    = len(dataSignal)
    usableChannels = range(2,numIPs+1,1)#[2,3,4,6,7,8,9,10]#

    """ Perform main scan """
    chiSqMatrix, chiSqMin, minChSqInd, bestEng, bestTemp, bestFitSignals, usbaleChanInds = mainLoop(tempArray,engArray,dataSignal,dataStdDev,usableChannels,E_MeV,responseFunctions)

    """ Plot the results """
    crazyFigure(chiSqMatrix,chiSqMin,minChSqInd,bestEng,bestTemp,engArray,tempArray,bestFitSignals,usbaleChanInds)
    plt.show()
