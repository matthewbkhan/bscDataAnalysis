from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import numpy as np
import scipy as sp
import sys

# Simple 1D super-Gaussian function
def superGaussian(x, x0, c, A, n, y0):
    """
    x  - pixel number x-axis
    x0 - centre of the sG
    A  - amplitude of the sG (no background)
    c  - standard deviation
    n  - order of the sG (higher is more top-hat like), given as n/2 in equation
    y0 - background level
    """
    return A*np.exp(-((x-x0)*(x-x0)/(2.*c*c))**(n/2.)) + y0

# 2D super-Gaussian for fitting individual IPs
def twoDGaussian(meshTuple, A, x0, y0, a, b, n, bg, theta):
    """
    meshTuple - (x,y) grid
    x0, y0    - position of the sG centre in x and y
    a,  b     - related to sigma_x and sigma_y
    bg        - background signal level
    theta     - offest angle of sG
    output is ravelled at end to improve speed
    """
    (x,y) = meshTuple
    alpha =  (np.cos(theta)**2.) / (a**2.)     +  (np.sin(theta)**2.) / (b**2.)
    beta  = -(np.sin(2.*theta))  / (2.*a**2.)  +  (np.sin(2.*theta))  / (2.*b**2.)
    gamma =  (np.sin(theta)**2.) / (a**2.)     +  (np.cos(theta)**2.) / (b**2.)
    output = A*np.exp(-((alpha*((x-x0)*(x-x0)) + 2.*beta*(x-x0)*(y-y0) + gamma*((y-y0)*(y-y0))))**(n/2.)) + bg
    return output.ravel()

# contrained 2D super-Gaussian for fitting individual IPs
def twoDGaussianCons(meshTuple, A, x0, y0, n, bg):
    """
    meshTuple - (x,y) grid
    x0, y0    - position of the sG centre in x and y
    bg        - background signal level
    output is ravelled at end to improve speed
    """
    (x,y) = meshTuple
    a, b  = 1.236e2, 1.321e2 # Numbers found from the mean of uncontrained fits
    theta = 5.569e-1
    alpha =  (np.cos(theta)**2.) / (a**2.)     +  (np.sin(theta)**2.) / (b**2.)
    beta  = -(np.sin(2.*theta))  / (2.*a**2.)  +  (np.sin(2.*theta))  / (2.*b**2.)
    gamma =  (np.sin(theta)**2.) / (a**2.)     +  (np.cos(theta)**2.) / (b**2.)
    output = A*np.exp(-((alpha*((x-x0)*(x-x0)) + 2.*beta*(x-x0)*(y-y0) + gamma*((y-y0)*(y-y0))))**(n/2.)) + bg
    return output.ravel()
# twoDGaussian = twoDGaussianCons

# Load in the IP data and the scanning info
def loadShotData(shotNum,dateTime):
    """
    shotNum  - shot number
    dateTime - additional file name, some form of the date/time of the scan
    """
    shotNumStr = str(shotNum)
    if(shotNum<10):filePath, fileName = "./shot_"+"0"+shotNumStr+"/", "Shot"+shotNumStr+"_Scan1_"+dateTime+"-[Phosphor]"
    else:          filePath, fileName = "./shot_"+shotNumStr+"/",     "Shot"+shotNumStr+"_Scan1_"+dateTime+"-[Phosphor]"
    imgFilename, scanInfo = fileName+".img", fileName+".inf"
    with open(filePath+scanInfo) as scanInfoFile:
        lines = scanInfoFile.readlines()
        lines = [line.rstrip() for line in lines]
    shapeY, shapeX = int(lines[7]), int(lines[6])
    shape = (shapeY, shapeX) # matrix size
    dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
    sensitivity = int(lines[8]) # scanner sensitivity, typically 4000
    resolution  = int(lines[3]) # pixel resolution in um
    latitude    = int(lines[9]) # dynamic range, typically 5
    bitDepth    = int(lines[5]) # bit depth, typically 16
    imgFile  = open(filePath+imgFilename, 'rb')
    imgData  = np.fromfile(imgFile, dtype)
    return imgData,shape,sensitivity,resolution,latitude,bitDepth,filePath

# Convert the raw .img file to PSL
def convertToPSL(imgData,shape,sensitivity,resolution,latitude,bitDepth):
    """
    imgData     - the raw .img data
    shape       - the x, y shape of the data
    sensitivity - sensitivity of the scanner
    resolution  - resolution of the scanner
    latitude    - latitude of the scanner
    bitDepth    - bitDepth  of the scanner
    DOI's: 10.1063/1.4935582, 10.1088/0957-0233/19/9/095301, 10.1063/1.4893780
    """
    pslData  = (resolution/100.)*(resolution/100.) * (4000./sensitivity) * pow(10., latitude*((imgData/(2.**bitDepth-1.))-0.5))
    pslImage = pslData.reshape(shape)
    return pslImage

"""
Dictionary for the date/time of the scan
"""
dateTimeDict = {4:"20221004-205157", 5:"20221005-113627", 8:"20221006-105247", 9:"20221006-122227", 10:"20221006-141808",12:"20221006-175103",
                13:"20221006-194259",14:"20221007-105352",15:"20221007-130759",16:"20221007-152615",17:"20221007-164930",19:"20221010-130735",
                20:"20221010-152126",21:"20221010-173612",23:"20221011-111756",24:"20221011-132551",25:"20221011-152747",26:"20221011-173525",
                27:"20221011-185933",28:"20221011-202851",29:"20221012-113214",30:"20221012-132525",31:"20221012-152155",32:"20221012-170146",
                33:"20221012-183716",34:"20221012-200323",35:"20221013-124806",36:"20221013-161334",37:"20221013-183554",38:"20221013-200158",
                40:"20221014-123102",41:"20221014-141338",42:"20221014-154322",43:"20221014-171545",44:"20221014-184133",45:"20221014-200955"}


shotNums = [4,5,8,9,10,12,13,14,15,16,17,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,40,41,42,43,44]
for shotNum in shotNums:
    print(shotNum)
    BSCnum   = 1
    figNum   = 1
    plotting = False
    printing = False
    dateTime = dateTimeDict[shotNum]
    imgData, shape, sensitivity, resolution, latitude, bitDepth, filePath = loadShotData(shotNum,dateTime)
    pslImage = convertToPSL(imgData,shape,sensitivity,resolution,latitude,bitDepth)

    """ Copied from online: https://www.tutorialspoint.com/store-mouse-click-event-coordinates-with-matplotlib """
    # Function to print mouse click event coordinates
    def onclick(event):
        global centresArray
        centresArray.append([int(event.xdata),int(event.ydata)])
    # Create a figure and a set of subplots
    fig, ax   = plt.subplots()
    plotImage = np.log(ndimage.median_filter(pslImage.ravel(),size=5))
    plotImage = np.reshape(plotImage,np.shape(pslImage))
    ax.imshow(plotImage,cmap="inferno",vmax=-4.5,vmin=-6.5)
    centresArray = []
    # Bind the button_press_event with the onclick() method
    fig.canvas.mpl_connect('button_press_event', onclick)
    # Display the plot
    plt.show()

    """
    2D fits to individual IPs: crop off an IP and perform a fit to that
    A quick vertical fit is done to find the approximate y0
    """
    peakSignalList  = []
    ipPrefitSignals = []
    ipXCentres      = []
    ipYCentres      = []
    peakErrorsList  = []
    for i in range(len(centresArray)):
        ipCenX    = centresArray[i][0]
        ipCenY    = centresArray[i][1]
        halfWidth = 175
        ipImage   = pslImage[np.max([ipCenY-halfWidth,0]):ipCenY+halfWidth,ipCenX-halfWidth:ipCenX+halfWidth]
        ipImage   = gaussian_filter(ipImage,sigma=3)

        """ y Pre-fit """
        yWidth    = np.shape(ipImage)[0]
        yLineOut  = np.mean(ipImage[:,int(yWidth/2.-25):int(yWidth/2.+25)],axis=1)
        yFitGuess = [2.*yWidth/4.,95., np.mean(yLineOut[int(-50.+yWidth/2.):int(50.+yWidth/2.)])-np.min(yLineOut),25., np.min(yLineOut)]
        uppBound  = [3.*yWidth/4.,105.,np.max(yLineOut),                                                          100.,np.mean(yLineOut)]
        lowBound  = [1.*yWidth/4.,85., 0.,                                                                        2.,  np.min(yLineOut)]
        try:
            yFit,_ = curve_fit(superGaussian, np.arange(0,yWidth,1), yLineOut, p0=yFitGuess,bounds=[lowBound,uppBound])
            # for l,g,p,u in zip(lowBound,yFitGuess,yFit,uppBound):
            #     print(l,g,p,u)
        except Exception as ExceptErr:
            print("yFit fail, IP Num: %s"%(i+1),ExceptErr)
            for l,g,u in zip(lowBound,yFitGuess,uppBound):
                print(l,g,u)
            plt.plot(np.arange(0,yWidth,1),yLineOut)
            plt.plot(np.arange(0,yWidth,1),superGaussian(np.arange(0,yWidth,1),*yFitGuess))
            plt.show()

        """ x Pre-fit """
        xWidth    = np.shape(ipImage)[1]
        xLineOut  = np.mean(ipImage[int(yWidth/2.-25):int(yWidth/2.+25),:],axis=0)
        xFitGuess = [2.*xWidth/4.,95., np.mean(xLineOut[int(-50.+xWidth/2.):int(50.+xWidth/2.)])-np.min(xLineOut),25., np.min(xLineOut)]
        uppBound  = [3.*xWidth/4.,105.,np.max(xLineOut),                                                          100.,np.mean(xLineOut)]
        lowBound  = [1.*xWidth/4.,85., 0.,                                                                        2.,  np.min(xLineOut)]
        try:
            xFit,_ = curve_fit(superGaussian, np.arange(0,xWidth,1), xLineOut, p0=xFitGuess,bounds=[lowBound,uppBound])
            # for l,g,p,u in zip(lowBound,xFitGuess,xFit,uppBound):
            #     print(l,g,p,u)
        except Exception as ExceptErr:
            print("xFit fail, IP Num: %s"%(i+1),ExceptErr)
            for l,g,u in zip(lowBound,xFitGuess,uppBound):
                print(l,g,u)
            plt.plot(np.arange(0,xWidth,1),xLineOut)
            plt.plot(np.arange(0,xWidth,1),superGaussian(np.arange(0,xWidth,1),*xFitGuess))
            plt.show()

        """ Full 2D fit; make guess, sort bounds, perform fit, plot results """
        x, y  = np.linspace(0, np.shape(ipImage)[1], np.shape(ipImage)[1]), np.linspace(0, np.shape(ipImage)[0], np.shape(ipImage)[0])
        x, y  = np.meshgrid(x, y)
        AGuess  = np.mean([xFit[2],yFit[2]])
        x0Guess = xFit[0]
        y0Guess = yFit[0]
        aGuess  = 0.975*np.sqrt(2.)*np.mean([xFit[1],yFit[1]])
        bGuess  = 1.050*np.sqrt(2.)*np.mean([xFit[1],yFit[1]])
        nGuess  = np.mean([xFit[3],yFit[3]])
        bgGuess = np.mean([xFit[4],yFit[4]])
        thtaGuess = 30.*np.pi/180.
        guessIP = [AGuess,x0Guess,y0Guess,aGuess,bGuess,nGuess,bgGuess,thtaGuess]
        try:
            uppBnds  = [np.max(ipImage),2.5*xWidth/4.,2.5*yWidth/4.,np.sqrt(2.)*105.,np.sqrt(2.)*110.,100.,np.mean(ipImage),np.pi]
            lowBnds  = [0.,             1.5*xWidth/4.,1.5*yWidth/4.,np.sqrt(2.)*85., np.sqrt(2.)*85., 2.,  0.,             -np.pi]
            """ Bounds are set relative to the fit of the first IP """
            if(i!=0 and IPpopt0[7]>0.0):
                uppBnds[3:5],uppBnds[7]=1.05*IPpopt0[3:5],1.1*IPpopt0[7]
                lowBnds[3:5],lowBnds[7]=0.95*IPpopt0[3:5],0.9*IPpopt0[7]
            elif(i!=0 and BSCnum<0.0):
                uppBnds[3:5],uppBnds[7]=1.05*IPpopt0[3:5],0.9*IPpopt0[7]
                lowBnds[3:5],lowBnds[7]=0.95*IPpopt0[3:5],1.1*IPpopt0[7]
            """ Change guess to bounds if out of bounds """
            for ind in range(len(guessIP)):
                if(guessIP[ind]<lowBnds[ind]):guessIP[ind]=lowBnds[ind]
                if(guessIP[ind]>uppBnds[ind]):guessIP[ind]=uppBnds[ind]
            """ Ignore the pixels greater than 10 times the peak signal guess """
            xMesh, yMesh = x[ipImage<(10.*AGuess)+bgGuess], y[ipImage<(10.*AGuess)+bgGuess]
            ipFit        = ipImage[ipImage<(10.*AGuess)+bgGuess]
            IPpopt,IPpcov = curve_fit(twoDGaussian, (xMesh, yMesh), ipFit.ravel(), p0=guessIP, bounds=[lowBnds,uppBnds])
            fittingErrors = np.sqrt(np.diag(IPpcov))
            peakSignalList.append(IPpopt[0]/((resolution*1e-3)*(resolution*1e-3)))
            ipPrefitSignals.append(AGuess/((resolution*1e-3)*(resolution*1e-3)))
            ipXCentres.append(ipCenX)
            ipYCentres.append(ipCenY)
            peakErrorsList.append(np.abs((AGuess-IPpopt[0])/((resolution*1e-3)*(resolution*1e-3))))
            if(i==0):IPpopt0=np.copy(IPpopt)
        except Exception as ExceptErr:
            print("2D IP Fit, IP Num: %s"%(i+1),ExceptErr)
            for l,g,u in zip(lowBnds,guessIP,uppBnds):
                print(l,g,u)
            IPpopt = guessIP
        if(printing):
            if(i==0):
                print("\n###############")
                print("2D IP Fits")
                print("###############")
            params   = ["A   ", "x0  ", "y0  ", "a   ", "b   ", "n   ", "bg  ", "thta"]
            print("###############")
            print("IP Num: %s"%(i+1))
            for param,p,g in zip(params,IPpopt,guessIP):
                print("param %s: popt %.3e, guess %.3e"%(param,p,g))
        if(plotting):
            plt.figure(figNum)
            figNum += 1
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(np.arange(0,xWidth,1),xLineOut,                                       label="xlineout")
            ax1.plot(np.arange(0,xWidth,1),superGaussian(np.arange(0,xWidth,1),*xFit),     label="xFit")
            ax1.plot(np.arange(0,xWidth,1),superGaussian(np.arange(0,xWidth,1),*xFitGuess),label="xFitGuess")
            ax1.set_xlabel("x-pixel")
            ax1.set_ylabel("mean(PSL)")
            ax1.legend(loc="best")

            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(np.arange(0,yWidth,1),yLineOut,                                       label="ylineout")
            ax2.plot(np.arange(0,yWidth,1),superGaussian(np.arange(0,yWidth,1),*yFit),     label="yFit")
            ax2.plot(np.arange(0,yWidth,1),superGaussian(np.arange(0,yWidth,1),*yFitGuess),label="yFitGuess")
            ax2.set_xlabel("y-pixel")
            ax2.set_ylabel("mean(PSL)")
            ax2.legend(loc="best")

            plt.figure(figNum)
            figNum += 1
            plt.title("IP Num %i: data"%(i+1))
            plt.imshow(np.log(ipImage),cmap="inferno",vmax=np.max(np.log(ipImage)),vmin=np.min(np.log(ipImage)))

            plt.figure(figNum)
            figNum += 1
            plt.title("IP Num %i: guess"%(i+1))
            dataGuess = twoDGaussian((x, y),*guessIP).reshape(np.shape(ipImage)[0],np.shape(ipImage)[1])
            plt.imshow(np.log(dataGuess),cmap="inferno",vmax=np.max(np.log(ipImage)),vmin=np.min(np.log(ipImage)))

            plt.figure(figNum)
            figNum += 1
            plt.title("IP Num %i: fit"%(i+1))
            dataPlot = twoDGaussian((x, y),*IPpopt).reshape(np.shape(ipImage)[0],np.shape(ipImage)[1])
            plt.imshow(np.log(dataPlot),cmap="inferno",vmax=np.max(np.log(ipImage)),vmin=np.min(np.log(ipImage)))

            plt.figure(figNum)
            figNum += 1
            plt.title("IP Num %i: diff"%(i+1))
            dataDiff = ipImage-dataPlot
            plt.imshow(dataDiff,cmap="inferno")
            plt.show()
    print("Shot Num.: %i"%shotNum)
    for ipNum,peakVal,peakErr in zip(range(1,len(peakSignalList)+1,1),peakSignalList,peakErrorsList):
        print("%i, %.3f, %.3f"%(ipNum,peakVal,peakErr))
    df = pd.DataFrame({'IPnum':          range(1,len(peakSignalList)+1,1),
                       'ipSignal':       peakSignalList,
                       'ipPrefitSignal': ipPrefitSignals,
                       'xCentre':        ipXCentres,
                       'yCentre':        ipYCentres,
                       'ipError':        peakErrorsList})
    if(shotNum<10):df.to_csv(filePath+"IP_Signals_shot0"+str(shotNum)+"_BSC"+str(BSCnum)+".csv")
    else:          df.to_csv(filePath+"IP_Signals_shot"+str(shotNum)+"_BSC"+str(BSCnum)+".csv")
