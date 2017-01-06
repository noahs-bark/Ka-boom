print "\n                                                          DFD-Delhi - DYNAMITE FISHING DETECTION AND LOCALIZATION \n"
print "Improting libraries..."
import scipy
import numpy
import bigfloat
from scipy.io import wavfile
from scipy import signal
from numpy import *
from TanSig import TanSig
from scipy.linalg import svd
from lpc import levinson_1d
from scikits.audiolab import wavread
print "All libraries imported \n"
##############################################################################
print "Reading sound file..."
filename = "/root/DFD-delhi/vagrant-python/sounds/bomb.wav"
y,fs,encoding = wavread(filename)
#y =y[:,0]
print "Wav file obtained \n"
##############################################################################
print "Filtering..."
a=1
b=loadtxt('/root/DFD-delhi/vagrant-python/ham.txt')
yw=signal.lfilter(b,a,numpy.squeeze(y))
x=yw
x[0]=10e-08
print "Filtered output \n"
##############################################################################
print "Calculating Power Spectrum..."
N = len(x)   
N_inv = 10e+05*1/N         
ws = 2*pi/N 
p1 = numpy.abs(power(numpy.fft.fft(x),2))
p1 = numpy.multiply(p1,N_inv/10e+05)
wnorm = numpy.linspace(-pi,pi,N)
wnorm = numpy.multiply(wnorm[1:len(p1)],fs)
M = len(wnorm)
print "Power Spectrum Calculated \n"
##############################################################################
print "Extraction of Statistical Features..."
print "1.   Total Power"
ttp = 0;
for i in range (0,M):
	ttp = ttp + p1[i]
############################
print "2.   Mean Power"
mnp = ttp/M
############################
print "3.   Median Power"
mdf = ttp/2
############################
print "4.   Mean Frequency"
mnf = 0
for i in range (0,M):
	mnf = mnf + numpy.multiply(wnorm[i],p1[i])
mnf = mnf/ttp
############################
print "5.   Peak Frequency"
pkf = numpy.max(p1)
############################
print "6.   Mean Absolute Value" 
mav = 0
for i in range(0,N):
	mav = mav + numpy.abs(x[i])
mav = mav/N
############################
print "7.   Variance" 
var = 0
for i in range(0,N):
	var = var + power(x[i]-mav,2)
var = var/(N-1)
############################
print "8.   Standard Deviation"
sd = numpy.sqrt(var)
############################
print "9.   Root Mean Square"
rms = 0
for i in range(0,N):
	rms = rms + power(x[i],2)
rms = numpy.sqrt(rms/N)
############################
print "10.  Average Amplitude Change"
aac = 0
for i in range(0,N-1):
	aac = aac + numpy.abs(x[i+1]-x[i])
aac = aac/N
##############################################################################
fv = [[mav],[var],[sd],[rms],[aac],[mnp],[ttp],[mdf],[mnf],[pkf]]
fv = numpy.asarray(fv)
print "Statistical Features Calculated \n"
##############################################################################
print "Calculating LPC coefficients..."
r=numpy.correlate(x,x,'full')
(temp,e,k)=levinson_1d(r,49)
temp = zip(temp)
test=numpy.concatenate([fv,temp])
x1=numpy.zeros(60)
for i in range(0,60):
	if(i==0):
		x1[i]=1
		continue
	x1[i]=test[i-1]
xx1 = x1[0:len(x1)-2]
print "LPC coefficients Calculated \n"
##############################################################################
print "Neural Network Feedforwarding..."
print "Obtaining weights and bias matrix..."
w1=loadtxt('/root/DFD-delhi/vagrant-python/w1.txt')
w2=loadtxt('/root/DFD-delhi/vagrant-python/w2.txt')
b1=loadtxt('/root/DFD-delhi/vagrant-python/b1.txt')
b2=loadtxt('/root/DFD-delhi/vagrant-python/b2.txt')

print "Layer - 1 calculation: Tansig "
v1=numpy.dot(w1,xx1)
v1=numpy.add(v1,b1)
y1=numpy.zeros(len(v1))
for i in range(0, len(v1)):
	y1[i]=2/(1+bigfloat.exp((numpy.multiply(-2,v1[i]))))-1

print "Layer - 2 calculation: Softmax"
v2=numpy.dot(w2,y1)
v2=v2+b2;


def softmax(w, t = 1.0):
    e = numpy.exp(numpy.array(w) / t)
    dist = e / numpy.sum(e)
    return dist

y3 = softmax(v2)
print "\nNeural Network Output :"
print y3
print " \nHard thresholding output:"

if ((y3[0] >= 0.5) and (y3[1] < 0.5)):
	y3 = [1, 0]
	print y3
	print "\n         Blast-Detected           \n"

else :
	y3 = [0,1]
	print y3
	print "\n          Not-a-blast              \n"































