import numpy
import bigfloat

def TanSig(x):
	# n=len(x)
	# if(n==1):
	# print x
	y=2/(1+bigfloat.exp((numpy.multiply(-2,x))))-1
	# print y
	return y
	# else:
		# y=numpy.tanh(x)
		# mx=max(y)
		# mi=min(y)
		# if(mx==mi):
		# 	mi=0
		# #[minFrom..maxFrom] -> [minTo..maxTo]
		# #mappedValue = minTo + (maxTo - minTo) * ((value - minFrom) / (maxFrom - minFrom));
		# y=-1+2*((y-mi)/(mx-mi))
		# return y
