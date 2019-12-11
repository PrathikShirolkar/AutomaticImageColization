import os
from os import listdir
from os.path import isfile, join
src='/home/ashwin/Desktop/PixColor/refinementdata/'
src1='/home/ashwin/Desktop/PixColor/graydata/'
files = [f for f in listdir(src) if isfile(join(src, f))]
for f in files:
	#x=f.replace('generate','hr')	
	if isfile(join(src1,f)):
		#y=f.replace('generate','lr')
		with open('reftrain.txt','a+',encoding='utf-8') as myfile:
			myfile.write(join(src,f)+" "+join(src,f)+" "+join(src1,f)+'\n')
