from img_converter import *
import os
i=1
#path1='/home/ashwin/Desktop/ResNet101/dataset_files'
with open('newtest.txt','r') as myfile:
	for line in myfile:
		line=line.replace('\n','')		
		#x=path1+line
		print(line)		
		i+=1
		convert(line)		

