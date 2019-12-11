from os import walk
import os
f = []
mypath='/home/ashwin/Desktop/lsun_test/'
for (dirpath, dirnames, filenames) in walk(mypath):
	f.extend(filenames)
for fi in f:
	with open('newtest.txt','a+') as myfile:
		myfile.write(os.path.join(mypath,fi)+'\n')
	
    
