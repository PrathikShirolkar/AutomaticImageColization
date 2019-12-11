from skimage import color
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
def ycbcr2rgb(im):
	xform = np.squeeze(np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]]))
	rgb = im.astype(np.float)
	rgb[:,:,[1,2]] -= 128
	rgb = rgb.dot(xform.T)
	np.putmask(rgb, rgb > 255, 255)
	np.putmask(rgb, rgb < 0, 0)
	return np.uint8(rgb)
def rgb2ycbcr(im):
	xform = np.squeeze(np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]]))
	ycbcr = im.dot(xform.T)
	ycbcr[:,:,[1,2]] += 128
	return np.uint8(ycbcr)

def convert(path):
	path1='/home/ashwin/Desktop/PixColor/testchromadata/'
	path2='/home/ashwin/Desktop/PixColor/testgraydata/'
	im = Image.open(path,mode='r')
	#im = im.resize((224, 224), Image.ANTIALIAS)
	shapes=np.asarray(im).shape
	w=shapes[1]	
	h=shapes[0]
	image = rgb2ycbcr(np.squeeze(np.asarray(im)))

	img=Image.fromarray(image)
	YY=image[:,:,0]

	a=np.full((h,w), 128, dtype=np.uint8)
	Cb=image[:,:,1]
	Cr=image[:,:,2]


	just_Y=cv2.merge((YY,a,a))
	just_Cb=cv2.merge((a,Cb,a))
	just_Cr=cv2.merge((a,a,Cr))
	just_CbCr=cv2.merge((a,Cb,Cr))
		
	YY=ycbcr2rgb(just_Y)
	Cb=ycbcr2rgb(just_Cb)
	Cr=ycbcr2rgb(just_Cr)
	CbCr=ycbcr2rgb(just_CbCr)
	
	name=path.split('/')[-1]
	yyimg=Image.fromarray(np.roll(YY, 1, axis=-1))
	yyimg.save(path2+name)
	
	#Cbimg=Image.fromarray(np.roll(Cb, 1, axis=-1))
	#Cbimg.show()

	#Crimg=Image.fromarray(np.roll(Cr, 1, axis=-1))
	#Crimg.show()

	CbCrimg=Image.fromarray(np.roll(CbCr, 1, axis=-1))
	CbCrimg.save(path1+name)
	
