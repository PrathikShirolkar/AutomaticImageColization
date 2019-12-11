import tensorflow as tf
import numpy as np
class Refinement(object):
    def get_kernel_size(self,factor):
        return 2 * factor - factor % 2
    def upsample_filt(self,size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    def bilinear_upsample_weights(self,factor, number_of_classes):
        filter_size = self.get_kernel_size(factor)
        weights = np.zeros((filter_size,filter_size,number_of_classes,number_of_classes), dtype=np.float32)
        upsample_kernel = self.upsample_filt(filter_size)
        for i in range(number_of_classes):
            weights[:, :, i, i] = upsample_kernel
        return weights
    def maskconv2d(self, x, kernel_size, num_o, dilation_factor,biased=False,scope='conv2d'):
        name=scope
        num_x = x.shape[3].value
        with tf.variable_scope(name) as scope:
            w = tf.get_variable('weights', shape=[kernel_size, kernel_size, num_x, num_o])
            o = tf.nn.atrous_conv2d(x, w, dilation_factor, padding='SAME')
            if biased:
                b = tf.get_variable('biases', shape=[num_o])
                o = tf.nn.bias_add(o, b)
        return o

    def __init__(self,lowres,highres,gray,scope):
        self.lowres=tf.convert_to_tensor(lowres,dtype=tf.float32)
        self.highres=highres
        self.gray=gray
        with tf.variable_scope(scope) as scope:
            self.train = tf.placeholder(tf.bool)
            self.construct_net()
    def softmax_loss(self,logits,labels):
        logits = tf.reshape(logits, [-1, 256])
        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, [-1])
        return tf.losses.sparse_softmax_cross_entropy(labels, logits)
    def construct_refinement_encoder(self):
        '''    
        outputs=self.maskconv2d(self.lowres,64,[3,3],strides=[2,2],mask_type=None,scope='conv1')
         
        outputs=self.maskconv2d(outputs,128,[3,3],strides=[1,1],mask_type=None,scope='conv2')
        outputs=self.maskconv2d(outputs,128,[3,3],strides=[2,2],mask_type=None,scope='conv3')
	
        outputs=self.maskconv2d(outputs,256,[3,3],strides=[1,1],mask_type=None,scope='conv4')
        outputs=self.maskconv2d(outputs,256,[3,3],strides=[2,2],mask_type=None,scope='conv5')
	
        

        outputs=self.maskconv2d(outputs,512,[3,3],strides=[1,1],mask_type=None,scope='conv6')
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[1,1],mask_type=None,scope='conv7')
        outputs=self.maskconv2d(outputs,256,[3,3],strides=[1,1],mask_type=None,scope='conv8')

        
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[2,2],mask_type=None,scope='conv9')
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[1,1],mask_type=None,scope='conv10')
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[2,2],mask_type=None,scope='conv11')
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[1,1],mask_type=None,scope='conv12')

        outputs=self.maskconv2d(outputs,1024,[5,5],strides=[1,1],mask_type=None,scope='conv13')
        
        outputs=self.maskconv2d(outputs,512,[3,3],strides=[1,1],mask_type=None,scope='conv14')
        outputs=self.maskconv2d(outputs,128,[3,3],strides=[1,1],mask_type=None,scope='conv15')
        outputs=self.maskconv2d(outputs,128,[3,3],strides=[1,1],mask_type=None,scope='conv16')
       

        outputs=tf.image.resize_images(outputs, [112, 112])
        outputs=self.maskconv2d(outputs,64,[3,3],strides=[1,1],mask_type=None,scope='conv17')
        outputs=self.maskconv2d(outputs,64,[3,3],strides=[1,1],mask_type=None,scope='conv18')
        outputs=tf.image.resize_images(outputs, [224, 224])
        outputs=self.maskconv2d(outputs,32,[3,3],strides=[1,1],mask_type=None,scope='conv19')
        outputs=self.maskconv2d(outputs,32,[3,3],strides=[1,1],mask_type=None,scope='conv20')
        outputs=self.maskconv2d(outputs,3,[1,1],strides=[1,1],mask_type=None,scope='conv21')
        '''
	
        layer=tf.image.resize_bilinear(self.gray, (28,28))
        self.lowres=tf.concat([self.lowres,layer],axis=3)
			
        outputs=self.maskconv2d(self.lowres,3,64,2,scope='conv1')
         
        outputs=self.maskconv2d(outputs,3,128,2,scope='conv2')
        outputs=self.maskconv2d(outputs,3,128,4,scope='conv3')
	
        outputs=self.maskconv2d(outputs,3,256,2,scope='conv4')
        outputs=self.maskconv2d(outputs,3,256,4,scope='conv5')
	
        

        outputs=self.maskconv2d(outputs,3,512,2,scope='conv6')
        outputs=self.maskconv2d(outputs,3,512,2,scope='conv7')
        outputs=self.maskconv2d(outputs,3,256,2,scope='conv8')

        
        outputs=self.maskconv2d(outputs,3,512,4,scope='conv9')
        outputs=self.maskconv2d(outputs,3,512,2,scope='conv10')
        outputs=self.maskconv2d(outputs,3,512,4,scope='conv11')
        outputs=self.maskconv2d(outputs,3,512,2,scope='conv12')

        outputs=self.maskconv2d(outputs,5,1024,2,scope='conv13')
        
        outputs=self.maskconv2d(outputs,3,512,2,scope='conv14')
        outputs=self.maskconv2d(outputs,3,128,2,scope='conv15')
        outputs=self.maskconv2d(outputs,3,128,2,scope='conv16')
       
        batch_size,h,w,c=outputs.shape
        #outputs=tf.image.resize_images(outputs, [112, 112])
        factor=4
        upsample_filter_np = self.bilinear_upsample_weights(4,128)
        output_shape=tf.convert_to_tensor([batch_size, 112, 112, 128],dtype=tf.int32)
        outputs = tf.nn.conv2d_transpose(outputs, upsample_filter_np,output_shape=output_shape,strides=[1, factor, factor, 1],name='upsample1')

        outputs=self.maskconv2d(outputs,3,64,2,scope='conv17')
        outputs=self.maskconv2d(outputs,3,64,2,scope='conv18')
        batch_size,h,w,c=outputs.shape
        #outputs=tf.image.resize_images(outputs, [224, 224])
        factor=2
        upsample_filter_np = self.bilinear_upsample_weights(2,64)
        output_shape=tf.convert_to_tensor([batch_size, 224, 224, 64],dtype=tf.int32)
        outputs = tf.nn.conv2d_transpose(outputs, upsample_filter_np,output_shape=output_shape,strides=[1, factor, factor, 1],name='upsample2')

        outputs=self.maskconv2d(outputs,3,32,2,scope='conv19')
        outputs=self.maskconv2d(outputs,3,32,2,scope='conv20')
        outputs=self.maskconv2d(outputs,1,256*3,2,scope='conv21')
        return outputs
    
    def construct_net(self):
        self.refoutputs = self.construct_refinement_encoder()
        self.loss = self.softmax_loss(self.refoutputs, self.highres)
        tf.summary.scalar('loss', self.loss)
