import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os;
import scipy.misc;
import cv2;

def script_saveTrainParameters():
	net = caffe.Net('../models/deploy.prototxt', '../models/alexnet_cvgj_iter_320000.caffemodel',caffe.TRAIN);
	out_dir='../models';
	for layer_name in net.params.keys():
		print layer_name;
		if 'scale' in layer_name or 'bn' in layer_name:
			out_file_name=os.path.join(out_dir,'alexnet_'+layer_name.replace('/','_'));
			print out_file_name;
			weight=net.params[layer_name][0].data;
			np.save(out_file_name+'_weights.npy',weight);
			bias=net.params[layer_name][1].data;
			np.save(out_file_name+'_bias.npy',bias);
			if 'bn' in layer_name:
				ma=net.params[layer_name][2].data;
				np.save(out_file_name+'_avg.npy',ma);
			
			if len(net.params[layer_name])>2:
				print net.params[layer_name][2].data;

	
def script_saveTest():
	net = caffe.Net('../models/deploy.prototxt', '../models/alexnet_cvgj_iter_320000.caffemodel',caffe.TEST);
	out_dir='../models';
	resize_size=[227,227];

	test_file='../data/train/ALB/img_01105.jpg';

	img=cv2.imread(test_file);
	img = cv2.resize(img, (resize_size[1],resize_size[0]))
	img=np.array(img);

	img=np.transpose(img,(2,0,1));

	im_input=img[np.newaxis,:,:,:,]
	net.blobs['data'].reshape(*im_input.shape)
	net.blobs['data'].data[...] = im_input
	
	net.forward();

	for layer_name in net.blobs.keys():
		print layer_name,np.min(net.blobs[layer_name].data),np.max(net.blobs[layer_name].data);
	
	output=net.blobs['prob'].data
	np.save('../models/test_caffe.npy',output);

def main():
	caffe.set_device(1);
	caffe.set_mode_gpu();
	script_saveTest();
	# script_saveTrainParameters();

if __name__=='__main__':
	main();