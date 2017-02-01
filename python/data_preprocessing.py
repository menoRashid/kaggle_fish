import numpy as np;
import util;
import os;
import random;
import cv2;

def resize(in_file,out_file,resize_size):
	im=cv2.imread(in_file);
	im=cv2.resize(im,tuple(resize_size));
	cv2.imwrite(out_file,im)

def resizeImages(in_dir,out_dir,ext='.jpg',resize_size=[256,256],overwrite=False):
	in_files=util.getFilesInFolder(in_dir,ext);
	out_files=[];
	for idx_in_file,in_file in enumerate(in_files):
		if idx_in_file%100==0:
			print idx_in_file;
		out_file=os.path.join(out_dir,os.path.split(in_file)[1]);
		out_files.append(out_file);
		if not os.path.exists(out_file) or overwrite:
			resize(in_file,out_file,resize_size);
	return out_files;

def script_resizeTrainData():
	path_to_train='../data/train';
	dir_train_new='../data/train_256';
	paths_file='../data/paths_256.txt';
	util.mkdir(dir_train_new);	

	sub_dirs=['BET','ALB','DOL','LAG','NoF','OTHER','SHARK','YFT'];
	sub_dirs.sort();

	lines_all=[];

	for idx_sub_dir,sub_dir in enumerate(sub_dirs):
		out_dir_curr=os.path.join(dir_train_new,sub_dir);
		in_dir_curr=os.path.join(path_to_train,sub_dir);
		util.mkdir(out_dir_curr);
		im_files=resizeImages(in_dir_curr,out_dir_curr);
		lines=[file_curr+' '+str(idx_sub_dir) for file_curr in im_files];
		lines_all=lines_all+lines;

	util.writeFile(paths_file,lines_all);

def script_resizeTestData():
	path_to_test='../data/test_stg1';
	dir_test_new='../data/test_256';
	paths_file='../data/test_256.txt';
	util.mkdir(dir_test_new);
	out_files=resizeImages(path_to_test,dir_test_new);
	out_files=[file_curr+' 0' for file_curr in out_files];
	util.writeFile(paths_file,out_files)

def script_createTrainValSplit(val_split=0.1):
	paths_file='../data/paths_256.txt';
	train_file='../data/train_256.txt';
	val_file='../data/val_256.txt';

	lines=util.readLinesFromFile(paths_file);
	classes=[line_curr.split(' ')[1] for line_curr in lines];
	lines=np.array(lines);
	classes=np.array(classes);
	train_lines=[];
	val_lines=[];
	for class_curr in np.unique(classes):
		idx_rel=np.where(classes==class_curr)[0];
		num_to_val=int(idx_rel.size*val_split);
		np.random.shuffle(idx_rel);
		val_lines_curr=lines[idx_rel[:num_to_val]];
		train_lines_curr=lines[idx_rel[num_to_val:]];
		train_lines=list(train_lines_curr)+train_lines;
		val_lines=val_lines+list(val_lines_curr);

	assert not np.any(np.in1d(train_lines,val_lines));
	random.shuffle(train_lines);
	random.shuffle(val_lines);
	print train_file,len(train_lines);
	print val_file,len(val_lines);
	util.writeFile(train_file,train_lines);
	util.writeFile(val_file,val_lines);

def main():

	paths_file='../data/paths_256.txt';
	train_file='../data/train_256.txt';
	val_file='../data/val_256.txt';
	weights_file='../data/train_weights.npy';

	lines=util.readLinesFromFile(paths_file);
	classes=[int(line_curr.split(' ')[1]) for line_curr in lines];
	classes_set=set(classes);
	n_samples=len(classes);
	n_classes=len(classes_set);
	class_counts=[classes.count(c) for c in range(n_classes)]
	print class_counts;
	
	balanced_weights=float(n_samples)/(n_classes*np.array(class_counts))
	balanced_weights=balanced_weights/np.sum(balanced_weights);
	print balanced_weights
	np.save(weights_file,balanced_weights);
	
	
	

		


if __name__=='__main__':
	main();