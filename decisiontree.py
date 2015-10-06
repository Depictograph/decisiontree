import numpy as np
import scipy.io
from scipy.stats import mode
import math
import csv
#  IG(i, t) = (H - |D1| * H1 + |D2| * H2) / |D|
#  find t such that IG is maximized. If feaature instances are sorted, changing threshold according to feature value makes IG a step function
#  naive computational complexity D*N*N by trying all threshold values. If feature instances are sorted, iterating over 1 value of the threshold only changes cardinality of sets by 1, if we keep track of data set sizes 

	
class node: 
	def __init__(self, data, labels):
		self.data = data
		self.left = None
		self.right = None
		self.splitrule = None #(feature, threshold) 
		self.label = None
		self.labels = labels
	def splitnode(self):
		assert self.splitrule != None
		Z = self.data[:, self.splitrule[0]] >= self.splitrule[1]  
		data_left = self.data[Z == 1, :]
		labels_left = self.labels[Z == 1]
		data_right = self.data[~Z, :] 
		labels_right = self.labels[~Z]
		assert len(labels_left) != 0 and len(labels_right != 0)
		# print "splitting into left:" + str(len(labels_left)) +" and right: " + str(len(labels_right))
		self.left = node(data_left, labels_left)
		self.right = node(data_right, labels_right)
		return
	def makeleaf(self):
		self.label = mode(self.labels)[0][0]
		# print 'made label'
		# print self.label
		# print '\n'
		return 
class Dtree:
	maxdepth = 140
	impurity_threshold = 0.3
	def __init__(self, impurity, segmenter):
		self.root = None
		self.impurity = impurity
		self.segmenter = segmenter 
	def growtree(dt, node, depth):
		#stopping criterion
		if depth >= Dtree.maxdepth:
			# print "got to max depth, making leaf"
			node.makeleaf()
			return
		if node_impurity(label_histogram(node.labels)) < Dtree.impurity_threshold:
			# print "impurity is " + str(node_impurity(label_histogram(node.labels))) + " making leaf"
			node.makeleaf()
			return
		node.splitrule = dt.segmenter(node.data, node.labels, dt.impurity)
		if node.splitrule == None:
			# print "splitrule is none"
			node.makeleaf()
			return
		node.splitnode() #split a node, setting its left and right branches  
		Dtree.growtree(dt, node.left, depth+1)
		Dtree.growtree(dt, node.right, depth+1)
		return

	def train(self, data, labels):
		self.data = data
		self.labels = labels
		self.root = node(data, labels)
		print 'begin growing tree'
		Dtree.growtree(self, self.root, 0)
		return self
  #example should be a 1d array of features
	def make_prediction(self, node, example):
		if (node.label != None):
			print 'classified as ' + str(node.label)
			return node.label
		split_feature = node.splitrule[0]
		split_threshold = node.splitrule[1]

		if (example[split_feature] >= split_threshold):
			print 'split left on ' + str(split_feature) + '>= ' + str(split_threshold)
			return self.make_prediction(node.left, example)
		if (example[split_feature] < split_threshold): 
			print 'split right on ' + str(split_feature) + '< ' + str(split_threshold)
			return self.make_prediction(node.right, example)
		assert False
	def predict(self, test_data):
		predictions = []
		for i in range(len(test_data)):
			result = self.make_prediction(self.root, test_data[i])
			predictions.append(result)
		return predictions


class label_histogram: 
	def __init__(self, labels):
		if len(labels) == 0:
			self.class1 = 0; 
			self.class0 = 0; 
			self.frac0 = 1;
			self.frac1 = 1
		else :
			self.class1 = np.count_nonzero(labels)
			self.class0 = len(labels) - self.class1 
			self.frac0 = self.class0 / float(len(labels))
			self.frac1 = self.class1 / float(len(labels))

class RandomForest:
	def __init__(self, impurity, segmenter, numtrees):
		self.trees = []
		self.numtrees = numtrees 
		for i in range(self.numtrees):
			self.trees.append(Dtree(segmenter, impurity))
	def train(self, data, labels, bagsize):
		self.data = data
		self.labels = labels
		self.bagsize = bagsize
		for i in range(self.numtrees):
			argsamples = np.random.randint(0, high=len(self.data), size=bagsize)
			databag = self.data[argsamples, :]
			labelsbag = self.labels[argsamples]
			self.trees[i].train(databag, labelsbag)
		return self
	def predict(self, test_data):
		allpredictions = []
		for i in range(self.numtrees):
			allpredictions.append(self.trees[i].predict(test_data))
		finalpredictions = mode(allpredictions, axis=0)[0][0]
		return finalpredictions

def impurity(left_label_histogram, right_label_histogram): 
	left = left_label_histogram
	right = right_label_histogram
	left_impurity = node_impurity(left_label_histogram)
	right_impurity = node_impurity(right_label_histogram)
	total_impurity = left_impurity + right_impurity
	return total_impurity
def node_impurity(node_histogram):
	if node_histogram.frac0 == 0.0:
		part0 = 0.0
	else: 
		part0 = node_histogram.frac0 * math.log(node_histogram.frac0, 2)
	if node_histogram.frac1 == 0.0:
		part1 = 0.0
	else:
		part1 = node_histogram.frac1 * math.log(node_histogram.frac1, 2)
	impurity = -(part0 + part1)
	return impurity

def segmenter(data, labels, impurity):
	#split on mean of features for now	
	parent_impurity = node_impurity(label_histogram(labels))
	feature_mean = np.mean(data, axis=0, dtype=np.float64) #calculate mean along each feature
	max_splitrule = None
	max_ig = 0
	for i in range(len(data[0])):
		#set threshold to Vi,
		#update counters 
		#calculate information gain
		#choose argmax information gain
		Z = data[:, i] >= feature_mean[i]
		left_labels = labels[Z == 1]
		right_labels = labels[~Z]
		if len(right_labels) == 0 or len(left_labels) == 0:
			continue
		left_histogram = label_histogram(left_labels)
		right_histogram = label_histogram(right_labels)
		ig = information_gain(parent_impurity, node_impurity(left_histogram), node_impurity(right_histogram), len(left_labels), len(right_labels)) 
		if ig > max_ig:
			max_ig = ig
			max_splitrule = (i, feature_mean[i])
	# print "splitrule is " + str(max_splitrule) + " with information gain: " + str(max_ig)
	return max_splitrule

def forest_segmenter(data, labels, impurity):
	#split on mean of features for now	
	parent_impurity = node_impurity(label_histogram(labels))
	feature_mean = np.mean(data, axis=0, dtype=np.float64) #calculate mean along each feature
	max_splitrule = None
	max_ig = 0
	allowedsplits = np.random.randint(0, len(data[0]), size=int(np.sqrt(len(data[0]))))
	for i in allowedsplits:
		Z = data[:, i] >= feature_mean[i]
		left_labels = labels[Z == 1]
		right_labels = labels[~Z]
		if len(right_labels) == 0 or len(left_labels) == 0:
			continue
		left_histogram = label_histogram(left_labels)
		right_histogram = label_histogram(right_labels)
		ig = information_gain(parent_impurity, node_impurity(left_histogram), node_impurity(right_histogram), len(left_labels), len(right_labels)) 
		if ig > max_ig:
			max_ig = ig
			max_splitrule = (i, feature_mean[i])
	# print "splitrule is " + str(max_splitrule) + " with information gain: " + str(max_ig)
	return max_splitrule
def information_gain(impurity_parent, impurity_left, impurity_right, size_left, size_right):
	ig = impurity_parent  - (size_left * impurity_left + size_right * impurity_right)/float((size_left + size_right))
	return ig

def main():
	#change file path
	mat = scipy.io.loadmat('spam-dataset/spam_data.mat')
 	data = np.array(mat['training_data'])
 	train_data = data[0:4000,:]
 	validation_data = data[4001:5172, :]
	labels = np.array(mat['training_labels'].flatten())
	train_labels = labels[0:4000]
	validation_labels = labels[4001:5172]
	test_set = np.array(mat['test_data'])
	# decisiontree = Dtree(impurity, segmenter)
	# classifier = decisiontree.train(train_data, train_labels)
	# predictions = classifier.predict(validation_data)

	randforest = RandomForest(forest_segmenter, impurity, 6)
	randforest.train(train_data, train_labels, 3000)
	predictions = randforest.predict(validation_data)
	errors = 0
	for i in range(len(predictions)):
		if predictions[i] != validation_labels[i]:
			errors += 1
	print errors/float(len(validation_labels))


	with open('test.csv', 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, dialect='excel')
		for item in predictions:
			spamwriter.writerow([item])



if __name__ == '__main__':
	main()



