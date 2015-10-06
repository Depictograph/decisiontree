Decision Tree can be used for classification with the following interface: 
decisiontree = Dtree(segmenter, impurity)
decisiontree.train(train_data, train_labels)
predictions = decisiontree.predict(test_data) 

randforest = RandomForest(forest_segmenter, impurity, numtrees)
randforest.train(train_data, train_labels, bag size)
predictions = randforest.predict(validation_data)

Currently there is no data preprocessing. 
The main function in decisiontree.py can be modified to run Decision tree or Random Forest as required. 

Predictions are written to a csv file, test.csv