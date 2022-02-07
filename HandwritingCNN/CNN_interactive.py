import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
import keras.utils
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
from tensorflow.keras import utils as np_utils
from tensorflow.keras import models


'''
Filters the dataset according to which numbers you want to train
'''
def filter_data(num_list, X_train, Y_train, X_test, Y_test):

	train_filter=[]
	test_filter = []

	# Creates filter arrays to keep track of indices
	# of the numbers we want
	for i in range(len(num_list)):
		train_filter.append(np.where((Y_train == int(num_list[i]))))
		test_filter.append(np.where((Y_test == int(num_list[i]))))

	tmp1 = []
	tmp2 = []

	for i in range(len(train_filter)):
		alist = train_filter[i]
		for j in range(len(alist[0])):
			tmp1.append(alist[0][j])

	for i in range(len(test_filter)):
		alist = test_filter[i]
		for j in range(len(alist[0])):
			tmp2.append(alist[0][j])

	train_filter = np.array(tmp1)
	test_filter = np.array(tmp2)

	train_filter = np.sort(train_filter)
	test_filter = np.sort(test_filter)

	X_train, Y_train = X_train[train_filter], Y_train[train_filter]
	X_test, Y_test = X_test[test_filter], Y_test[test_filter]


	return X_train, Y_train, X_test, Y_test


'''
Reshape the image data and put in the correct order
Also normalize the pixel data to fit between 0 and 255 (color values)
'''
def reshape_x(X_train, X_test):

	# The size of the images
	rows, cols = 28, 28

	# Records the information about the images
	# Number of channels and size of the images
	input_x = []
	size = X_train.shape[0]

	# Check to see if the images should be channels first or last
	# Channels first indicate channel, row, col
	# Channels last is reverse
	if k.image_data_format() == 'channels_first':
		X_train = X_train.reshape(X_train.shape[0], 1, rows, cols)
		X_test = X_test.reshape(X_test.shape[0], 1, rows, cols)
		input_x = (1, rows, cols)
	else:
		X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
		X_test = X_test.reshape(X_test.shape[0], rows, cols, 1)
		input_x = (rows, cols, 1)

	# Cast the datatype to float32 for better speed
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')

	# Normalize the image pixel data
	X_train /= 255
	X_test /= 255

	# Returns the reshaped image data arrays and shape of them
	return X_train, X_test, input_x, size

'''
Reshape the label data to mark 1 at the index of the number
[2] -> [0, 0, 1]
There are 10 categories in the dataset, keep it the same
no matter which numbers are being filtered for simplicity
'''
def reshape_y(Y_train, Y_test):
	num_category = 10
	Y_train = np_utils.to_categorical(Y_train, num_category)
	Y_test = np_utils.to_categorical(Y_test, num_category)

	return Y_train, Y_test


'''
Visualize first 16 images in data
'''
def visualize(X_train):

	for i in range(16):
		# define subplot
		plt.subplot(4, 4, i+1)
		# plot raw pixel data
		plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

	# show the figure
	plt.show()


'''
Create the CNN model by adding layers to a sequential model
Sequential models take in one input and produce one output
The number of categories in the data is 10, 0-9 are the numbers
'''
def create_model(input_x):

	num_category = 10
	# Explanation of the layers provided in the paper
	model = models.Sequential()
	model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_x))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_category, activation='softmax'))

	# Return the model to use in later functions
	return model


def print_summary(model):

	model.summary()


'''
Trains and runs the model
Trains using the X_train, Y_train data
Tests it against X_test, Y_test data to
calculate accuracy and validation accuracy
'''
def train_model(model, batch, epoch, X_train, Y_train, X_test, Y_test):

	model.compile(optimizer = 'adam',
				loss = keras.losses.categorical_crossentropy,
				metrics=['accuracy'])
	history = model.fit(X_train, Y_train, batch_size = batch, epochs = epoch, validation_data = (X_test, Y_test))

	test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

	print("Test Accuracy: ", test_acc)
	# Return history to use it for the accuracy plot
	return history


'''
Plot the accuracy and validation accuracy
Default batch size is 32
Default epoch is 10
'''
def plot_acc(history):

	plt.plot(history.history['accuracy'], label = 'accuracy')
	plt.plot(history.history['val_accuracy'], label='val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	plt.show()


'''
Create a probability model to test the
testing data and see if the model works
'''
def create_prob_model(model, X_test):

	probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
	predictions = probability_model.predict(X_test)

	# Return predictions to use in plot
	return predictions


'''
Plots the images that are being tested in the
prediction model
'''
def plot_img(i, predictions_arr, true_label, img):

	true_label = true_label[i].tolist()
	true_label = true_label.index(1)
	img = img[i]
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	plt.imshow(img, cmap = plt.get_cmap('gray'))

	# Check if the predicted label equals the true label
	predicted_label = np.argmax(predictions_arr)
	if predicted_label == true_label:
		color = 'blue'
	else:
		color = 'red'

	plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
									100*np.max(predictions_arr),
									class_names[true_label]),
									color=color)


'''
Plots the prediction probabilities of the images
being tested
'''
def plot_value(i, predictions_arr, true_label):

	true_label = true_label[i].tolist()
	true_label = true_label.index(1)	
	plt.grid(False)
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), predictions_arr, color='#777777')
	plt.ylim([0, 1])
	predicted_label = np.argmax(predictions_arr)

	thisplot[predicted_label].set_color('red')
	thisplot[true_label].set_color('blue')

'''
Plot the predictions from the model
Plots first num_images images and their predicted labels
'''
def plot_predict(num_rows, num_cols, num_images, predictions, Y_test, X_test):

	num_images = num_rows*num_cols
	plt.figure(figsize=(2*2*num_cols, 2*num_rows))

	for i in range(num_images):
		plt.subplot(num_rows, 2*num_cols, 2*i+1)
		plot_img(i, predictions[i], Y_test, X_test)
		plt.subplot(num_rows, 2*num_cols, 2*i+2)
		plot_value(i, predictions[i], Y_test)

	plt.tight_layout()
	plt.show()


def main():

	# Load the data from Keras
	# X contains the image data
	# Y contains the label data
	print("Loading data...")
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
	X_train_orig = X_train
	Y_train_orig = Y_train
	X_test_orig = X_test
	Y_test_orig = Y_test

	# Prints info about the program
	print("Welcome to our CNN model!")
	dialogue = ["Filter the data (optional)", "Visualize the data (optional)",
				"Reshape the image data and label data", "Create the model",
				"Print the model summary (optional)", "Train the model",
				"Plot the model accuracy (optional)",
				"Create a probability model to test the CNN (optional)",
				"Plot the results of the probability model (optional)"]

	print("Here is a list of the steps in this program:")
	for i in range(len(dialogue)):
		print("%d. %s" % (i+1, dialogue[i]))
	print("Type q to quit.")

	# Dialogue to use in the while loop
	dialogue2 = ["Filter the data (optional)?", "Visualize the data (optional)?",
				"Reshaping the image data and label data", "Creating the model",
				"Print the model summary (optional)?", "Training the model",
				"Plot the model accuracy (optional)?",
				"Create a probability model to test the CNN (optional)?",
				"Plot the results of the probability model (optional)?"]


	# Initialize outside while loop for scope issues within if statements
	predictions = ""
	model = ""
	history = ""
	input_x = ""
	ifPredict = False

	track = 0
	inp = 0

	# i keeps track of which step the program is in
	i = 0
	while(inp != 'q'):
		if 'optional' in dialogue2[i]:
			print(dialogue2[i])
			inp = input("y/n: ")
			inp = inp.lower()

			if inp == 'q':
				break

			if inp != 'y' and inp != "n":
				while inp != "y" and inp != "n":
					print("Invalid input")
					inp = str(input("Please type y or n: "))
					inp = inp.lower()

			if inp == 'y':
				if i == 0:
					print("Which numbers would you like to keep?")
					inp = str(input("Type any numbers 0-9 using commas to separate: "))
					numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
					num_list = inp.replace(" ", "")
					num_list = num_list.split(",")

					check = True
					for item in num_list:
						if item not in numbers:
							check = False

					if check == True:
						all_data = filter_data(num_list, X_train, Y_train, X_test, Y_test)
						X_train = all_data[0]
						Y_train = all_data[1]
						X_test = all_data[2]
						Y_test = all_data[3]
					else:
						print("Incorrect input. Skipping filtering.")

				elif i == 1:
					print("Visualizing the data...")
					visualize(X_train)

				elif i == 4:
					print("Printing model summary...")
					print_summary(model)

				elif i == 6:
					print("Plotting accuracy and validation accuracy...")
					plot_acc(history)

				elif i == 7:
					print("Creating probability model...")
					predictions = create_prob_model(model, X_test)
					ifPredict = True

				elif i == 8 and ifPredict == True:
					print("Plotting the first 16 results of the probability model...")
					plot_predict(4, 4, 16, predictions, Y_test, X_test)

				elif i == 8 and ifPredict == False:
					print("Cannot plot the results of the prediction model because it wasn't created.")
					print("Continuing program and skipping plot.")


		else:
			print(dialogue2[i])
			if i == 2:
				all_data = reshape_x(X_train, X_test)
				X_train = all_data[0]
				X_test = all_data[1]
				input_x = all_data[2]
				size = all_data[3]
				all_data = reshape_y(Y_train, Y_test)
				Y_train = all_data[0]
				Y_test = all_data[1]

			elif i == 3:
				model = create_model(input_x)

			elif i == 5:
				batch = 32
				epoch = 5
				print("Do you want to specify the batch and epoch?")
				inp = input("y/n: ")
				if inp == 'y':
					print("The batch size must be more than 1 and less than ", size)
					batch = (input("Batch: "))
					print("The epoch number must be 1 or greater.")
					epoch = (input("Epoch: "))
					if batch.isdigit() == False:
						batch = 32
						print("Invalid input. Batch will be 32.")
					else:
						batch = int(batch)

					if epoch.isdigit() == False:
						epoch = 5
						print("Invalid input. Epoch will be 5.")
					else:
						epoch = int(epoch)

					history = train_model(model, batch, epoch, X_train, Y_train, X_test, Y_test)
				else:
					print("Invalid input, or you chose n. Using batch size = 32 and epoch = 5.")
					history = train_model(model, batch, epoch, X_train, Y_train, X_test, Y_test)

		i += 1

		if i == 9:
			print("Would you like to run through the program again?")
			inp = input("y/n: ")
			inp = inp.lower()
			if inp == 'y':
				i = 0
				predictions = ""
				history = ""
				model = ""
				ifPredict = False
				X_train = X_train_orig
				Y_train = Y_train_orig
				X_test = X_test_orig
				Y_test = Y_test_orig
			elif inp == 'n':
				inp = 'q'
			else:
				print("Invalid input. Quitting.")
				break

main()


