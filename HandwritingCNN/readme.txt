Files
CNN.py
The non-interactive version of the convolutional neural network. 
This program loads the MNIST dataset and creates a CNN. 
It then trains the CNN using the dataset and creates a probability 
model to test the CNN model. It also graphs the accuracy of the CNN
and plots some images from the output of the prediction model.
This has the batch size set to 32 and the epoch size set to 5.
You can try changing 5 to 10 if your computer has enough memory. 
Otherwise it might seg fault with 10. 

CNN_interactive.py
The interactive version of the convolution neural network. 
This program allows the user to choose which numbers to filter out and whether
to plot the various graphs. 
There are instructions within the program that will tell you what to do once you 
run it. 


Running the program
Check to make sure all libraries have been installed in the latest version.
This is especially important for Tensorflow. Tensorflow2 is required for Keras.
The important libraries are:
Numpy
Tensorflow
Keras
Matplotlib

Also check to make sure at least version 3.9 of Python is installed.

Run the programs using python or python3 depending on your machine.

python3 CNN.py

python3 CNN_interactive.py


Possible Errors/Things that might go wrong
If the program crashes/says segmentation fault, you are probably training 
for too long or with data sets that are too large. Either reduce the epoch size 
and/or increase the batch size. Alternatively, filter out some of the numbers.

If a graph pops up, make sure you click back on the terminal to continue the 
program in the interactive version. 