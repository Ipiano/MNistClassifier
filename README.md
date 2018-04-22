# MNIST Digit Classifier in Lua using Torch
#### For: CSC 416 - Advanced Algorithms in Robotics
#### By: Andrew Stelter
#### Submitted To: Dr. Larry Pyeatt
#### Date: 4/22/18

## Submission Contents
* Lua Source code for a CNN capable of identifying digits with at least 99.37% accuracy
* A research paper compliant with the NIPS 2017 guidelines
* All testing/training data used in the paper
* Various scripts used to parse logs and generate graphs

## Source Code
All source code for the project is contained in the directory ```src/```. The CNN is built from a number of separate ```.lua``` source files; the main entrypoint is ```main.lua```.
### Dependencies
The project requires having the Torch project installed on your system. Details about installing Torch can be found at this link: http://torch.ch/docs/getting-started.html#_.

Once Torch is installed, a couple of specific LuaRocks packages are required:
* ```struct```
* ```lfs``` - Should be installed with Lua
* ```cunn``` - Only required if you want to use CUDA
* ```graphicsmagick``` - Only required if you want to output images of the test set. This requires having the GraphicsMagick software installed and in your path.

The last dependency of the project is the MNIST Dataset. It can be found at http://yann.lecun.com/exdb/mnist/. All of the dataset files should be placed the the directory ```data/``` relative to the project root.

#### The Makefile
Included with the project is a standard GNU Makefile. Because Lua is an interpreted or JIT Compiled language, the Makefile doesn't actually operate on the source. The Makefile has two purposes:
* Download and Unzip the MNIST Data
* Install the Lua dependencies other than ```lfs```

If you run the makefile, it will attempt to install both ```cunn``` and ```graphicsmagick```. If you do not need or want either of these, use LuaRocks to install whichever packages you do want.
Installing just the data files or just the dependencies can be achieved with ```make data``` and ```make installdeps``` respectively.

### Running the Project
Once all dependencies are installed, the project can be run directly from the command line with

```
th ./src/main.lua [args]
```
(Where ```th``` is the command line invocation of the torch REPL)

Running the main file will load the data set (Training and testing), build the Neural Net, train the Neural Net, and save the final weights of the Neural Net. The output folder of the final weights will be a subdirectory of ```logs/```. This subdirectory is named automatically dependent on the command line arguments specified.
#### Command Line Arguments
* ```--graph```: Runs the testing data after each training epoch. Each training epoch and testing result is outputted to ```train.log``` and ```test.log``` in the same log folder where the final weights will be saved. These files are formatted such that they can be plotted easily with gnuplot.
* ```--profile```: Collects timing statistics for different sections of the training process and prints them to stdout for each epoch.
* ```--cuda```: Runs training and testing on a CUDA-enabled GPU. Requires the ```cunn``` lua package.
* ```--minibatch [n]```: Trains randomized minibatches of size n instead of training the entire set at once. May be required if using ```--cuda``` and the entire training set cannot fit in VRAM. (Only n training images are loaded into VRAM at once). n is default 10.
* ```--batchnorm```: Adds Batch Normalization steps after each layer of the Neural Net. Requires the ```--minibatch``` argument.
* ```--epochs n```: Specifies the number of epochs to train. n is 25 if this argument is omitted.
* ```--learning_multiplier [m [e]]```: Multiplies the learning rate by m every e epochs. Defaults are 0.3 and 100, respectively.
* ```--data_size n```: Limits the training set to be at most n images. Does not modify the testing set. n is 2^32 if this argument is omitted.
* ```--transforms n```: Creates n random affine transformations of each training set image and adds these transformed images to the training set.
* ```--deforms n [sigma [alpha]]```: Creates n random elastic deformations of each training set image and adds these deformed images to the training set. Sigma is the standard deviation of the convolution Gaussian, and Alpha is a constant scaling factor applied to the deformation field.
* ```--make_images [n]```: Will output the first n images of the training set, modifications to the training set, and the testing set in the directory ```images/```. n is default 100. Requires GraphicsMagick and the ```graphicsmagick``` lua package.

## Paper
The research paper and all extra images (other than graphs) associated with this project can be found in the directory ```paper/```. The paper is named ```nips_2017```, the default name of the 2017 NIPS LaTeX file. A pre-built copy of the file has been included in the repository for convenience. Building the paper requires
* The TexLive Distribution
* Inkscape installed (to auto-convert .svg files)

The paper should be built by running ```pdflatex```, followed by ```bibtex``` followed by ```pdflatex``` twice. The following arguments are required for ```pdflatex```:
```
-recorder -shell-escape
```

See the Data Processing section below for information about how to generate the ```.eps``` graph files used in the paper.

## Data Collected
All data collected for the paper is included in the repository in the ```logs/``` subdirectory. Running the Neural Net with the same command line arguments used to generate these folders will overwrite them.

The script ```iterate.bash``` was used to run the Neural Net and iterate over different command line arguments to generate these subdirectories. ```iterate.bash``` contains three functions for iterating over different parts of the arguments:
* ```itermini()```: Tests from minibatch size 100 up to 1000, incrementing by 100. For each size, the base training set is tested, along with training sets augmented by transformations, deformations, and both transformations and deformations.
* ```iterdeform()```: Tests various sigma and alpha values for deformations. The double loop tests sigma from 0.001 to 10 incrementing by 0.5 and alpha from 5 to 50 incrementing by 5.
* ```iterlearn()```: Tests different learning rate multipliers. The double loop tests m from 0.1 to 1 incrementing by 0.1 and e from 25 to 150 incrementing by 25.

Both ```iterdeform()``` and ```iterlearn()``` require the command line tool ```bc``` for floating point operations. 

Any command line arguments passed to the script will be sent through to the Neural Net code; however, it is not recommended to use arguments that will be modified by the script. (```--minibatch```, ```--deforms```, ```--transforms```, ```--learning_multiplier```, and ```--batchnorm```)
The script runs each of the three functions and H.then adds the ```--batchnorm``` argument before running them all again.

Due to time and hardware constraints, only ```itermini()``` was run, the other two functions are currently commented out.
### Data Processing
A number of bash and gnuplot scripts are included with the project to help process data files. Most of them are sub-scripts called by the main scripts. The two main scripts are:
* ```get_scores.bash```
* ```make_all_graphs.bash```

The first script will simply read all of the logs in ```logs/``` and, for each subdirectory, output the maximum testing score and maximum training score. At the end, it outputs which folders contained the overall highest and lowest testing scores.

The second script calls the ```graph-minibatches.gplt``` and ```graph-single-size.gplt``` gnuplot scripts to generate all of the graphs required by the paper. This, obviously, requires having gnuplot in your PATH.