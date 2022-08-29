# Machine-Remaining-Useful-Life-Estimation-Using-LSTNet

Estimating the remaining useful life of machine using LSTNet work proposed by [Lai Guokun](https://github.com/laiguokun/LSTNet)

## Dataset
The dataset used is a part of NCMAPPS dataset and can be downloaded from [here](https://drive.google.com/file/d/190WjT_ojIkw7m4NQRjnUBg2EKvHzNYEu/view?usp=sharing).

## Models :
Three models were trained and tested to predict the RUL.

 + **LSTNet :** the implementation of the LSTNet model can be found in the file 'model.py' it is adopted from [here](https://github.com/gokulkarthik/LSTNet.pytorch/blob/master/LSTNet-For-Cryptocurrency-Market-Prediction.ipynb), to  create and train an LSTNet model use the file 'LSTNet.ipynp'.
 + **Deep Convolutional Neural Networks (DCNN)** : to implement a DCNN model use the file 'DCNN.ipynp', a fined tuned DCNN model whcih gives us the best results-until now- can be downloaded from [here](https://github.com/gokulkarthik/LSTNet.pytorch/blob/master/LSTNet-For-Cryptocurrency-Market-Prediction.ipynb)
 + **Long Short Term Memory Network(LSTM)** : to implement an LSTM model use the file 'LSTM.ipynp'.


**Note:** <br />
All files with extension '.ipynp' are implemented using 'Google Colab Notebook'. <br/>
The file 'PreProcessing.py' contains the necessary functions to download , plot , and pre-process the data.


## Environment :
The neccessary libararies and packages are : <br/>
- h5py==3.1.0
- matplotlib==3.2.2
- numpy==1.21.6
- pandas==1.3.5
- scikit_learn==1.1.0
- seaborn==0.11.2
- torch==1.11.0+cu113

and can be downloaded using the 'requirements.txt' file.
