# COMP6721_Project
# Zitao Wang(40171434) | Jaming Han(40185367) | Leyu Dai(40199072)
# Environment Version: 
-python 3.6.13
# Machine Learning Farmeworks: 
-Pytorch 1.3.1, sklearn 0.24.2, skorch 0.11.0
# files included
- In CNN folder, there are three files. (1. CNN.py is the CNN model we established. And it is also the main funtion of our code. (2.The CNN_Structure.png shows the sual CNN structure of our model. (3.The CNN_Skorch is a sklearn compatible neural network model that warps PyTorch.
- In data folder, it contains five different types of pictures: Cloth Mask(432)/ N95 Mask with Valve(436)/ N95 Mask(416)/ No Face Mask(549)/ Surgical Mask(409)
- DataNamed.py is using to processing the data. We use OS, re PIL.Image and python pandas to process our data from data folder and import to our model.
- DataSet.png shows the Histogram of data.

# How to run
- Open the whole project, run the CNN.py under CNN folder. Then it will show the results. Due to the the number of epochs is 10. So it may needs 10 to 15 mins to present the result.

# The results shows the accuracy for our dataset, one accuracy each. And the duration for our model.

- Classes    | Cloth Mask	| N95 Mask	| N95 With Valve | No Face Mask | Surgical Mask
- Accuracy	 | 64.84374%  | 64.4531%	|   61.71875%    |  63.08593%   |   63.03571%
- Precision	 | 29.97812%  | 30.4017%	|   26.0541%	   |  35.6701%	  |   27.5161%
- Recall	   | 64.77541%  | 64.2202%	|   57.9326%     |  63.0237%	  |   62.8362%
- F1-measure | 40.98729%  |	41.2675%	|   35.9433%	   |  45.5563%	  |   38.2725%
- The Confusion Matrix can find on our report due to the format reason of Readme. It is hard to present clearly.

                  
