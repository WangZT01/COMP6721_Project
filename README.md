# COMP6721_Project
# Zitao Wang(40171434) | Jaming Han(40185367) | Leyu Dai(40199072)
# Environment Version: 
-python 3.6.13
# Machine Learning Farmeworks: 
-Pytorch 1.3.1, sklearn 0.24.2, skorch 0.11.0
# files included
- In CNN folder, there are three files. CNN.py is the CNN model we established. And it is also the main funtion of our code. The CNN_Structure.png shows the sual CNN structure of our model. The CNN_Skorch is a sklearn compatible neural network model that warps PyTorch.
- In data folder, it contains five different types of pictures: Cloth Mask(432)/ N95 Mask with Valve(436)/ N95 Mask(416)/ No Face Mask(549)/ Surgical Mask(409)
- DataNamed.py is using to processing the data. We use OS, re PIL.Image and python pandas to process our data from data folder and import to our model.
- DataSet.png shows the Histogram of data.

# How to run
- Open the whole project, run the CNN.py under CNN folder. Then it will show the results. Due to the the number of epochs is 10. So it may needs 10 to 15 mins to present the result.

# The results shows the accuracy for our dataset, one accuracy each. And the duration for our model.

- Classes    | Cloth Mask	| N95 Mask	| N95 With Valve | No Face Mask | Surgical Mask
- Accuracy	 | 54.6875%	  | 52.5625%	|   54.6875%	   |  54.6875%	  |   55.3571%
- Precision	 | 22.7612%	  | 21.1397%	|   21.2092%	   |  28.1572%	  |   21.4836%
- Recall	   | 56.4815%	  | 52.7523%	|   53.125%	     |  548270%	    |   54.5233%
- F1-measure | 32.4468%   |	30.1837%	|   30.3155%	   |  37.2064%	  |   30.8224%
The Confusion Matrix can find on our report due to the format reason of Readme. It is hard to present clearly.

                  
