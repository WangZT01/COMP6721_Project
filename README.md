# COMP6721_Project_part 2
# Zitao Wang(40171434) | Jaming Han(40185367) | Leyu Dai(40199072)
# Environment Version: 
-python 3.6.13
# Machine Learning Farmeworks: 
-Pytorch 1.3.1, sklearn 0.24.2, skorch 0.11.0 matplotlib 3.3.1
# major files included
- In CNN folder, we run CNN.py then get the a CNN MODEL. This Model stored in CNN_Model.pt. Then we use the Data from Data folder to train the model by using demo.py. 
- Data folder is the original dataset. It is not adjust any data on it(no adding or deleting). 
- We trained male first then female. This is in CNN_Gender. After Trained on gender, we use the sameway on the Race. 
- After the training, we found the errors of results. So then we adjusted the data, thus, we have DATA_Balanced folder. 
- Using DATA_Balanced folder to run our model, we have the CNN_MODEL_Update.pt. 
- Get CNN_Model_Update, we put it on the demo, we have the new Confusion Matrix. 
- In CNN_Part2_Kfold, we have the CNN_KFOLD.py as the model. Using the Demo_KFOLD.py, we can have 10 model(CNN_Model_Kfold0.pt to CNN_Model_Kfold9.pt). 
- In the same method as mentioned above, we using balanced data, then have the new ten model(CNN_Update_Kfold0.pt to CNN_Update_Kfold9.pt).
- 
# How to run
- Open the whole project, run the CNN.py under CNN folder. Then it will show the results. Demo.py will give the cofusion matrix. 
- Open CNN_Part2_Kfold folder, run the CNN_KFOLD.py. Then it will show the results .Demo_KFOLD.py will give the cofusion matrix. 
