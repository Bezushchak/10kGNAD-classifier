# 10kGNAD-classifier
Once you have python installed, you get access to the pip command.
You should run ```pip install pipenv``` to be able to install all the dependancies on a virtual environment.
Now you want to run ```pipenv shell```. That will create a virtuall environment and a Pipfile which will include any packages that you install. Now you can use pipenv to install the requirements by the command ```pipenv install -r requirementx.txt```.
Now you are good to proceed with the app.
Make sure you clone into the corresponding repo and change your directory.
In the terminal execute the following command ```streamlit run german.py```, which runs the app in the localhost webpage in your default browser (google chrome recommended). To use the app in the different browser - copy the output of the command in the terminal in a form of a host:port (192.168.0.120:8501) and paste it in the search string.

Inside the app you are given the opportunities to 'Show 10kGNAD Data Chart', which shows a brief summary of the dataset in a form of a bar chart.
Next you may choose the model to work with and paste your text in the Text Area.
Submit button provides you with the comprehensive results of the classifier workflow as well as the probabilities and visualizations.
'Show model report' may provide you with the table of metrics that estimate each model performance.
