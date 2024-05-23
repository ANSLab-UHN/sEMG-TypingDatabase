#Root Script to run the Classficiation algorithms from.
#Author: Jonathan Eby
#Date: Apr 25, 2024

from Classify import *

#Relative path to the participant folders
folder_path = './../CleanData/P'
#20 participants were originally recorded. Note participant 11 is excluded as the data for participant 11 was corrupted.
participant_list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
#Two test days for each participant
desired_tests = ['1','2']
#Best features for a 0.2s window determined from a feature analysis
features = 
best_features = ['RMS', 'LOGVAR','WL','WAMP', 'AR1', 'AR2']
#Window length to be extracted for each keypress
winlength = 0.2
#Options: TrainTest, Kfold, CrossTest, CrossParticipant
ClassifyType = 'Kfold'
#If the results folder does not yet exist create one
if not os.path.exists("./../ClassificationResults/"):
    os.mkdir("./../ClassificationResults/")
    write_path = "./../ClassificationResults/"
else:
    print("Path exists")
    write_path = "./../ClassificationResults/"
#Begin Classification
Classify(folder_path,participant_list,desired_tests,best_features,winlength,ClassifyType)