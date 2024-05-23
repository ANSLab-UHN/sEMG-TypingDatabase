# Functions to perform classification on the provided dataset. Main classification function is called classify.
#Author: Jonathan Eby
#Date: Apr 25, 2024

import pandas as pd
import os
import datetime
import numpy as np
from load_intan_rhd_format import *
import csv
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import shutil
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Feature extraction functions:
def featextract(data,ModelFeatures):
    feat = {}
    threshWAMP = 6
    threshZC = 10
    threshSSC = 8
    # RMS
    if "RMS" in ModelFeatures:
        rms = np.sqrt(np.mean(np.power(data, 2)))
        feat["RMS"] = rms
    if "MAV" in ModelFeatures:
        mav = np.mean(abs(data))
        feat["MAV"] = mav
    # WL
    if "WL" in ModelFeatures:
        i = 0
        while i < (len(data) - 1):
            if i == 0:
                WL = np.abs(data[i+1] - data[i])
            else:
                WL = WL + np.abs(data[i+1] - data[i])
            i += 1
        feat["WL"] = WL
    # LOGVAR
    if "LOGVAR" in ModelFeatures:
        LogVar = np.log(np.var(data))
        feat["LOGVAR"] = LogVar
    #AR
    if "AR1" in ModelFeatures:
        ar_model = AutoReg(data,1).fit()
        AR = ar_model.params
        feat["AR1"] = AR[1]
    if "AR2" in ModelFeatures:
        ar_model = AutoReg(data,2).fit()
        AR = ar_model.params
        feat["AR2"] = AR[2]
    if "AR3" in ModelFeatures:
        ar_model = AutoReg(data,3).fit()
        AR = ar_model.params
        feat["AR3"] = AR[3]
    if "AR4" in ModelFeatures:
        ar_model = AutoReg(data,4).fit()
        AR = ar_model.params
        feat["AR4"] = AR[4]
    #WAMP
    if "WAMP" in ModelFeatures:
        i = 0
        WAMP = 0
        while i < len(data) - 1:
            if np.abs(data[i] - data[i+1]) > threshWAMP: #Where 0.00005 is in V for the threshold (should be between 50 uV and 100 mV)
                WAMP = WAMP + 1
            i += 1
        feat["WAMP"] = WAMP
    #SSC
    if "SSC" in ModelFeatures:
        SSC = 0
        for i in range(1,len(data) - 1):
            if (np.sign(data[i] - data[i-1])*np.sign(data[i] - data[i+1]) == 1)  and (abs(data[i] - data[i-1])>threshSSC or abs(data[i] - data[i+1])>threshSSC):
                SSC += 1
        feat["SSC"] = SSC
    #ZC
    if "ZC" in ModelFeatures:
        ZC = 0
        zero_crossings = np.where(np.diff(np.signbit(data)))[0]
        for i in zero_crossings:
            if np.abs(data[i] - data[i+1]) > threshZC:
                ZC += 1
        feat["ZC"] = ZC
    #VAR
    if "VAR" in ModelFeatures:
        var = np.sum(np.power(data, 2))/(len(data)-1)
        feat["VAR"] = var
    #IEMG
    if "IEMG" in ModelFeatures:
        iemg = np.sum(np.abs(data))
        feat["IEMG"] = iemg
    #MAVS
    if "MAVS" in ModelFeatures:
        k=3
        curindx = 0
        segment_length = np.floor(len(data)/k)
        mavs = np.zeros(k-1)
        for i in mavs:
            i = np.mean(abs(data[int(curindx+segment_length):int(curindx+2*segment_length)])) - np.mean(abs(data[int(curindx):int(curindx+segment_length)]))
            curindx += segment_length
        feat["MAVS"] = np.mean(mavs)
    return feat

#Reads the keylogs.txt file into a list
def read_file_to_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip().startswith('2022')]
        data = []
        #Pull time portion from the total keylog line
        for item in lines:
            parts = item.split(' - ')
            time_part = parts[0].split()[1]
            letter = parts[1].strip("'")
            data.append([time_part, letter])
        df = pd.DataFrame(data, columns=['Time', 'Letter'])
        df['Time'] = pd.to_datetime(df['Time'],format='%H:%M:%S,%f')
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return []

# Function to plot a confusion matrix for TrainTest or Kfold classification algorithms. Returns the confusion matrix.
def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
    matrix = metrics.confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    lab = np.array(list(range(len(sorted_labels))))
    lab = lab[~(matrix == 0).all(1).astype(bool)]
    matrix = matrix[:,lab]
    matrix = matrix[lab,:]
    maxval = matrix.sum(axis=1).max()
    #matrix = matrix*100/maxval
    confmat = plt.figure(figsize=(6.5, 4))
    confmat.set_tight_layout('tight')
    sns.heatmap(matrix, annot=True, xticklabels=lab,yticklabels=lab, cmap="viridis", fmt=".3g",cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix')
    return confmat

## ============================== MAIN CLASSIFICATIO ALGORITHM ============================== ##
# Inputs:
    # folder_path = path to Participant folders
    # participant_list = list of participants (type str)
    # desired_tests = list of desired test days (type str)
    # features = a list of strings with short form of the features from featextract function
    # winlength = float to describe the desired window length (s) per keypress
    # ClassifyType = String describing desired classification to be performed. Options: TrainTest, Kfold, CrossTest, CrossParticipant
def Classify(folder_path,participant_list,desired_tests,features,winlength,ClassifyType):
    #Initial parameters for execution
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    err_dataframe = pd.DataFrame({'Participant':[],'Test':[],'Filename':[],'Error':[],'Warning':[], 'Used':[]})
    chanels = 16
    trial_length = 15
    maxthresh = 500
    minkeynum = 5
    c = 0
    ModelFeatures = features
    #Creation of column names for the structure to hold data
    featurenames = ['Participant','Test','Label']
    for i in range(chanels):
        for j in ModelFeatures:
            featurenames.append(str(i)+j)
    Even_Data = pd.DataFrame(columns=featurenames)
    #Cycle through each participant and test in the provided lists
    for p in participant_list:
        for t in desired_tests:
            #Create new dataframes to store this sessions data
            Feature_Data = pd.DataFrame(columns=featurenames)
            keys = ['Filename','Lag','Error']
            lag_dataframe = pd.DataFrame(columns=keys)
            path = folder_path+p+'/T'+t+'/Data/'
            keylog = read_file_to_list(folder_path+p+'/T'+t+'/Keylogs.txt')
            lag_timings = pd.read_csv(folder_path+p+'/T'+t+'/LAG_TIMINGS.csv')
            for letter in alphabet:
                #for each letter in the alphabet find the associated files in the data directory
                all_files = os.listdir(path)
                letter_files = [file for file in all_files if file.startswith(letter)]
                num_files = len(letter_files)
                for f in letter_files:
                    f = f.split('.')[0]
                    print("P"+p)
                    print("T"+t)
                    print(f)
                    BAD_FILE = False

                    # Find where in keylog the trial is and create timings for INTAN file accordingly
                    start_time = f[9:11]+":"+f[11:13]+':'+f[13:15]
                    kelog_sorted = keylog.sort_values(by='Time')
                    # Convert given time to datetime format
                    start_time = pd.to_datetime(start_time,format='%H:%M:%S')
                    # Find the nearest time after the file start time
                    firstkeystroke = kelog_sorted[kelog_sorted['Time'] >= start_time]['Time'].iloc[0]
                    firstkeystroke_index = keylog[keylog['Time'] == firstkeystroke].index[0]
                    
                    in_trial = False
                    first_key_time = keylog.loc[firstkeystroke_index]['Time']
                    # Go to closest keylog time to start and look for the next spacebar press to indicate the start of a trial.
                    # If no space press is found label as a bad trial
                    while not in_trial:
                        cur_val = keylog.loc[firstkeystroke_index]['Letter']
                        cur_time = keylog.loc[firstkeystroke_index]['Time']

                        if cur_val == 'Key.space':
                            in_trial = True
                        else:
                            firstkeystroke_index += 1
                            if cur_time > (first_key_time+ datetime.timedelta(0,1)):
                                BAD_FILE = True
                                err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Error':'Could not find start of trial'})])
                    
                    #Finding key timings in the key log:
                    perfect_keylog = 0
                    keylog_err_code = 0
                    keylog_clear = 0
                    keystroke_index = firstkeystroke_index
                    excess = 0
                    i = 0;
                    trial_timings_absolute = pd.DataFrame()
                    #Check that the right sequence of keylogs is present after the initial time.
                    while (BAD_FILE == False and keylog_clear == 0):
                        trial_window = keylog.loc[firstkeystroke_index:firstkeystroke_index + trial_length-1+i]
                        window_letters = np.array(trial_window['Letter'])
                        unique_items, counts = np.unique(window_letters, return_counts=True)
                        #If there is no letter found in the next 15 keypresses its a bad file and discarded.
                        if letter.lower() not in unique_items:
                            BAD_FILE = True
                            err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Error':'No correct keypress found at that time'})])
                        #Otherwise count the number of spaces and keypresses present in the current window.
                        else:
                            space_count = counts[np.where(unique_items == 'Key.space')[0][0]]
                            key_count = counts[np.where(unique_items == letter.lower())[0][0]]
                            wrong_keys = unique_items[(unique_items != letter.lower()) & (unique_items != 'Key.space')]
                            keylog_err_code = 2
                            #If it is 5 and 10 space and key presses respectivly the keylog is perfect and the timings are extracted
                            if (i == 0 and space_count == 5 and key_count == 10):
                                perfect_keylog = 1
                                keylog_clear = 1
                                trial_timings_absolute = trial_window[np.logical_or(trial_window['Letter'] == letter.lower(), trial_window['Letter'] == 'Key.space')]
                            #If we have 10 keypresses then we are still satisfied despite the keylog not being perfect. This is noted in the error file.
                            elif key_count == 10:
                                keylog_clear = 1
                                wrong_keys = unique_items[(unique_items != letter.lower()) & (unique_items != 'Key.space')]
                                trial_timings_absolute = trial_window[np.logical_or(trial_window['Letter'] == letter.lower(), trial_window['Letter'] == 'Key.space')]
                                err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Warning':'Non perfect keylog'})])
                            #If we have not found 10 keypresses look at the next element and check if the next element is the desired letter
                            else:
                                if key_count < 10:
                                    next_key = keylog.loc[firstkeystroke_index + trial_length-1+i]['Letter']
                                    if next_key == letter.lower():
                                        i+=1
                                    else:
                                        #If it is not the desired lettter as long as we have more than the minkeynum (8) correct keypresses the file is kept. Otherwise it is rejected.
                                        if key_count >= minkeynum:
                                            keylog_clear = 1
                                            wrong_keys = unique_items[(unique_items != letter.lower()) & (unique_items != 'Key.space')]
                                            trial_timings_absolute = trial_window[np.logical_or(trial_window['Letter'] == letter.lower(), trial_window['Letter'] == 'Key.space')]
                                            err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Warning':'Less than 10 keypresses found'})])
                                        else:
                                            BAD_FILE = True
                                            err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Error':'Less than minimum acceptable keypresses found'})])
                    
                    #If the current file corresponds to a good keylog sequence extract the timings and adjust them to be relative to the timing for the start of the file. Then adjust for the lag from LAG_TIMINGS.txt
                    if not BAD_FILE:
                        trial_timings_relative = (trial_timings_absolute['Time']-start_time)
                        trial_timings_relative = [round(time.total_seconds(),4) for time in trial_timings_relative]
                        key_timings_adjusted = trial_timings_relative + np.full(len(trial_timings_relative), lag_timings.loc[lag_timings['Filename']==f,'Lag'])
                        #Try to read associated intan file. If not possible note this into the error file.
                        try:
                            datafile = read_data(path+f+".rhd")
                            samplingrate = datafile['frequency_parameters']['amplifier_sample_rate']
                            data_timestamps = datafile["t_amplifier"]
                            data = datafile["amplifier_data"]
                        except:
                            err_dataframe = pd.concat([err_dataframe,pd.DataFrame({'Participant':[p],'Test':[t],'Filename':[f],'Error':'Incomplete RHD file'})])
                            BAD_FILE = True
                        
                        if not BAD_FILE:
                            #Initialize labels and data storage
                            P_data = np.empty((len(key_timings_adjusted),chanels,int(winlength*samplingrate)))
                            P_lables = np.array(trial_timings_absolute['Letter'])

                            instance_count = 0
                            #For each time associated with a keypress extract a window centered at that time with length = winlength
                            for time in key_timings_adjusted:
                                if np.isin(round(float(time)-winlength/2,4),data_timestamps):
                                    if np.isin(round(float(time)+winlength/2,4),data_timestamps):
                                        key_data = data[:,(np.where(data_timestamps == round(float(time)-winlength/2,4))[0][0]):(np.where(data_timestamps == round(float(time)+winlength/2,4))[0][0])] 
                                    else:
                                        c+=1
                                        end_lastindex = len(data_timestamps)
                                        start_lastindex = int(end_lastindex - winlength*samplingrate)
                                        key_data = data[:,start_lastindex:end_lastindex]
                                else:
                                    start_firstindex = 0
                                    end_first_index = int(winlength*samplingrate)
                                    key_data = data[:,start_firstindex:end_first_index]
                                #Extract the features associated with the keypress
                                FeatVect = np.zeros(chanels*len(ModelFeatures))
                                chancount = 0
                                for i in range(key_data.shape[0]):
                                    feat = featextract(key_data[i,:],ModelFeatures)
                                    featcount = 0
                                    for j in ModelFeatures:
                                        FeatVect[len(ModelFeatures)*chancount + featcount] = feat[j]
                                        featcount += 1
                                    chancount += 1
                                #Save the features into a feature data array that contains all the features and labels for the corresponding file
                                Feature_Data.loc[len(Feature_Data['Label'])] = ([p,t,P_lables[instance_count]] + FeatVect.tolist())
                                instance_count+=1
            #Check for imbalence in the dataset. Since all keypresses should have 20 presses per file we expect to have 20 elements for each class. Execpt spacebar which has 5 presses in each recording.
            #Any class with more than 20 presse (spacebar) is randomly downsampled to the expected number of samples in this case 20.
            labels = np.array(Feature_Data['Label'])
            data = np.array(Feature_Data.iloc[:, 3:])
            unique_labels, counts = np.unique(labels, return_counts=True)
            expected_number = 20
            balenced_data = []
            balenced_labels = []
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                if len(indices) > expected_number + 0.25*expected_number:
                    sampled_indices = np.random.choice(indices, expected_number, replace=False)
                    balenced_data.extend(data[sampled_indices])
                    balenced_labels.extend(labels[sampled_indices])
                else:
                    balenced_data.extend(data[indices])
                    balenced_labels.extend(labels[indices])
            
        
            balenced_array = np.concatenate((np.full((1,len(balenced_labels)),p),np.full((1,len(balenced_labels)),t),np.array(balenced_labels).reshape(1,-1),np.transpose(np.array(balenced_data))),axis=0)
            Balenced_section = pd.DataFrame(np.transpose(balenced_array), columns=featurenames)
            Even_Data = pd.concat([Even_Data,Balenced_section],ignore_index=True)

    #Classification
    if ClassifyType == 'TrainTest' or ClassifyType == 'Kfold' or ClassifyType == 'CrossTest':
        for p in participant_list:
            if ClassifyType == 'TrainTest' or ClassifyType == 'Kfold':
                for t in desired_tests:
                    #For each participant and session the associated labels and data are extracted from the overall dataframe.
                    labels = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == t)]['Label'])
                    data = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == t)].iloc[:, 3:])
                    #Model is initialized
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data)
                    SVM = svm.SVC(kernel='rbf')

                    if ClassifyType == 'TrainTest':
                        #Declare folder for result to be saved to
                        if not os.path.exists("./../ClassificationResults/TrainTest_classification/"):
                            os.mkdir("./../ClassificationResults/TrainTest_classification/")
                            write_path = "./../ClassificationResults/TrainTest_classification/"
                        else:
                            print("Path exists")
                            write_path = "./../ClassificationResults/TrainTest_classification/"
                        #Initialize the train/test percent
                        testpercent = 0.3
                        X_train,X_test,y_train,y_test = train_test_split(data_scaled,labels,test_size=testpercent,random_state=1,stratify=balenced_labels)
                        count = pd.Series(y_train).value_counts()
                        SVM.fit(X_train,y_train.reshape(-1))

                        y_pred = SVM.predict(X_test)

                        ClassAcc = metrics.accuracy_score(y_test,y_pred)
                        F1 = metrics.f1_score(y_test,y_pred,average='micro')
                        print(ClassAcc)

                        if not os.path.exists(write_path + '/ClassificationResult.csv'):
                            fileout = open(write_path+'/ClassificationResult.csv', mode='w',newline='')
                            csvFile = csv.writer(fileout,delimiter=',')
                            csvFile.writerow(['Participant','Test',"Features","CA","F1"])
                            csvFile.writerow([p,t,ModelFeatures,ClassAcc,F1])
                        else:
                            fileout = open(write_path+'/ClassificationResult.csv', mode='a', newline='')
                            csvFile = csv.writer(fileout,delimiter=',')
                            csvFile.writerow([p,t,ModelFeatures,ClassAcc,F1])
                        #Can uncomment to obtain a confusion matrix for each session.
                        #confmat = plot_confusion_matrix(np.array(y_test),np.array(y_pred), unique_labels)
                        #plt.plot()
                        #plt.show()
                    
                    elif ClassifyType == 'Kfold':
                        #For each participant and each session perform k-fold cross validation
                        if not os.path.exists("./../ClassificationResults/kfold_classification/"):
                            os.mkdir("./../ClassificationResults/kfold_classification/")
                            write_path = "./../ClassificationResults/kfold_classification/"
                        else:
                            print("Path exists")
                            write_path = "./../ClassificationResults/kfold_classification/"
                        numfolds = 4
                        kfold = KFold(n_splits=numfolds,shuffle=True,random_state=1)

                        scores = cross_val_score(SVM, data_scaled, labels, cv=kfold, scoring='accuracy')
                        y_pred = cross_val_predict(SVM, data_scaled, labels, cv=kfold, method='predict')
                        print("Cross-validation scores:", scores)

                        if not os.path.exists(write_path + '/ClassificationResult.csv'):
                            fileout = open(write_path+'/ClassificationResult.csv', mode='w',newline='')
                            csvFile = csv.writer(fileout,delimiter=',')
                            csvFile.writerow(['Participant','Test',"Features","m_CA","StdDev_CA"])
                            csvFile.writerow([p,t,ModelFeatures,scores.mean(),scores.std()])
                        else:
                            fileout = open(write_path+'/ClassificationResult.csv', mode='a', newline='')
                            csvFile = csv.writer(fileout,delimiter=',')
                            csvFile.writerow([p,t,ModelFeatures,scores.mean(),scores.std()])

                        #confmat = plot_confusion_matrix(np.array(balenced_labels),np.array(y_pred), unique_labels)
                        #plt.plot()
                        #plt.show()
            else:
                #Train on one session for a participant and test on the other. Both options are performed here CA1 corresponds to the classifiaction accuracy where session 1 is tested
                # CA2 is where session 2 was used as the test set. Both are output to the results file.
                if not os.path.exists("./../ClassificationResults/CrossTest_classification/"):
                    os.mkdir("./../ClassificationResults/CrossTest_classification/")
                    write_path = "./../ClassificationResults/CrossTest_classification/"
                else:
                    print("Path exists")
                    write_path = "./../ClassificationResults/CrossTest_classification/"

                T1_labels = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == '1')]['Label'])
                T1_data = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == '1')].iloc[:, 3:])
                T2_labels = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == '2')]['Label'])
                T2_data = np.array(Even_Data[(Even_Data['Participant'] == p) & (Even_Data['Test'] == '2')].iloc[:, 3:])

                scaler = StandardScaler()
                scaler.fit(np.append(T1_data,T2_data,axis=0))
                T1_data = scaler.transform(T1_data)
                T2_data = scaler.transform(T2_data)
                CA1 = 0
                CA2 = 0
                for i in range(2):
                    if i == 0:
                        SVM = svm.SVC(kernel='rbf')
                        SVM.fit(T2_data,T2_labels.reshape(-1))
                        labels_guess = SVM.predict(T1_data)
                        CA1 = metrics.accuracy_score(T1_labels,labels_guess)
                        #confmat = plot_confusion_matrix(np.array(T1_labels),np.array(labels_guess), unique_labels)
                        #plt.plot()
                        #plt.show()
                    if i == 1:
                        SVM = svm.SVC(kernel='rbf')
                        SVM.fit(T1_data,T1_labels.reshape(-1))

                        labels_guess = SVM.predict(T2_data)
                        CA2 = metrics.accuracy_score(T2_labels,labels_guess)
                        #confmat = plot_confusion_matrix(np.array(T2_labels),np.array(labels_guess), unique_labels)
                        #plt.plot()
                        #plt.show()
                average_crosstest_CA = np.mean([CA1,CA2])
                if not os.path.exists(write_path + '/ClassificationResult.csv'):
                    fileout = open(write_path+'/ClassificationResult.csv', mode='w',newline='')
                    csvFile = csv.writer(fileout,delimiter=',')
                    csvFile.writerow(['Participant',"Features","CA1",'CA2'])
                    csvFile.writerow([p,ModelFeatures,CA1,CA2])
                else:
                    fileout = open(write_path+'/ClassificationResult.csv', mode='a', newline='')
                    csvFile = csv.writer(fileout,delimiter=',')
                    csvFile.writerow([p,ModelFeatures,CA1,CA2])
    elif ClassifyType == 'CrossParticipant':
        #Cross participant classification is done using a LOPO methodology. All data from all other participants is used to train the model which is then tested
        # on data from a single participant across both days of testing.
        if not os.path.exists("./../ClassificationResults/CrossParticipant_classification/"):
            os.mkdir("./../ClassificationResults/CrossParticipant_classification/")
            write_path = "./../ClassificationResults/CrossParticipant_classification/"
        else:
            print("Path exists")
            write_path = "./../ClassificationResults/CrossParticipant_classification/"
        for p in participant_list:
            test_labels = np.array(Even_Data[(Even_Data['Participant'] == p)]['Label'])
            test_data = np.array(Even_Data[(Even_Data['Participant'] == p)].iloc[:, 3:])  
            train_labels = np.array(Even_Data[(Even_Data['Participant'] != p)]['Label'])
            train_data = np.array(Even_Data[(Even_Data['Participant'] != p)].iloc[:, 3:])

            scaler = StandardScaler()
            scaler.fit(np.append(train_data,test_data,axis=0))
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)
            
            SVM = svm.SVC(kernel='rbf')
            SVM.fit(train_data,train_labels.reshape(-1))
            y_pred = SVM.predict(test_data)

            ClassAcc = metrics.accuracy_score(test_labels,y_pred)
            F1 = metrics.f1_score(test_labels,y_pred,average='micro')

            if not os.path.exists(write_path + '/ClassificationResult.csv'):
                fileout = open(write_path+'/ClassificationResult.csv', mode='w',newline='')
                csvFile = csv.writer(fileout,delimiter=',')
                csvFile.writerow(['Participant',"CA","F1"])
                csvFile.writerow([p,ClassAcc,F1])
            else:
                fileout = open(write_path+'/ClassificationResult.csv', mode='a', newline='')
                csvFile = csv.writer(fileout,delimiter=',')
                csvFile.writerow([p,ClassAcc,F1])
    else:
        print('Classify type not recognized. Classification was not performed')
    err_dataframe.to_csv(write_path+'ClassificationERRORS.csv',index=False)
                    
    	
