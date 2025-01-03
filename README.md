# KeypressEMG


 ![image](images/electrode_positioning.png) 

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data Preparation Process](#data-preparation-process)
- [Classification Baselines](#classification-baselines)
- [Baseline Results](#baseline-results)
  
## Link to full dataset
[Electromyographic typing gesture classification dataset
for neurotechnological human-machine interfaces](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/KV65VI)

## Overview
Code and sample participant for dataset containing sEMG data for every alphabetical letter collected from 19
participants across 2 days of testing. The full dataset is provided [here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/KV65VI).


## Dataset Description
19 able-bodied participants were instrumented with 16 channels of sEMG,
8 channels per arm, placed in a circular arrangement on their upper forearm.
Participants are labelled from **P1-P19**.

Two sessions on different days were conducted for each participant.
During each session participants performed two recordings for each letter,
where each key was pressed at a steady rate based off a metronome. 
In each recording the space bar was pressed 5 times followed by 10 presses 
of a single alphabetical key. 

Labelling was done using a keylog that was recorded along with the sEMG data.
Alignment between the keylog and sEMG data was performed between the keylogger 
data and the sEMG data. Both the original key-logs and the csv containing the lag 
timing data between the sEMG and the keylogger data is included with the dataset.

Additionally, baseline classification results are provided for reference. These were obtained using the provided scripts. Results can be found in the folder ClassificationResults\Results\. Within the results folder there are results for four different classification tests. The four baseline tests were a sweep across common feature sets, a sweep across window sizes for feature extraction, and cross participant and cross session classification results. The raw results for each test are provided in the respective folders. The results from these four tests are summarized in the ResultsSummary.xlsx file provided in the main Results folder.

## Directory Structure
Full participant directory can be found in the provided link. A sample participant (P1) is included in the CleanData folder here. The directory structure for the CleanData folder is provided below.
```
CleanData/
	P1-20/
		T1/
			keylogs.txt
			LAG_TIMINGS.csv
			Data/
				A_....rhd
				A_....rhd
				B_....rhd
				B_....rhd
				C_....rhd
				.
				.
				.

		T2/...
```
### File Count
An overview of the number of files in each participants dataset for each test day.

A complete list of the number of key presses for each trial is included in the csv file press_counts.csv.
We can see that there are some tests that are missing 1-3 recordings. 
This is due either to errors in collection leading to stoppage in the 
recording before the end of the trial or to a keylog error not detecting the 
space bar key preses to indicate the start of a trial.

Any warnings or errors generated during data sorting are included in Warnings.csv.

```
P1; T1: Number of Files: 52
P1; T2: Number of Files: 52
P2; T1: Number of Files: 52
P2; T2: Number of Files: 52
P3; T1: Number of Files: 52
P3; T2: Number of Files: 52
P4; T1: Number of Files: 52
P4; T2: Number of Files: 52
P5; T1: Number of Files: 50
P5; T2: Number of Files: 52
P6; T1: Number of Files: 52
P6; T2: Number of Files: 52
P7; T1: Number of Files: 52
P7; T2: Number of Files: 52
P8; T1: Number of Files: 52
P8; T2: Number of Files: 52
P9; T1: Number of Files: 52
P9; T2: Number of Files: 52
P10; T1: Number of Files: 52
P10; T2: Number of Files: 52
P12; T1: Number of Files: 51
P12; T2: Number of Files: 49
P13; T1: Number of Files: 52
P13; T2: Number of Files: 51
P14; T1: Number of Files: 52
P14; T2: Number of Files: 52
P15; T1: Number of Files: 52
P15; T2: Number of Files: 52
P16; T1: Number of Files: 52
P16; T2: Number of Files: 52
P17; T1: Number of Files: 52
P17; T2: Number of Files: 52
P18; T1: Number of Files: 52
P18; T2: Number of Files: 52
P19; T1: Number of Files: 52
P19; T2: Number of Files: 51
P20; T1: Number of Files: 52
P20; T2: Number of Files: 52

```

## Installation
Clone the repository - This will also download the sample P1 data recording. 
```
git clone https://github.com/ANSLab-UHN/sEMG-TypingDatabase.git 
```
Next download full dataset from [link](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/KV65VI) and move the full dataset to the CleanData folder in the downloaded repository.

* Change directory to the cloned folder
```angular2html
cd sEMG-TypingDatabase
```
Install dependencies inside a virtual environment
```angular2html
python -m venv ./venv
```
```angular2html
source venv/bin/activate
```
```angular2html
pip install -e .
```

## Data Preparation Process
### tl;dr
KeypressEMG folder contains a script called `prepare_data.sh`.
Run the script:
```
bash prepare_data.sh
```
and go grab a cup of coffee. 


### Data Preparation Steps


`data_prepare.sh` runs the following 4 python modules
(that you can also run manually):
```angular2html
python -m keypressemg.data_prep.validate
```
```angular2html
python -m keypressemg.data_prep.slice
```
```angular2html
python -m keypressemg.data_prep.extract
```
```angular2html
python -m keypressemg.data_prep.user_features
```

that create the following corresponding folders:
```angular2html
CleanData/
    valid_experiments/
    valid_windows/
    valid_features/
    valid_user_features/
```

The last 3 folders contain `.npy` files of numpy arrays
of signal windows and calculated features of windows and per user features respectively. 
The `valid_experiments` folder contains an `.npy` file for each validated `.rhd` file in the recordings.
Each file contains the relevant data from the corresponding recording.

### Data Preparation Detailed Explanation
#### Validation and Slicing - Create Pi_Tj Labeled Raw Signal Windows
`CleanData/valid_windows` is a folder of numpy arrays stored in npy files. 
Each experiment day of a user is represented by a pair of files.
For example the second experiment day of the first participant is stored in:
```angular2html
P1_T2_X.npy
P1_T2_y.npy
```
The first file contains an array of shape (520, 16, 400) which relates
to 520 16-channel signal windows where each window last 400 timestamps (=0.2 sec).
The second file contains an array of shape (520,), which relates
to the corresponding true labels of each window.

**Note:** Some arrays may contain less than 520 windows.

Produce this folder by:
```angular2html
python -m keypressemg.data_prep.validate
python -m keypressemg.data_prep.slice
```
The first command will go over all experiments recordings, 
preform a validation process and keep the validated signals in `CleanData/valid_experiments`.

The second command then slice each validated file to 0.2 second
windows centered at a key press. These windows are stored in `CleanData/valid_windows`.
#### Feature Vectors of Windows
##### Feature Extraction
Calculate high level features of each window and prepare a folder of
numpy arrays stored in npy files.
The features that are calculated are:
* RMS - Root mean squared.
* LOGVAR - Log of variance.
* WL - Sum of absolute differences between timestamps
* WAMP - Count number of timestamps that their difference
from the previous timestamp is more than a given threshold.
* AR1 and AR2 - Autoregressive model of lags 1 and 2.
All features are calculated per channel to produce a feature vector of shape (96,).

Each feature vector is stored in a file and all these files are stored in `CleanData/valid_features`.

Produce this folder by:
```angular2html
python -m keypressemg.data_prep.extract
```
##### Aggregate User Features
The final file organization step is accumulating all feature vectors relevant to a certain participant in 4 files
named: 
```
P1_T1_X.npy
P1_T1_y.npy
P1_T2_X.npy
P1_T2_y.npy 
```
for participant 1 e.g.
All these files are stored in a folder named `valid_user_windows`.

Produce this folder by:
```angular2html
python -m keypressemg.data_prep.user_features
```
## Docker Setup
As an alternative to Installation and Data preparation, there is a Dockerfile that when built does all that and 
a container based on that image is ready to work.

**Note**: 
Before `docker build`:
* Download the full keypress data from 
[here](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/KV65VI)
  and extract `CleanData` subfolder contents  into this repo `./CleanData`.
* Verify docker daemon (Docker Desktop on Windows) is running.

In order to build the image run:
```angular2html
docker build -t semg-typingdatabase .
```
when finished, the docker image `semg-typingdatabase` is ready to and installed.

Run a container based on built image:
```angular2html
docker run -it semg-typingdatabase bash
```

## Classification Baselines
### Split Day Data for Participant

#### SVM Classification
```angular2html
python -m keypressemg.trainers.train_svm_split_day
```
#### MLP Classification
```angular2html
python -m keypressemg.trainers.train_mlp_split_day

```
### Train on First Day - Eval on other

#### SVM Classification
```angular2html
python -m keypressemg.trainers.train_svm_between_days
```
#### MLP Classification
```angular2html
python -m keypressemg.trainers.train_mlp_between_days
```

## Baseline Results
The results from the preliminary testing have been included in the folder ClassificationResults\Results\. The raw output from these tests is provided in each of the corresponding folders. In each of these folders there is a ClassificationResult.csv that provides the raw numbers output and a ClassificationErrors.csv that lists all the warnings or errors that were noted by the scripts while running. The raw results have been summarized and are provided in ResultsSummary.xlsx

### Cross Participant Classification
A baseline cross participant test was done with the SVM model using LOPO cross validation. The data from all other participants across both test sessions were used to train the model which was then tested on the data from a single participant across both days. From this we obtained a classification accuracy and F1 score for each individual participant. These results are provided in CrossParticipant_classification/ClassificationResult.csv.

### CrossTest_classification
A cross-test classification baseline was done using the SVM baseline model. Here the data from the first test session were used to train a model to be tested on data from the second session for a single participant and vice versa. The raw results of the classification accuracy and F1 score for each participant’s session 1 and 2 are shown in CrossTest_classification/ClassificationResult.csv. These have been summarized and averaged in ResultsSummary.xlsx.

### Feature_classification
A series of feature sets were tested for the SVM classifier. These were tested on a single session of data where 70% was used for training and 30% of the session was used for a test set. The performance for each participant in each of their two sessions is listed in Feature_classification/ClassificationResult.csv. The feature set used is also provided in each row. The mean of each feature set’s performance across all participants is provided in ResultsSummary.xlsx.

### Window_classification
Multiple window sizes were tested during development of the SVM classifier. These were: 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 second non-overlapping windows. Similar to the Feature tests, the classification accuracy and F1 score were found for each participant and each session individually with a 70/30 train/test split. The results are provided in Window_Classification/ClassificationResult.csv. The mean of these results is provided in ResultsSummary.xlsx.



