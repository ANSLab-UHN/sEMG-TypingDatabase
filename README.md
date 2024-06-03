# KeypressEMG


 ![image](images/electrode_positioning.png) 

- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data Preparation Process](#data-preparation-process)
   * [Raw Signal Windows](#raw-signal-windows)
   * [Feature Vectors of Windows](#feature-vectors-of-windows)
- [Classification Baselines](#classification-baselines)
   *[Split Day Data for Participant](#split-day-data-for-participant)
   *[Train on First Day Test on Other](#train-on-first-day-test-on-other)

## Paper
[Electromyographic typing gesture classification dataset
for neurotechnological human-machine interfaces](link)

## Overview
Dataset containing EMG data from 27 typed characters (alphabetical + space) for 19 participants across 2 days


## Dataset Description
19 able bodied participants were instrumented with 16 channels of sEMG, 8 channels per arm, placed in a circular arrangement on their upper forearm. Participants are labelled from P1-P19. 

Two sessions on different days were conducted for each participant. During each session participants performed two recordings for each letter where each key was pressed at a steady rate based off a metronome. In each recording the the spacebar was pressed 5 times followed by 10 presses of a single alphabetical key. 

Labelling was done using a keylog that was recorded along with the sEMG data. Alignment between the keylog and sEMG data was performed between the keylogger data and the sEMG data. Both the original keylogs and the csv containing the alignment timing data between the sEMG and the keylogger data is included with the dataset.

## Directory Structure
Each participant has one file containing recordings from day one and day two individually. Within this file there are the original recordings in .rhd format contained in KeyData, keylogs, and alignment timings in LAG_TIMINGS.csv. There are not always 52 recordings from each participant as some were corrupted or noisy during recording and were excluded in post processing. 

```
Data/
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

A list of the number of files for each recording is included in filesummary.txt. Baseline results as discussed in the paper are included within the ClassificationResults folder. This includes both "RawResults" and a ResultsSummary excel sheet.

## Installation

* Clone the repository - This will also download the data recordings. 
```
git clone https://github.com/ANSLab-UHN/sEMG-TypingDatabase.git 
```

* Change directory to the cloned folder
```angular2html
cd sEMG-TypingDatabase
```
* Install dependencies inside a virtual environment
```angular2html
python -m venv .venv
```
```angular2html
source venv/bin/activate
```
```angular2html
pip install -e .
```
or alternatively, if [Poetry](https://python-poetry.org/docs/) is used
```
poetry shell
```
```angular2html
poetry install
```

## Data Preparation Process
### tl;dr
Run these 4 commands
```angular2html
python -m keypressemg.validate
```
```angular2html
python -m keypressemg.slice
```
```angular2html
python -m keypressemg.extract
```
```angular2html
python -m keypressemg.user_features
```
for the following folders to be created:
```angular2html
CleanData/
    valid_experiments/
    valid_windows/
    valid_features/
    valid_user_features/
```
The last 3 folders contain `.npy` files of numpy arrays
of signal windows and calculated features of windows and per user features respectively. 
The `valid_experiments` folder contain an `.npy` file for each validated `.rhd` file in the recordings.
Each file contains the relevant data from the corresponding recording.
### Raw Signal Windows
Prepare a folder of numpy arrays stored in npy files. 
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
python -m keypressemg.validate
python -m keypressemg.slice
```
The first command will go over all experiments recordings, 
preform a validation process and keep the validated signals in a folder named `valid_experiments`.
The second command then slice each validated file to 0.2 second
windows centered at a key press. These windows are stored in a folder named `valid_windows`.
### Feature Vectors of Windows
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
Each feature vector is stored in a file and all these files are stored in a folder named `valid_features`.
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


## Classification Baselines
### Split Day Data for Participant
Split 
* SVM Classification
```angular2html
python -m keypressemg.train_svm_split_day
```
* MLP Classification
```angular2html
python -m keypressemg.train_mlp_split_day

```
### Train on First Day - Eval on other

* SVM Classification
```angular2html
python -m keypressemg.train_svm_between_days
```
* MLP Classification
```angular2html
python -m keypressemg.train_mlp_between_days
```

## Contact Information
For any questions regarding the dataset you can contact the principal investigator of this work Dr. Jose Zariffa at jose.zariffa@utoronto.ca
