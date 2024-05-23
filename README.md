# EMG-TypingDatabase
Dataset containing EMG data from 27 typed characters (alphabetical + space) for 19 participants across 2 days

## Description
19 able bodied participants were instrumented with 16 channels of sEMG, 8 channels per arm, placed in a circular arrangement on their upper forearm. Participants are labelled from P1-P19. Two sessions on different days were conducted for each participant. During each session participants performed two recordings for each letter where each key was pressed at a steady rate based off a metronome. In each recording the the spacebar was pressed 5 times followed by 10 presses of a single alphabetical key. Labelling was done using a keylog that was recorded along with the sEMG data. Alignment between the keylog and sEMG data was performed between the keylogger data and the sEMG data. Both the original keylogs and the csv containing the alignment timing data between the sEMG and the keylogger data is included with the dataset.

## Structure
Each participant has one file containing recordings from day one and day two individually. Within this file there are the original recordings in .rhd format contained in KeyData, keylogs, and alignment timings in LAG_TIMINGS.csv. There are not always 52 recordings from each participant as some were corrupted or noisy during recording and were excluded in post processing. A list of the number of files for each recording is included in filesummary.txt.

Data/ P1-19/ T1; T2/ keylogs.txt; LAG_TIMINGS.csv; KeyData/ A_....rhd; A_....rhd; B_....rhd; ...

The Code folder contains necessary scripts to perform the classification from validation testing. Within the code folder the program classify_root.py is the script where all different settings are set for classification. Classify.py is called from this script and performs the classification according to parameters set in classify_root.py. The other scripts are to read the original intan files.

Baseline results as discussed in the paper are included within the ClassificationResults folder. This includes both "RawResults" and a ResultsSummary excel sheet.

## Contact Information
For any questions regarding the dataset you can contact the principal investigator of this work Dr. Jose Zariffa at jose.zariffa@utoronto.ca
