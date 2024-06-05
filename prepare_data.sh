#!/bin/bash

# Run the validation command
python -m keypressemg.data_prep.validate
if [ $? -ne 0 ]; then
  echo "Validation failed."
  exit 1
fi
echo "Validation done."

# Run the slice command
echo "Slice signal to windows"
python -m keypressemg.data_prep.slice
if [ $? -ne 0 ]; then
  echo "Slicing failed."
  exit 1
fi
echo "Slicing done."

# Run the extract command
echo "Extract High Level Features"
python -m keypressemg.data_prep.extract
if [ $? -ne 0 ]; then
  echo "Extraction failed."
  exit 1
fi
echo "Extraction done."

# Run the user features command
echo "Aggregate to user feature vectors"
python -m keypressemg.data_prep.user_features
if [ $? -ne 0 ]; then
  echo "User features extraction failed."
  exit 1
fi

echo "Data preparations executed successfully."
