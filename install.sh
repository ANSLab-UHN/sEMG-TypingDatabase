#!/bin/bash

echo "Creating a virtual environment"
python3 -m venv ./venv
if [ $? -ne 0 ]; then
  echo "Virtual environment creation failed."
  exit 1
fi
echo "Virtual environment created."

echo "Activating a virtual environment"
source venv/bin/activate
if [ $? -ne 0 ]; then
  echo "Virtual environment activation failed."
  exit 1
fi
echo "Virtual environment activated."

echo "pip install requirements"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Requirements installation failed. Check that 'requirements.txt' exists"
  exit 1
fi

echo "keypressemg installed successfully."