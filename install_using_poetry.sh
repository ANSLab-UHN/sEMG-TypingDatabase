#!/bin/bash

echo "Activating a virtual environment"
poetry shell
if [ $? -ne 0 ]; then
  echo " 'poetry shell' failed."
  echo "Do you have Poetry installed?"
  echo "Do you have a pyproject.toml file? If poetry is installed consider 'poetry init' "
  exit 1
fi
echo "Virtual environment activated."

echo "poetry lock - prepare dependencies"
poetry lock --no-update
if [ $? -ne 0 ]; then
  echo " 'poetry lock' failed. Probably some dependency problem. See output from first error."
  exit 1
fi

echo "Install keypressemg and its dependencies"
poetry install
if [ $? -ne 0 ]; then
  echo " 'poetry install' failed."
  exit 1
fi

echo "keypressemg installed successfully."