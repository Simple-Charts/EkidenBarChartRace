# Ekiden Bar Chart Race
  This is the source code for creating a bar chart race animation of Hakone Ekiden Marathon Relay.  

## Example  
  [![Simple Charts / Hakone Ekiden Marathon Relay](https://img.youtube.com/vi/oZQfj7Tp0No/0.jpg)](https://www.youtube.com/watch?v=oZQfj7Tp0No "Simple Charts / Hakone Ekiden Marathon Relay")  

## Data Source
  https://www.hakone-ekiden.jp/record/  

## Setup  
  1. install Anaconda3-2020.02  
      https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe
  2. create virtual environment and install packages  
      conda create -n bca python==3.7.6 anaconda  
      conda activate bca  
      conda install -c conda-forge opencv==3.4.1  

## Usage  
  Send the following commands in virtual environment of python  
  &nbsp;&nbsp;&nbsp;&nbsp; cd {folder path}  
  &nbsp;&nbsp;&nbsp;&nbsp; python 1_get_raw_data.py  
  &nbsp;&nbsp;&nbsp;&nbsp; python 2_preprocess.py  
  &nbsp;&nbsp;&nbsp;&nbsp; python 3_EkidenBarChartRace.py  
