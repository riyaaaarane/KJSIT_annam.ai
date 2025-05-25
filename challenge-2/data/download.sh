# Downloading a dataset in google-colab
# === Example: Download a dataset === 
pip install kaggle

#Create kaggle api (kaggle.json)
from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!mv "kaggle (2).json" ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json

!kaggle competitions download -c soil-classification- part 2

!unzip soil-classification- part 2.zip
