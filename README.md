# Text-AudioDatasets
Data pipeline for different audio data sets 

To install 
```
git clone git@github.com:tmontaj/Text-AudioDatasets.git
cd Text-AudioDatasets
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
mkdir dataset/librispeech
```

To run auto test
```
pytest
```

To run test usecase (to test the pipeline)
```
python usecases/test.py
```
