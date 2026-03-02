This is a misinformation detection Project for a final year honours project, the goal is to use Machine learning to gain insight into the authentisity of statements and to detect misinformation.
In order to run this system: 

1. Create a python virutal Enviroment.
2. Install dependencies: see requirements.txt
3. run preprocessing or back-end scripts as required. 

Returning users: 
1. start Python Enviroment ".\.venv\Scripts\Activate.ps1"
2. run files for single testing "python Foldername/filename.py"
3. ensure requirements are updated after installing packages with "pip freeze > requirements.txt"


1. run preprocessing "python -m Backend.Pipeline.Preprocessing"

1. Run DistilBERT loader "python -m Backend.Models.DistillBERT_loader"
2. Run DistilBERT Trainer "python -m Backend.Models.DistillBERT_Train" 
3. Run DistilBERT pred "python -m Backend.Models.DistillBERT_Predict"

1. Run CLIP loader "python -m Backend.Models.CLIP_loader"
2. Run CLIP Trainer "" 
3. Run CLIP pred "python -m Backend.Models.CLIP_predict"

1. Run both "python -m Backend.run"
2. run dashboard terminal 2 "cd FrontEnd
npm start"