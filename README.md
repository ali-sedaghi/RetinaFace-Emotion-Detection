# change directory to code folder
cd code

# install dependencies
conda env create -f environment.yml
conda activate retinaface-roi

# image - results will be saved in results folder
python run.py --img_path="../tests/test1.jpg"

# webcam
python run.py --webcam=True

