conda create --prefix ./env python=3.7 -y
source activate ./env
pip install -r requirements.txt
conda env export > conda.yaml


## to remove everything -
# rm -rf env/ .gitignore conda.yaml README.md .git/

