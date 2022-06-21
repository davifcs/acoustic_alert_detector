
conda create -n venv-aad
conda activate venv-aad
conda config --add channels conda-forge
conda config --add channels pytorch
conda install --file requirements.txt
pip install git+https://github.com/SiggiGue/pyfilterbank.git
pip install -r requirements2.txt
python main.py --yaml_file config_ref.yaml
