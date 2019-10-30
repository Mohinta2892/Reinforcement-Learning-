Please install Python 3

LINUX
And then load the packages using :
pip install -r requirements.txt

OR

pip3 install -r requirements.txt



WINDOWS ANACONDA
You may use conda like:

conda create -y --name py37 python==3.7
conda install -f -y -q --name py37 -c conda-forge --file requirements.txt
conda activate py37


OR

conda create -y --name py37 python==3.7
conda activate py37

#  Install via `conda` directly.
#  This will fail to install all
#  dependencies. If one fails,
#  all dependencies will fail to install.
#
conda install --yes --file requirements.txt

#
#  To go around issue above, one can
#  iterate over all lines in the
#  requirements.txt file.
#
while read requirement; do conda install --yes $requirement; done < requirements.txt
