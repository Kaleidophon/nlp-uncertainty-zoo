conda create -n test-installation -y python=3.8
conda activate test-installation
pip3 install .
mkdir test_folder
cd test_folder
echo "import nlp_uncertainty_zoo" >> test.py
python3 test.py
cd ..
rm -r test_folder
conda deactivate
conda env remove test-installation