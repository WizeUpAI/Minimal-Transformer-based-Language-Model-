pip install .
mkdir data
echo "Your text data here..." > data/train.txt
python train.py

python test.py
