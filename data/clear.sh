rm -rf train 2> /dev/null
rm -rf valid 2> /dev/null
mkdir train
mkdir train/image
mkdir train/image/0
mkdir train/label
mkdir train/label/0
cp -r train valid
