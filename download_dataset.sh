#!/bin/sh

mkdir -p dataset/images predictions

wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip --no-check-certificate
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip --no-check-certificate
wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip --no-check-certificate

wget https://storage.googleapis.com/vw-c/annotations.json

unzip -q -o train.zip -d dataset/images
unzip -q -o val.zip -d dataset/images
unzip -q -o test.zip -d dataset/images

rm train.zip val.zip test.zip

mv -v dataset/images/train/* dataset/images/
mv -v dataset/images/val/* dataset/images/
mv -v dataset/images/test/* dataset/images/

mv -v annotations.json dataset/
