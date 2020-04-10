#!/bin/bash

cd original_labels

for filename in *.png; do
	echo $filename
	# Remove \! if you want to keep the original ratio of the label image
	convert -resize 1024x512 $filename ../resized_labels/$filename
done
