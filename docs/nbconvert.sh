#!/bin/bash
test_str=$(echo $1 | cut -d "/" -f1)
if [ $1 = "--help" ] 
then
	echo "insert notebook dir: ./notebook/~"
else
	if [ $test_str = "." ]
	then
		convert_path=$(echo $1 | cut -d "/" -f2-)
	else
		convert_path=$1
	fi
	jupyter nbconvert --to markdown --output-dir _nbconvert $convert_path
	filename=$(echo $convert_path | cut -d "/" -f3 | cut -d "." -f1)
	filedir="${filename}_files"
	new_filename=$(echo $filename | cut -d "_" -f2)
	mv ./_nbconvert/$filedir ./_nbconvert/$new_filename
	cd ./_nbconvert/$new_filename
	rename -v 's/^nb_//' *
	cd $HOME/code/prml_study
	rsync -a ./_nbconvert/$new_filename ./posts/assets/
	rm -rf ./_nbconvert/$new_filename
	echo "all _nbconvert/${filedir} moved to asset/${new_filename}"
fi
