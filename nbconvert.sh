#!/bin/bash
test_str=$(echo $1 | cut -d "/" -f1)
if [ $1 = "--help" ] 
then
	echo "insert notebook dir: ./notebook/~"
	exit
else
	if [ $test_str = "." ]
	then
		convert_path=$(echo $1 | cut -d "/" -f2-)
	else
		convert_path=$1
	fi
	chapterdir=$(echo $convert_path | cut -d "/" -f2)
	filename=$(echo $convert_path | cut -d "/" -f3 | cut -d "." -f1)
        filedir="${filename}_files"
        new_filename=$(echo $filename | cut -d "_" -f2)
	
	jupyter nbconvert --to markdown --output-dir _nbconvert/$chapterdir $convert_path
	mv ./_nbconvert/$chapterdir/$filedir ./_nbconvert/$chapterdir/$new_filename
	cd ./_nbconvert/$chapterdir/$new_filename
	rename -v 's/^nb_//' *
	cd $HOME/code/prml_study
	rsync -avh ./_nbconvert/$chapterdir ./posts/assets/
	rm ./posts/assets/$chapterdir/nb*
	rm -rf ./_nbconvert/$chapterdir/$new_filename
	echo "all _nbconvert/${chapterdir}/${filedir} moved to asset/${chapterdir}/${new_filename}"
fi
