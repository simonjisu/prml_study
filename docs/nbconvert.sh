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
	rsync -a ./_nbconvert/$filedir ./assets/
	rm -rf ./_nbconvert/$filedir
	#mv -f ./_nbconvert/$filedir ./assets
	#rm "./_nbconvert/${filename}.md"
	echo "all _nbconvert/${filedir} moved to asset/${filedir}"
fi
