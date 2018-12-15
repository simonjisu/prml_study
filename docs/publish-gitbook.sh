#!/bin/bash
rm -rf _book
rm -rf docs

gitbook install && gitbook build

mkdir docs
cp -R _book/* docs/

git clean -fx _book

git add .
git commit -a -m "update docs"
git push -u origin master
