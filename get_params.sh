#!/bin/sh
mkdir params
cd params
curl -L -o params.zip https://www.dropbox.com/sh/6wwryuc1zouxajp/AAC7FjQJes2FqcRQhmz-udT8a?dl=1
unzip params.zip
rm params.zip
cd ../