#!/bin/sh

mkdir params
cd params

# download wordnet parameters
mkdir wordnet
cd wordnet
curl -L -o wordnet.zip https://www.dropbox.com/sh/jonav59hcaoy82a/AAAhPs3i1DzbZF_kao-UrVyNa?dl=1
unzip wordnet.zip
rm wordnet.zip
cd ../

# download freebase parameters (one by one due to dropbox file size limits)
mkdir freebase
cd freebase
mkdir bilinear
mkdir bilinear_diag
mkdir transE

cd bilinear
curl -L -o bilinear.zip https://www.dropbox.com/sh/fo5u9jmvhw648gr/AABtc2sHiCZasc2mQDpuHXVUa?dl=1
unzip bilinear.zip
rm bilinear.zip

cd ../bilinear_diag
curl -L -o bilinear_diag.zip https://www.dropbox.com/sh/r3dis9sfxufsdf3/AAAe3wPZw4GusWmm-b1FMEjwa?dl=1
unzip bilinear_diag.zip
rm bilinear_diag.zip

cd ../transE
curl -L -o transE.zip https://www.dropbox.com/sh/hc3ww77rp06a1x9/AACtXmhXHaln3i1gneYgVomTa?dl=1
unzip transE.zip
rm transE.zip
cd ../../../