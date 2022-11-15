# Rebuild documentation
cd docs
rm ./README_DOCS.md
cat ../README.md | sed 's/:\([a-z_]*\):/|:\1:| /g' > ./README_DOCS.md
mv ../README.md ../README.mm
make clean && make html
mv ../README.mm ../README.md
cp -r _build/html/* ./
rm -r _build/html/*
cd ..
