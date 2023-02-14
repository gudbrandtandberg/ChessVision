# Use sed to make a copy of index.html in the current directory, replacing {{endpoint}} with lambda
sed 's/{{endpoint}}/lambda/g' webroot/templates/index.html > webroot/index.html

# Then copy webroot to the chessvision-web bucket
python copy_bucket.py