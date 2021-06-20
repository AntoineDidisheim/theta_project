#!/bin/bash
for filename in ./tex/*; do
  echo "$filename"
  cd "$filename"
  pdflatex theta.tex
  scp theta.pdf ~/Dropbox/phd/Projects/theta_project/theta_code/pdf/"$filename".pdf
  cd ~/Dropbox/phd/Projects/theta_project/theta_code
done
