#!/bin/bash
if [ -z "$BUCKET" ]
then
      echo '$BUCKET variable must be defined'
      exit 1
fi
tar czf thepackage.tar.gz --transform 's,^,thepackage/,' --add-file test_train_mp_petastorm.py --add-file go-peta.sh
gsutil cp thepackage.tar.gz gs://${BUCKET}/thepackage.tar.gz