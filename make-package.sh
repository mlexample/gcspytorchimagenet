#!/bin/bash
if [ -z "$BUCKET" ]
then
      echo '$BUCKET variable must be defined'
      exit 1
fi
tar czf thepackage.tar.gz --transform 's,^,thepackage/,' --add-file test_train_mp_imagenet.py  --add-file mydataloader.py --add-file go.sh
gsutil cp thepackage.tar.gz gs://${BUCKET}/thepackage.tar.gz
