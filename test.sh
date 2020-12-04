#!/bin/bash
docker build -t foobar .
docker run -w /gcspytorchimagenet -v $PWD:/gcspytorchimagenet foobar bash -c 'cd /gcspytorchimagenet; env PYTHONPATH=/pytorch/xla/test:$PYTHONPATH pytype .; python /gcspytorchimagenet/test_gcsdataset.py'
