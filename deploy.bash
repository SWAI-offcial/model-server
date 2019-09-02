#!/bin/bash

git add *
git commit -m "`date +%Y-%m-%d-%H:%S`"
git push origin master