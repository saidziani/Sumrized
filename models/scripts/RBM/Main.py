#!/usr/bin/python3

import os
import Summary
from Model import rbm_model 

article = "article1.txt"
lang = "en"
summary = Summary.Summary(article, lang)
summary.main()
summary_path = rbm_model(article, lang)
os.system("gedit "+summary_path)