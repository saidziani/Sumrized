#!/usr/bin/python3

from bs4 import BeautifulSoup
import os, re

root = "../data/raw"
directory = os.listdir(root)

if not os.path.exists("_preprop"):
    os.makedirs("_preprop")

for path in directory:
    if os.path.isdir(root+"/"+path):
        if not os.path.exists("_preprop/"+path+"_preprop"):
            os.makedirs("_preprop/"+path+"_preprop")

        files = os.listdir(root+"/"+path)
        for file in files:
            if os.path.isfile(root+"/"+path+"/"+file):
                name = file.split('.')
                packet_to_list = []
                newFile = open("_preprop/"+path+"_preprop/"+name[0]+"_prep.xml", 'w')
                content = open(root+"/"+path+"/"+file, 'r').read()
                soup = BeautifulSoup(content,'lxml')
                packet = soup.find_all("packet", attrs={"operation" : ["S", "R", "C", "G"]})
                for pack in packet:
                    pack = str(pack)
                    pack = re.sub(r'<compression>.*</compression>\n*', '', pack)
                    pack = re.sub(r'<generalization>.*</generalization>\n*', '', pack)
                    pack = re.sub(r'(?:<source>|</source>)', '', pack)
                    pack = re.sub(r'packet', 'source', pack)
                    pack = re.sub(r'(?:C|G)', '?', pack)
                    packet_to_list.append(pack)
                head = '<?xml version="1.0" encoding="UTF-8"?>\n<file>\n'
                foot = '\n</file>'
                newFile.write(head+'\n'.join(packet_to_list)+foot)
                print(file)

