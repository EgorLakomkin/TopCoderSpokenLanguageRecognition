############################################################
# Dockerfile to run Egor Lakomkin Spoken Language Recognition submission
# Based on Ubuntu Image
############################################################

# Set the base image to use to Ubuntu
FROM ubuntu:14.04

# Set the file maintainer (your name - the file's author)
MAINTAINER Egor Lakomkin

RUN apt-get update
RUN sudo apt-get install -y build-essential python-setuptools python-dev libfreetype6-dev libxft-dev wget unzip libav-tools
RUN sudo easy_install pip
RUN sudo pip install virtualenv
RUN sudo apt-get install -y zip liblapack-dev gfortran

#installing libs
RUN pip install cython joblib librosa numpy scipy scikit-learn path.py

#download source code
RUN wget https://www.dropbox.com/s/bcr0qs91rr4g5et/egorlakomkin-language-33cc5d631d24.zip -O source.zip
RUN unzip source.zip -d /root/source
RUN cp -R /root/source/egorlakomkin-language-33cc5d631d24/* /root/source

RUN mkdir -p /root/source/data

RUN wget http://www.topcoder.com/contest/problem/SpokenLanguages/S1.zip -O /root/source/data/S1.zip
RUN wget http://www.topcoder.com/contest/problem/SpokenLanguages/S2.zip -O /root/source/data/S2.zip

RUN mkdir -p /root/source/data/data
RUN unzip /root/source/data/S1.zip -d /root/source/data/data
RUN unzip /root/source/data/S2.zip -d /root/source/data/data
RUN wget http://www.topcoder.com/contest/problem/SpokenLanguages/trainingset.csv -O /root/source/data/trainingset.csv
RUN wget http://www.topcoder.com/contest/problem/SpokenLanguages/testingset.csv -O /root/source/data/testingset.csv

RUN mkdir -p /root/voxforge
#dl voxforge
RUN wget https://www.dropbox.com/s/phh99wstx5zo313/voxforge.zip -O /root/source/data/voxforge.zip
RUN mkdir -p /store/egor/voxforge
RUN unzip /root/source/data/voxforge.zip -d /store/egor/voxforge



RUN cd /root/source && python utils.py
RUN mv /root/source/data/*.wav /root/source/data/data
RUN cd /root/source && python bag_of_words.py use_voxforge

