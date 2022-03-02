FROM ubuntu:22.04

RUN apt-get update &&\
    apt-get install -y --fix-missing --no-install-recommends \
    python3 \
    python3-pip

RUN pip3 install folium termcolor geojson  

WORKDIR /tmp