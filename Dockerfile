# docker rmi
# docker images
FROM nvidia/cuda
WORKDIR functions/edge_detection/
RUN make
WORKDIR functions/rsa_crypt/
RUN make

MAINTAINER Lek Tin

RUN apt-get install -y software-properties-common python
RUN add-apt-repository ppa:chris-lea/node.js
RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ precise universe" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y nodejs
RUN npm install
RUN npm install
CMD ["/usr/bin/node", "/var/www"]

EXPOSE 3000