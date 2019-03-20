# docker rmi
# docker images
FROM socalucr/gpgpu-sim
WORKDIR functions/edge_detection/
RUN nvcc xxx
WORKDIR functions/image_blur/
RUN nvcc yyy
WORKDIR functions/encryption_sha/
RUN nvcc zzz

MAINTAINER Lek Tin

RUN apt-get install -y software-properties-common python
RUN add-apt-repository ppa:chris-lea/node.js
RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ precise universe" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y nodejs
#RUN apt-get install -y nodejs=0.6.12~dfsg1-1ubuntu1
CMD ["/usr/bin/node", "/var/www"]

EXPOSE 3000