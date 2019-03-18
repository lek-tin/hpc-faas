# docker rmi
# docker images
FROM socalucr/gpgpu-sim
WORKDIR functions/edge_detection/
RUN nvcc xxx
WORKDIR functions/image_blur/
RUN nvcc yyy
WORKDIR functions/encryption_sha/
RUN nvcc zzz

FROM nodejs/
start server

expose port 3000