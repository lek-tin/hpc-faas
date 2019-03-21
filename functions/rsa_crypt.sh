#!/bin/sh
ssh ltin@bender.engr.ucr.edu 'cd functions/rsa_crypt/ &&
./RSA 2 && exit'
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/decrypted_cpu.txt ./functions/rsa_crypt/output/decrypted_cpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/decrypted_gpu.txt ./functions/rsa_crypt/output/decrypted_gpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/encrypted_cpu.txt ./functions/rsa_crypt/output/encrypted_cpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/encrypted_gpu.txt ./functions/rsa_crypt/output/encrypted_gpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/input.txt ./functions/rsa_crypt/output/input.txt