#!/bin/sh
# ssh ltin@bender.engr.ucr.edu 'cd functions/rsa_crypt/ &&
# ./RSA 2 && exit'
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/decrypted_cpu.txt ../public/rest/rsa_crypt/decrypted_cpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/decrypted_gpu.txt ../public/rest/rsa_crypt/decrypted_gpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/encrypted_cpu.txt ../public/rest/rsa_crypt/encrypted_cpu.txt &&
scp ltin@bender.engr.ucr.edu:functions/rsa_crypt/encrypted_gpu.txt ../public/rest/rsa_crypt/encrypted_gpu.txt