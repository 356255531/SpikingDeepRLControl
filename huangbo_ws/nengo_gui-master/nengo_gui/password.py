"""Password hashing functions replicating bcrypt API."""

from __future__ import print_function

import binascii
from getpass import getpass
import hashlib
import os


def gensalt(size=16):
    return binascii.hexlify(os.urandom(size))

def hashpw(password, salt, algorithm='sha1'):
    h = hashlib.new(algorithm)
    h.update(password)
    h.update(salt)
    return algorithm + ':' + salt + ':' + h.hexdigest()

def checkpw(password, hashed):
    algorithm, salt, _ = hashed.split(':')
    return hashpw(password, salt, algorithm) == hashed

def prompt_pw():
    while True:
        p0 = getpass("Enter password: ")
        p1 = getpass("Enter password: ")
        if p0 == p1:
            return p0
        print("Passwords do not match. Please try again.")
