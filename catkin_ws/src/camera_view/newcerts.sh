#!/bin/bash


# remove any old files
rm -f camera_view/ssl/cert.pem
rm -f ssl/key.pem

# generate one pair for the camera_view/ssl folder
openssl req -x509 -newkey rsa:2048 \
    -keyout ssl/key.pem \
    -out camera_view/ssl/cert.pem \
    -days 365 -nodes \
    -subj "/CN=localhost"


# fix permissions
chmod 644 camera_view/ssl/cert.pem
chmod 600 ssl/key.pem
