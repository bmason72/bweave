#!/bin/bash
# AMI_bweave_cluster_setup.sh

sudo yum install openmpi-devel
#export PATH=/usr/lib64/openmpi/bin:$PATH
#export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib
sudo yum install git
sudo yum install blas
sudo yum install lapack
sudo pip install numpy
sudo pip install pyfits
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-2.0.0.tar.gz
tar xzf mpi4py-2.0.0.tar.gz
cd mpi4py-2.0.0
python setup.py build
sudo python setup.py install

cd ~/
git clone https://github.com/bmason72/bweave.git
cd bweave

# next run
# aws configure
#   have your key ID and secret access key avail.
# then 
#aws s3 cp s3://bweavedata/AGBT09A_085_05.raw.acs.fits ./
#aws s3 cp s3://bweavedata/Segue1_20150205_seczaSubd.fits ./
