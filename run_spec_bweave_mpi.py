#
# ~/local/bin/mpiexec -n 2 python2.7 run_bweave_mpi.py
#

import pyfits as p
# should redo with astropy but need it installed
# import pylab as pyl
import numpy as np 
from mpi4py import MPI
import bweavelib as bw
import pickle

comm = MPI.COMM_WORLD
process_rank = comm.Get_rank()
mpi_size = comm.Get_size()

# call mpiexec with -n nchunks+1
nchunks=16
chans_per_chunk=32

if process_rank == 0:
	my_file = 'AGBT09A_085_05.raw.acs.fits'

	hdulist=p.open(my_file)
	h0=p.getheader(my_file,0)
	h1=p.getheader(my_file,1)

	idx1=hdulist.index_of(1)
	data1=hdulist[idx1].data
	data1=hdulist[idx1].data
	# data1.columns to list these-
	xx=data1.field('CRVAL2')
	yy=data1.field('CRVAL3')
	stokes=data1.field('CRVAL4')
	data=data1.field('DATA')
	# LST in seconds-->
	time=data1.field('LST')
	scan=data1.field('SCAN')
	object=data1.field('OBJECT')
	all_ind = (object == 'UrsaMajIIDwarf                  ') & (stokes == -5)

	xx=xx[all_ind]
	yy=yy[all_ind]
	time=time[all_ind]
	scan=scan[all_ind]		
	nchan= data.shape[1]
	data=data[all_ind,:]

	bw.assignscanids(scan)
	scaninfo=bw.classifyscans(scan,xx,yy)
	xinfo=bw.getcrossings(xx,yy,scan,scaninfo)

	# in principle this is of size nchan, but we only actually compute
	#  nchunks * chans_per_chunk
	deltas=np.zeros([xinfo['nx'],nchunks*chans_per_chunk])

	alpha = bw.weavebasket(xinfo)
	for i in range(nchunks):
	 	my_info = {'scaninfo': scaninfo, 'xinfo': xinfo, 'data': data[:,(np.arange(chans_per_chunk)+chans_per_chunk*i)],
 			'scan': scan, 'time': time, 'xx': xx, 'yy': yy, 'alpha': alpha}
 		print 'SENDING INFO'
 		# NB send is whatever, Send is numpy arrays.
	 	#  -i will probably want to change this to a broadcast without the data.
 		comm.send(my_info,dest=(i+1))
	print 'RECEIVING INFO'
 	# and this to a gather-
 	#  -o
 	# i can delete this or use 
 	for i in range(nchunks):
 		print ' RECEIVING', i+1
 		my_deltas=comm.recv(source=(i+1))
 		deltas[:,(np.arange(chans_per_chunk)+chans_per_chunk*i)]=my_deltas
 	print 'DONE'
 	pickle.dump(deltas, open('test_deltas.pkl','wb'))

if process_rank > 0:
	my_info=comm.recv(source=0)
	# now use MPI-IO to read the data...
	print 'Calculating Delta vec!!!'
	# start parallelizing here
	data_wt = np.ones(my_info['xx'].size)
	all_deltas=np.zeros([ my_info['xinfo']['nx'],chans_per_chunk])

	#deltas=bw.makedeltavec(scan,time,xx,yy,data,data_wt,xinfo)
	for j in range(chans_per_chunk):			
		this_data = my_info['data'][:,j]
		print ' STARTING ', process_rank,j
		deltas=bw.makedeltavec(my_info['scan'],my_info['time'],
			my_info['xx'],my_info['yy'],my_info['data'],data_wt,my_info['xinfo'])
		# only accumulating data deltas here not weight from the returned dictionary...
		all_deltas[:,j]=deltas['delta']
		print ' DONE ', j, process_rank
	# actually will need to build up full data chunks from this_data here...
	comm.send(all_deltas,dest=0)
