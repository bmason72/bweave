
#
# ~/local/bin/mpiexec -n 2 python2.7 run_bweave_mpi.py
#

import pyfits as p
# should redo with astropy but need it installed
import pylab as pyl
import numpy as np 
from mpi4py import MPI
import bweavelib as bw

comm = MPI.COMM_WORLD
process_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if process_rank == 0:
	#my_file = 'AGBT09A_085_05.raw.acs.fits'
	my_file='Segue1_20150205_seczaSubd.fits'

	hdulist=p.open(my_file)
	h0=p.getheader(my_file,0)
	h1=p.getheader(my_file,1)

	idx1=hdulist.index_of(1)
	data1=hdulist[idx1].data
	xx=data1.field('dx')
	yy=data1.field('dy')
	scan=data1.field('scan')
	time=data1.field('time')
	data=data1.field('fnu')

	bw.assignscanids(scan)
	scaninfo=bw.classifyscans(scan,xx,yy)
	xinfo=bw.getcrossings(xx,yy,scan,scaninfo)
 	my_info = {'scaninfo': scaninfo, 'xinfo': xinfo, 'data': data,
 		'scan': scan, 'time': time, 'xx': xx, 'yy': yy}
 	print 'SENDING INFO'
 	# NB send is whatever, Send is numpy arrays.
 	#  -i will probably want to change this to a broadcast without the data.
 	comm.send(my_info,dest=1)
 	print 'RECEIVING INFO'
 	# and this to a gather-
 	#  -o
 	# i can delete this or use 
 	my_deltas=comm.recv(source=1)
 	print 'DONE'

if process_rank > 0:
	my_info=comm.recv(source=0)
	# now use MPI-IO to read the data...
	print 'Calculating Delta vec!!!'
	# start parallelizing here
	data_wt = np.ones(my_info['xx'].size)
	#deltas=bw.makedeltavec(scan,time,xx,yy,data,data_wt,xinfo)
	deltas=bw.makedeltavec(my_info['scan'],my_info['time'],
		my_info['xx'],my_info['yy'],my_info['data'],data_wt,my_info['xinfo'])
	comm.send(deltas,dest=0)
