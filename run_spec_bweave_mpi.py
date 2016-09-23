#
# ~/local/bin/mpiexec -n 2 python2.7 run_spec_bweave_mpi.py
#

import pyfits as p
# should redo with astropy but need it installed
# import pylab as pyl
import numpy as np 
from mpi4py import MPI
import bweavelib as bw
import pickle

# set object name and stokes value here
my_obj = "UrsaMajIIDwarf"
my_stokes = -5
my_file = 'AGBT09A_085_05.raw.acs.fits'
# call mpiexec with -n nchunks+1
nchunks=3
chans_per_chunk=8
lam = 0.05

###
comm = MPI.COMM_WORLD
process_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if process_rank == 0:
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
	object=np.char.strip(data1.field('OBJECT'))
	print "Unique Objects"
	print np.unique(object)
	print "Unique Stokes values"
	print np.unique(stokes)

	# select desired data & use np.memmap to write ndarray (ngood x nchan) to .npy binary file
	all_ind = (object == my_obj) & (stokes == my_stokes)
	z=np.where(all_ind)
	zz=z[0] # this is an array of indices....
	data_file = my_file + '.' + my_obj+ '.stokes' + np.str(my_stokes) +'.npy'
	n_good_rows = zz.shape[0]
	n_spec_chans = data.shape[1]
	fp = np.memmap(data_file, dtype=data.dtype, mode='w+', shape=(n_good_rows,n_spec_chans))
	for i in range(n_good_rows):
		fp[i,:] = data[zz[i],:]
	# flush buffer & close the file -
	del fp	

	xx=xx[all_ind]
	yy=yy[all_ind]
	time=time[all_ind]
	scan=scan[all_ind]		
	nchan= data.shape[1]
	data=data[all_ind,:]

	bw.assignscanids(scan)
	scaninfo=bw.classifyscans(scan,xx,yy)
	xinfo=bw.getcrossings(xx,yy,scan,scaninfo)

	# Save some meta information to a solution descriptor file
	#  mostly this is to make it easy to read the memmap files back in.
	#  raw nrows , raw nchan , nx , nchunks*chans per chunk, nchunks, chans per chunk, data type
	desc_fp = open(data_file+'.solndesc',mode='w')
	desc_fp.write(' rows_of_data: '+np.str(n_good_rows)+' \n')
	desc_fp.write(' raw_nchan: '+np.str(n_spec_chans)+' \n')
	desc_fp.write(' n_ix: '+np.str(xinfo['nx'])+' \n')
	desc_fp.write(' nchunks: '+np.str(nchunks)+' \n')
	desc_fp.write(' chans_per_chunk: '+np.str(chans_per_chunk)+' \n')
	desc_fp.write(' dtype: '+ np.str(data.dtype)+' \n')
	desc_fp.close()

	# in principle this is of size nchan, but we only actually compute
	#  nchunks * chans_per_chunk
	deltas=np.zeros([xinfo['nx'],nchunks*chans_per_chunk])

	alpha = bw.weavebasket(xinfo)
	for i in range(nchunks):
		my_info = {'scaninfo': scaninfo, 'xinfo': xinfo, 'data_file': data_file,'which_chunk':i,
			'n_good_rows':n_good_rows,'n_spec_chans':n_spec_chans,'data_type': data.dtype,
 			'scan': scan, 'time': time, 'xx': xx, 'yy': yy, 'alpha': alpha}
	 	#my_info = {'scaninfo': scaninfo, 'xinfo': xinfo, 'data': data[:,(np.arange(chans_per_chunk)+chans_per_chunk*i)],
 		#	'scan': scan, 'time': time, 'xx': xx, 'yy': yy, 'alpha': alpha}
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
 		blah=comm.recv(source=(i+1))
 		print 'RECEIVED: ',blah
 		#deltas[:,(np.arange(chans_per_chunk)+chans_per_chunk*i)]=my_deltas
 	print 'DONE'
 	#pickle.dump(deltas, open('test_deltas.pkl','wb'))

if process_rank > 0:
	my_info=comm.recv(source=0)
	# read data using np.memmap()
	my_data = np.memmap(my_info['data_file'], dtype=my_info['data_type'], mode='r',\
		shape=(my_info['n_good_rows'],my_info['n_spec_chans']))

	# create this by fiat for now-
	data_wt = np.ones(my_info['xx'].size)

	print 'Calculating Delta vec!!!'
	# start parallelizing here
	# output file(s)-
	file_root = (my_info['data_file'].split('npy'))[0]
	delta_file = file_root + 'delta.npy'
	all_deltas=np.memmap(delta_file,dtype=my_info['data_type'],mode='w+', \
		shape=(my_info['xinfo']['nx'],nchunks*chans_per_chunk))
	delta_mod_file = file_root + 'deltaMod.npy'
	all_delta_mod=np.memmap(delta_mod_file,dtype=my_info['data_type'],mode='w+', \
		shape=(my_info['xinfo']['nx'],nchunks*chans_per_chunk))
	#new_data_file = file_root + 'debase.npy'
	#all_deltas=np.zeros([ my_info['xinfo']['nx'],chans_per_chunk])

	#deltas=bw.makedeltavec(scan,time,xx,yy,data,data_wt,xinfo)
	for j in range(chans_per_chunk):			
		this_data = my_data[:,my_info['which_chunk']*chans_per_chunk+j]
		print ' STARTING ', process_rank,j
		deltas=bw.makedeltavec(my_info['scan'],my_info['time'],
			my_info['xx'],my_info['yy'],this_data,data_wt,my_info['xinfo'])
		#deltas=bw.makedeltavec(my_info['scan'],my_info['time'],
		#	my_info['xx'],my_info['yy'],my_info['data'],data_wt,my_info['xinfo'])
		# only accumulating data deltas here not weight from the returned dictionary...
		# save deltas - 
		all_deltas[:,my_info['which_chunk']*chans_per_chunk+j]=deltas['delta']
		# solve! ->
		a2=np.dot( my_info['alpha'].transpose() , my_info['alpha'])
		# regularized version - 
		# a3 = a2 + lambda * matrix_scale * identity()
		a3 = a2 + lam * np.trace(a2)/a2.shape[0] * np.identity(a2.shape[0])
		sol_par = np.dot(np.dot(np.linalg.inv(a3), my_info['alpha'].transpose()),deltas['delta'])
		# save delta mod
		all_delta_mod[:,my_info['which_chunk']*chans_per_chunk+j] = np.dot(my_info['alpha'],sol_par)		
		# save data model
		# save data-model
		print ' DONE ', j, process_rank
	# actually will need to build up full data chunks from this_data here...
	del my_data
	del all_deltas
	del all_delta_mod

	# del <other stuff>
	comm.send(process_rank,dest=0)
