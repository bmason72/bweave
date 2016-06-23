import pyfits as p
# should redo with astropy but need it installed
import pylab as pyl
import numpy as np 

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

import bweavelib as bw
bw.assignscanids(scan)
scaninfo=bw.classifyscans(scan,xx,yy)
xinfo=bw.getcrossings(xx,yy,scan,scaninfo)
deltas=bw.makedeltavec(scan,time,xx,yy,data1,data_wt,xinfo)
