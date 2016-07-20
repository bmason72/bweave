import numpy as np
import scipy.special as sps

# delete pylab for production use, this is just for debugging - 
#import pylab as pyl

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def assignscanids(scan):
	"""	re-number "scan" list (numpy nd-array) sequentially starting at zero,
	assuming for the moment that subsequent, actually distinct scans 
	never have the same scan #."""

	# **POSSIBLE PROBLEM*** newscan[0] is 0, newscan[1] is 1 then normal...

	nrec=scan.size
	# keep track of the actual scan #.
	thisscan=scan[0]
	# new, nominal scan # - 
	currscan=0
	scan[0]=0
	for ii in range(nrec-1):
		i=ii+1
		newscan=scan[i]
		#print i,newscan,thisscan,currscan
		#raw_input("Enter to continue: ")
		if (newscan != thisscan):
			currscan += 1
			thisscan = scan[i]
		scan[i]=currscan

	return 0

def classifyscans(scan,xx,yy):
	"""
	Classify and tabulate supporting information about scans
	described by 'scans,xx,yy' (identical sized numpy nd-arrays).
	returns scaninfo.dir = 0 -> scan is in xx direction; 1-> scan is in yy direciton.
	you should tirm zero weight data out before calling this...
	"""
	scanlist=np.unique(scan)
	nscans=scanlist.size

	zi=np.zeros(nscans,dtype=np.int32)
	zf=np.zeros(nscans,dtype=np.float64)
	scaninfo = {'dir': np.copy(zi), 'smax': np.copy(zf), 'smean': np.copy(zf), 'smin': np.copy(zf), 
		'xmax': np.copy(zf), 'xmean': np.copy(zf), 'xmin': np.copy(zf), 'nrows': 0, 'ncols':0}

	for i in range(nscans):
		print 'Scan ',scanlist[i]
		gi = (scan==scanlist[i])
		xrms=mad(xx[gi])
		yrms=mad(yy[gi])
		if (xrms > yrms):
			isdeclat=0
		else:
			isdeclat=1
		if (isdeclat):
			scaninfo['dir'][i] = 1
			# max/min/mean in the scan direction
			scaninfo['smax'][i]=np.max(yy[gi])
			scaninfo['smin'][i]=np.min(yy[gi])
			scaninfo['smean'][i]=np.median(yy[gi])
			# max/min/mean in the cross-scan direction
			scaninfo['xmax'][i]=np.max(xx[gi])
			scaninfo['xmin'][i]=np.min(xx[gi])
			scaninfo['xmean'][i]=np.median(xx[gi])
		else:
			# same for ra/long scan --
			scaninfo['dir'][i] = 0
			scaninfo['smax'][i]=np.max(xx[gi])
			scaninfo['smin'][i]=np.min(xx[gi])
			scaninfo['smean'][i]=np.median(xx[gi])
			scaninfo['xmax'][i]=np.max(yy[gi])
			scaninfo['xmin'][i]=np.min(yy[gi])
			scaninfo['xmean'][i]=np.median(yy[gi])
	gi= (scaninfo['dir'] == 0)
	rows=scanlist[gi]
	scaninfo['nrows'] = rows.size
	gi= (scaninfo['dir'] == 1)
	cols=scanlist[gi]
	scaninfo['ncols']= cols.size

	return scaninfo

def getcrossings(xx,yy,scan,scaninfo):
	"""
	Find crossings. Return dictionary xinfo with
	nx = list of ~nrows*ncols scans; 
	rint = scan # of the row (of the crossings);
	cint = scan # of the col. (of the crossings); 
	xint = xcoordinates of crossings
	yint = ycoordinates of crossings
	rtx = time of crossings for the row scan involved in the crossings
	ctx = time of crossings for the col scan involved in the crossings 
 	  NB: (rtx,ctx ) in IDL code were populated by MAKEDELTAVEC() not this routine
 	  not sure if there was a good reason for that ...
	"""

	# scan_list is of length NSCANS and has values = scan # of row or col scan --
	scan_list = np.arange(scaninfo['dir'].size)
	rlist = scan_list[(scaninfo['dir'] == 0)]
	clist = scan_list[(scaninfo['dir'] == 1)]

	nrow=rlist.size
	ncol=clist.size

	# list of scan #s of row and column for each intersection (to be built in the loop)
	rint=np.zeros(0)
	cint=np.zeros(0)
	# list of x and y coordinates at each intersection (to be built in the loop)
	xint=np.zeros(0)
	yint=np.zeros(0)

	xmin=np.min(xx)
	ymin=np.min(yy)
	xmax=np.max(xx)
	ymax=np.max(yy)
	firstplot=0

	cc=0
	# count crossings - 
	for i in range(ncol):
		# this is the median xross-scan coordinate value for this column scan -->
		tcolloc=scaninfo['xmean'][clist[i]]
		gic= (scan == clist[i]) 
		# trajectory of this scan
		xc=xx[gic]
		yc=yy[gic]
		# CFIT - for now this is a straight linear fit.
		#  if needed could add a round of robust outlier rejection & refit.
		cfit_coeff = np.polyfit(yc,xc,1)
		cfit_fn = np.poly1d(cfit_coeff)
		xcf=cfit_fn(yc)
		# (x1,y1) , (x2,y2) -> coordinates of
		#  beginning and end of the fit to the column trajectory.
		# if you add iteratively reject be careful with the following indices-->
		x1=xcf[0]
		y1=yc[0]
		x2=xcf[yc.size-1]
		y2=yc[yc.size-1]
		# now loop over rows. find rows that encompass this columns' central x location.
		#  increment crossing, append colnum and row scan #s to list of column and row scan #s.
		for j in range(nrow):
			#print i,j,rlist[j],scaninfo['smax'][rlist[j]], tcolloc, scaninfo['smin'][rlist[j]] 
			#raw_input("Enter to continue: ")
			if ( (scaninfo['smax'][rlist[j]] >= tcolloc) & (scaninfo['smin'][rlist[j]] <= tcolloc) ):
				if ((i % 25 == 0) & (j % 25 == 0)):
					print '***',i,j,clist[i],rlist[j]
				cc += 1
				cint=np.append(cint,clist[i])
				rint=np.append(rint,rlist[j])
				gir= (scan==rlist[j])
				# x & y trajectories for this row scan...
				xr=xx[gir]
				yr=yy[gir]
				rfit_coeff = np.polyfit(xr,yr,1)
				rfit_fn = np.poly1d(rfit_coeff)
				yrf=rfit_fn(xr)
				# coordinates of end of row scan -
				x3=xr[0]
				y3=yrf[0]
				x4=xr[xr.size-1]
				y4=yrf[xr.size-1]
				# expression for line-line interesction point -- 
				txint = ( ( (x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)))
				tyint = ( ( (x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)))
				xint = np.append(xint,txint)
				yint = np.append(yint,tyint)
				# make plots to test - 
				#pyl.plot(xc,yc,'b')
				#pyl.plot(xr,yr,'r')
				#pyl.plot(xcf,yc,'.')
				#pyl.plot(xr,yrf,'--')
				#pyl.plot((txint),(tyint),'*g')
				#pyl.show()
				#raw_input("Enter to continue")
				# --> is sort of kludgey. you can also plot the resulting xint,yint (easier, less diagnostic)
		# end loop over rows
	# end loop over columns
	allscans = np.unique(np.concatenate((rint,cint),0))
	nix=rint.size

	xinfo = { 'nx': cc, # intersections
		'rint': rint, # array of NX scan -- scan # of row for each IX
		'cint': cint, #   " col
		'xint': xint, 'yint': yint, 'rtx': np.zeros(nix), 'ctx': np.zeros(nix),'scans':allscans}

	return xinfo

def weavebasket(xinfo, npoly=0):
	"""
	Make design matrix of the problem given information in "xinfo" dictionary
	"""
	nscans=xinfo['scans'].size
	nix=xinfo['nx']
	nterms=npoly+1
	# NB - swapping R&C relative to IDL->
	basket=np.zeros([nix,nscans*nterms])
	for i in np.arange(nix):
		for j in np.arange(nterms):
			basket[i,xinfo['rint'][i]*nterms+j] = 1.0*sps.eval_legendre(j,xinfo['rtx'][i])
			basket[i,xinfo['cint'][i]*nterms+j] = -1.0*sps.eval_legendre(j,xinfo['ctx'][i])
			# translation -
			#  design_matrix[ scan & term index, intersection ] = +/- legendre_value(order j , time_at_crossing)
			#  + or - determined by whether it's a row or a column. rows by fiat are +ve in the model. 
			#  (that is determined by makedeltavec(), which does rowdata - columndata)
	return basket

def makedeltavec(scan,time,xx,yy,data,data_wt,xinfo):
	"""
	note time is assumed in seconds already ... if that matters.
	"""
	nx=xinfo['nx']

	dd=np.zeros(0)
	ww=np.zeros(0)

	for i in np.arange(nx):
		if ((i % 1000 ) == 0):
			print i
		rowind= (scan == xinfo['rint'][i])
		colind= (scan == xinfo['cint'][i])
		xrow=xx[rowind]
		yrow=yy[rowind]
		xcol=xx[colind]
		ycol=yy[colind]
		rdat=data[rowind]
		cdat=data[colind]
		rtime= time[rowind]
		ctime= time[colind]
		rdist2=((xrow-xinfo['xint'][i])**2 + (yrow-xinfo['yint'][i])**2)
		cdist2=((xcol-xinfo['xint'][i])**2 + (ycol-xinfo['yint'][i])**2)
		# rxi indexes the datum in the row scan closest to the intersection-
		rxi= (rdist2 == np.min(rdist2))
		cxi= (cdist2 == np.min(cdist2))
		# "row time": how long into row scan the intersection happens. measured fractionally...?
		xinfo['rtx'][i]  = np.median(rtime[rxi])/max(rtime)
		xinfo['ctx'][i]  = np.median(ctime[cxi])/max(ctime)
		# NOTE - sometimes i get more than one rxi or cxi (equidistant points) -- need to check
		#  i've done the right thing rather than just averaging over them...
		#
		# difference data
		#print rdat[rxi]
		#print cdat[cxi]
		#raw_input("Press ENTER!!!!")
		dd = np.append(dd,np.mean(rdat[rxi])-np.mean(cdat[cxi]))
		rwt=np.mean((data_wt[rowind])[rxi])
		cwt=np.mean((data_wt[colind])[cxi])
		if rwt < 0.0:
			rwt=0.0
		if cwt < 0.0:
			cwt=0.0
		if ( (rwt == 0) & (cwt == 0)):
			this_wt = 0.0
		else:
			this_wt = (rwt*cwt)/(rwt+cwt)
		ww=np.append(ww,this_wt)
	# end loop over intersection
	delta_vec = {'delta': dd, 'wt': ww}

	return delta_vec
