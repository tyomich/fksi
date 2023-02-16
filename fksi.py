import numpy as np
from scipy.special import factorial as fac
from scipy.special import j0, jv

def sc(xd, fd, o=1):
    '''
    sc(xd, fd, o=1) - computes interpolation coefficients for function f(x) and its first derivative
    given values fd sampled at positions xd (need not be uniform but assumed to be sorted either way).
    The interpolation is a spline of order o such that the function and its derivatives up to order o-1
    are continuous across sampling points; the first segment is linear. Returns polynomial coefficients.
    '''
    c=np.zeros(shape=(len(xd), o+1))
    d=np.zeros(shape=(len(xd)+1, o+2))
    d[:-1, 0]=fd
    d[1:-1, 1]=fd[:-1]
    for i in range(1, len(xd)):
        l, r=xd[i-1], xd[i]
        A=np.zeros(shape=(o+1, o+1))
        A[0, :]=np.array([r**n for n in range(o+1)])
        for nn in range(o):
            A[nn+1, nn:]=[fac(n)/fac(n-nn)*l**(n-nn) for n in range(nn, o+1)]
        c[i,:]=np.linalg.solve(A, d[i, :-1])
        B=np.zeros(shape=(o+2, o+1))
        B[0, :]=np.array([l**n for n in range(o+1)])
        for nn in range(o+1):
            B[nn+1, nn:]=[fac(n)/fac(n-nn)*r**(n-nn) for n in range(nn, o+1)]
        d[i+1,2:]=np.dot(B, c[i,:])[2:]
    return c#, d

def cderu(cf, a, b):
    '''
    cderu(cf, a, b) - computes maxima of absolute values of derivatives up to the fourth order for phase
    functions given by up to fourth order polynomials with coefficients cf_i on intervals (a_i, b_i).
    '''
    cd=np.zeros(shape=(cf.shape[0], 4))
    cd[:, :cf.shape[1]-1]=cf[:,1:]
    f4=24*np.abs(cd[:, 3])
    x3=np.array([a,b])
    f3=np.max(np.abs(6*(cd[:, 2]+4*cd[:,3]*x3)), axis=0)
    x2=np.array([a,b,a])
    x2i=-cd[:,2]/(4*cd[:,3])
    l,u=x3.min(axis=0), x3.max(axis=0)
    mask=np.logical_not(np.isnan(x2i))&np.isreal(x2i)&(l<x2i)&(x2i<u)
    x2[2, mask]=x2i[mask]
    f2=np.max(np.abs(2*cd[:,1]+6*x2*(cd[:, 2]+x2*2*cd[:,3])), axis=0)
    x1=np.array([a,b,a,a])
    sd=np.sqrt(9*cd[:,2]*cd[:,2]-24*cd[:,1]*cd[:,3])/(12*cd[:,3])
    mask=mask&np.isnan(sd)&np.isreal(sd)
    x1l, x1u=x2i-sd, x2i+sd
    maskl=mask&(l<x1l)&(x1l<u)
    x1[2, maskl]=x1l[maskl]
    masku=mask&(l<x1u)&(x1u<u)
    x1[3, masku]=x1u[masku]
    f1=np.max(np.abs(cd[:,0]+x1*(2*cd[:,1]+x1*(3*cd[:, 2]+x1*4*cd[:,3]))), axis=0)
    return np.array([f1, f2, f3, f4])

def fdu(fdsu, a, b, y, w):
    '''
    fdu(fdsu, a, b, y, w) - computes a bound on the maximum of the absolute values of the fourth derivative 
    of the Fresnel-Kirchhoff integrand defined in in Suvorov+ 2023 on intervals (a_i, b_i) given maxima fdsu 
    of the four derivatives of the phase function, source position y and frequency w.
    '''
    f1,f2,f3,f4=fdsu
    u=np.abs(np.array([a,b])).max(axis=0)
    u=np.tensordot(u, np.ones_like(w), axes=((),()))
    u2=u*u
    y2=y*y
    w2=w*w
    k0=2*w2*(w2*(u+4*y)*u2*u2+2*w*(5+3*w*y2)*u2*u+4*w*y*(6+w*y2)*u2+(w2*y2*y2+18*w*y2+15)*u+4*y*(3+w*y2))
    k1=4*w2*(3+w2*(u+y)**3+3*w*(2*u2+3*y*u+y2))
    k2=6*w2*(w*u2*u+2*y+2*w*y*u2+(3+w*y2)*u)
    k11=w*k2
    k111=4*w2*w*(1+w*(y+u)*u)
    k1111=w2*w2*u
    k12=12*w2*(1+w*(y+u)*u)
    k112=6*w2*w*u
    k22=3*w2*u
    k3=4*w*(1+w*(y+u)*u)
    k13=4*w2*u
    k4=w*u
    return (k0+f1*(k1+f1*(k11+f1*(k111+f1*k1111)))+f2*(k2+f1*(k12+f1*k112)+f2*k22)+f3*(k3+f1*k13)+k4*f4)

def fksi(y, ws2023, xd, fds2023, et=2e-17, o=1, maxev=np.infty, diagnose=False, ftype=float, ctype=complex):
    '''
    fksi(y, w, xd, fd, et=2e-17, o=1, maxev=1e+8, diagnose=False, ftype=float, ctype=complex) - estimates the value
    of the Fresnel-Kirchhoff integral for an observer at position y and frequency 2, as defined in Suvorov+2023,
    given phase values fd sampled at positions xd (need not be uniform but assumed sorted either way) 
    spline-interpolated to order o (i.e., continuously and smoothly to order o-1 across phase sampling points)
    to a requested precision of et using Simpson's method. If diagnose, would print the number of interval subdivisions used,
    as well as partial sums and other internal values. Unless diagnose, would not attempt to compute the integral 
    if the subdivision is so fine that the integrand would have to be computed at more than maxev number of points.
    ftype and ctype can be used to specify the type of arguments passed to np.exp and scipy.special.j0
    '''
    w=2*ws2023#according to frequency definition adopted in Suvorov+2023
    fd=fds2023/2#phase scaled consistently with the definition in Suvorov+2023
    c=sc(xd, fd, o)
    cf, a, b=c[1:], xd[:-1], xd[1:]
    ##
    fdsu=cderu(cf, a, b)
    h=b-a
    erfsu=fdu(fdsu, a, b, y, w)*np.abs(h)**5/180
    nis=np.array(np.ceil(((erfsu*(len(xd)-1)/et)**0.25-1)/2)*2+1, dtype=int)#et/(len(xd)-1) assumes uniformly distributed error contribution - this could have been optimised relative to the computation cost but the optimisation cost itself seems to be too high
    if diagnose:
        print(np.sum(np.abs(nis)), nis)
    else:
        if np.sum(np.abs(nis))>maxev:
            return
    ##
    lo=cf.shape[-1]
    no=np.ones(lo)
    ao=np.arange(lo)
    xcs=np.array([np.linspace(xa, xb, ni+2)[1:-1] for xa,xb,ni in zip(a,b,nis)], dtype=object)
    xcns=np.array([np.tensordot(xc, no, axes=0)**ao for xc in xcs], dtype=object)
    ##
    fcs=np.array([np.sum(cs*xcn, axis=-1) for cs, xcn in zip(cf, xcns)], dtype=object)
    ##
    effcs=np.array([np.exp(-1j*w*fc.astype(ctype)) for fc in fcs], dtype=object) #used to work without any .astype() but ceased to
    j0cs=np.array([j0(w*y*xc.astype(ftype)) for xc in xcs], dtype=object) #.astype()
    efrcs=np.array([np.exp(.5j*w*(xc*xc).astype(ctype)) for xc in xcs], dtype=object)#.astype()
    difcs=xcs*j0cs*efrcs*(effcs-1)
    odds=np.array([dif[0::2] for dif in difcs], dtype=object)
    evens=np.array([dif[1::2] for dif in difcs], dtype=object)
    lods=np.array([len(odd) for odd in odds])
    levs=np.array([len(even) for even in evens])
    j0d=j0(w*y*xd)
    effd=np.exp(-1j*w*fd)
    efrd=np.exp(.5j*w*(xd*xd))
    difd=xd*j0d*efrd*(effd-1)
    qds=(difd[:-1]+difd[1:]+np.array([4*np.sum(odd)+2*np.sum(even) for odd, even in zip(odds, evens)]))*h/(2+4*lods+2*levs)
    if diagnose:
        print(np.array([qds, difd[:-1], difd[1:], 
              np.array([4*np.sum(odd) for odd in odds]),
              np.array([4*np.sum(even) for even in evens]),
              h, lods, levs]).T)
    return 1-1j*w*np.exp(1j*w*y*y/2)*np.sum(qds)