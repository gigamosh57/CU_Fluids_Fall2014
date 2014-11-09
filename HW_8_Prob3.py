#### Page Weil
#### 11/8/14
#### MCEN 5041
#### Homework 8 , PROBLEM 3
#### This ODE Solver code was written as part of my Numerical Methods Coursework (CVEN 5537)

####################################################################
######## INITIALIZE PYTHON
import numpy as np
# For pi
import math
# for plotting
import matplotlib.pyplot as plt
# for quiverplot
from pylab import *
######## 
####################################################################

####################################################################
######## Runge-Kutta ODE solver with Cash-Karp 4th-5th order params

def pagerkck4(feval,x0,tstart,tfinal,dt,order = 4,tol=10**-2,errtol = 10**-20):
   # INITIALIZE ARRAY OF CASH-KARP FACTORS
   ck_a = np.array([[0],[1/5.],[3/10.],[3/5.],[1.],[7/8.]])
   ck_b = np.array([[0,0,0,0,0],
               [1/5.,0,0,0,0],
               [3/40.,9/40.,0,0,0],
               [3/10.,-9/10.,6/5.,0,0],
               [-11/54.,5/2.,-70/27.,35/27.,0],
               [1631/55296.,175/512.,575/13824.,44275/110592.,253/4096.]])
   ck_c =  np.array([[37/378.],[0.],[250/621.],[125/594.],[0.],[512/1771.]])
   ck_cs = np.array([[2825/27648.],[0.],[18575/48384.],[13525/55296.],[277/14336.],[1/4.]])
   
   ck_a = ck_a[range(0,order),:]
   ck_b = ck_b[range(0,order),][:,range(0,order)]
   ck_c = ck_c[range(0,order),:]
   ck_cs = ck_cs[range(0,order),:]
   
   #initialize time vector
   tvec=[tstart]
   #initialize x, initialize solution matrix xsol
   x0 = np.array([x0]).T
   x=x0
   xsol = np.empty((len(x0),0),float)
   xsol = np.append(xsol,x0,axis=1)
   ts = [0]
   
   # SINCE THIS FUNCTION USES ADAPTIVE TIMESTEPPING
   # THE TIME VECTOR IS NOT DEFINED INITIALLY
   t = tstart
   h = dt
   errvec = []
   while t < tfinal:
      
      ## RESET VALUES AND LOOP UNTIL ERROR IS LOW
      error = tol*1.01
      errloop = 0
      while errloop == 0:
         # EVALUATE STAGES
         f = np.empty((0,1),float)
         for fn in feval:
            f = np.append(f,fn(x,t))
         
         k = np.array([h*f]).T
         
         # LOOPS THROUGH ALL K VALUES DEPENDING ON ORDER OF FUNCTION
         for klev in range(1,order):
            # CALCULATES F FOR ALL FUNCTIONS IN FEVAL
            f = np.empty((0,1),float)
            # EVALUATES F DEPENDING ON LEVEL OF K
            for fn in feval:
               cb = np.array([ck_b[klev,range(0,klev)]])
               ca = ck_a[klev]
               kev = k[:,range(0,klev)]
               f = np.append(f,fn(x+np.array([np.sum(cb*kev,axis=1)]).T,t+ca*h))
            
            f = np.array([f]).T
            k = np.append(k,h*f,axis = 1)
            # End for klev
         
         # CALCULATE x values
         xnew = np.sum(x + np.array([np.sum(ck_c.T*k,axis = 1)]).T,axis = 1)
         xs = np.sum(x + np.array([np.sum(ck_cs.T*k,axis=1)]).T , axis = 1)
            
         # ESTIMATE ERROR
         abserror = np.absolute(xnew-xs)
         relerror = np.zeros((len(x0),1))
         for i in range(0,len(xnew)): 
            if xnew[i] > errtol: relerror[i] = np.absolute((xnew[i] - xs[i])/xnew[i])
         
         error = np.amin([np.amax(abserror),np.amax(relerror)])
         
         # ADJUST H BASED ON ERROR
         hnew = h*(tol/error)**(0.2)
         if hnew > h*1.2: hnew=h*1.2
         h = hnew
         if hnew >= h: errloop = 1
         
         #print 'error: ' + str(error)
         ### End while errloop = 0
      
      # UPDATE ALL VALUES FOR NEXT TIMESTEP
      h = hnew
      x = np.reshape(xnew,(-1,1))
      xsol = np.append(xsol,x,axis = 1)
      t = t + h 
      tvec = tvec + [t]
      ts = ts + [h]
      ### End while time < tfinal
         
   return tvec,xsol,ts

######## Solver ends here
################################################


################################################
######## PROBLEM 3
######## Define boundary conditions

feval = [lambda x,t: x[1,0],
         lambda x,t: x[2,0],
         lambda x,t: -1/2*(x[0,0]*x[2,0])]
x0 =  [0,0,1] 
tstart = 0
tfinal = 50
dt = (tfinal-tstart)/10000.
order = 4
tol=10**-4
errtol = 10**-10

######## End input 
################################################

################################################
######## this function is used to find the index of the closest value in an array

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
    
######## End find function
################################################

################################################
######## solve and plot eta vs. f(eta):

tvecf,xsolf,tsf = pagerkck4(feval,x0,tstart,tfinal,dt,tol = tol,order = order,errtol = errtol)

eta = tvecf
feta = xsolf[0,:]
fpeta = xsolf[1,:]

dx = 0.5
xmax = 10
dy = 0.5
ymax = 10
U=1
nu=0.01

### Shows the plot of Eta vs F(feta)
fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.set_xlabel('eta')
axes.set_ylabel('f(eta)')
axes.plot(eta,feta,label = "f(eta)")
axes.legend()
plt.show()

X,Y = meshgrid( arange(dx,xmax,dx),arange(dy,ymax,dy) )
# calculate eta
eta_plot = Y*np.sqrt(U/(2*nu*X))
f_plot = eta_plot*0
fp_plot = f_plot*0
# calculate f and f' on grid based on closest value of eta
for a in range(1,np.shape(eta_plot)[0]):
  for b in range(1,np.shape(eta_plot)[1]):
    B = eta_plot[a,b]
    idx = find_nearest(eta,B)
    f_plot[a,b]=feta[idx]
    fp_plot[a,b]=fpeta[idx]

u = U*fp_plot
v = np.sqrt(nu*U/2*X)*(eta_plot*fp_plot-f_plot)

figure()
Q = quiver( u, v)
#qk = quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$', labelpos='W',fontproperties={'weight': 'bold'})
l,r,b,t = axis()
dx, dy = r-l, t-b
axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])

title('Weathervanes showing U and V velocities at each point')

plt.show() 

######## End solve and plot
################################################

