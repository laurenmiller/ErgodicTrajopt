from utils import *
from numpy import dot
from numpy.linalg import pinv as inverse
from scipy.integrate import trapz
from scipy.integrate import ode, quad, odeint
from scipy.interpolate import interp1d



class ProjectionBasedOpt(object):
    def __init__(self, nx, nu, 
                R=.05, Q=0, xref=None):
        '''
        Class to represent an optimization problem for a system with dynamic constraints.
        Assume objective function of the form J = \int_0^T (x-xref)'.Q.(x-xref)+(u)'.R.(u) dt
        :nx dimension of state
        :nu dimension of the control
        :R, Q: are Weights in objective function. Can either be matrices or scalars, which we assume means weight*I
        :xref reference trajectory. xref can only be a single point, e.g. [0,0] needs to update code to take a trajectory
        '''

        self.nx=nx
        self.nu=nu

        if xref==None:
            self.xref=[0]*nx
        elif len(xref)!=nx:
            raise Exception("dimension of xref does not match dimension of state space")
        else:
            self.xref=xref

        # check the weight for the state
        print type(Q)
        print 'qing',Q
        if (type(Q)=='int') or (type(Q)=='float'):
            self.Q=Q*np.eye(self.nx)
            print Q
        self.Q=Q*np.eye(self.nx)
        #elif type(Q)=='list' and Q.shape==[self.nx,self.nx]:
        #   self.Q=Q
        #else:
        #    raise Exception("dimension ofe Q does not match dimension of state space")
        # self.Q=Q*np.eye(self.nx)

        # check the weight for the control
        if type(R)=='int' or type(R)=='float':
            self.R=R*np.eye(self.nu)
        self.R=R*np.eye(self.nu)
        #elif R.shape==[self.nu,self.nu]:
        #    self.R=R
        #else:
        #    raise Exception("dimension of R does not match dimension of input space")

        self.P1=np.eye(self.nx) 
        self.Qn=np.eye(self.nx)
        self.Rn=np.eye(self.nu)
        self.Qk=np.eye(self.nx)
        self.Rk=np.eye(self.nu)
        
    def peqns(self,t,pp,Al,Bl,Rn,Qn):
        pp=pp.reshape(self.nx,self.nx)
        matdiffeq=(matmult(pp,Al(t)) + matmult(Al(t).T,pp) -
                   matmult(pp,Bl(t),inverse(Rn),Bl(t).T,pp) + Qn)
        return matdiffeq.flatten()
 
    def reqns(self,t,rr,Al,Bl,a,b,Psol,Rn,Qn):
        t=self.time[-1]-t
        matdiffeq=(matmult(Al(t)-matmult(Bl(t),inverse(Rn),Bl(t).T,Psol(t)).T,rr.T)
                   +a(t)-matmult(Psol(t),Bl(t),inverse(Rn),b(t))) 
        return matdiffeq.flatten()
 
    def veqns(self,zz,Al,Bl,a,b,Psol,Rsol,Rn,Qn):
        vmatdiffeq=(matmult(-inverse(Rn),Bl.T,Psol,zz) - matmult(inverse(Rn),Bl.T,Rsol) -
                   matmult(inverse(Rn),b))
        return vmatdiffeq
     
    def zeqns(self,t,zz,Al,Bl,a,b,Psol,Rsol,Rn,Qn):
        vmateq=self.veqns(zz,Al(t),Bl(t),a(t),b(t),Psol(t),Rsol(t),Rn,Qn)
        matdiffeq=matmult(Al(t),zz) + matmult(Bl(t),vmateq)
        return matdiffeq.flatten()
   
    def Ksol(self, X, U):
        time=self.time
        P1 = np.eye(X.shape[1]).flatten()
        solver = ode(self.peqns).set_integrator('dopri5')
        solver.set_initial_value(P1,time[0]).set_f_params(self.A_interp,
                                                          self.B_interp,
                                                          self.Rk,
                                                          self.Qk)
        k = 0
        t=time
        soln = [P1]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)
 
        # Convert the list to a numpy array.
        psoln = np.array(soln).reshape(time.shape[0],X.shape[1],X.shape[1])
        K=np.empty((time.shape[0],X.shape[1],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            K[tindex,:,:]=matmult(inverse(self.Rk),self.B_current[tindex].T,psoln[tindex])
        self.K=K
        return K
 
    def Psol(self, X, U, time):
 
        P1 = np.eye(X.shape[1]).flatten()
        solver = ode(self.peqns).set_integrator('dopri5')
        solver.set_initial_value(P1,time[0]).set_f_params(self.A_interp,
            self.B_interp, self.Rn, self.Qn)
        k = 0
        t=time
        soln = [P1]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)
 
        soln = np.array(soln)
        return soln.reshape(time.shape[0],X.shape[1],X.shape[1])
 
    def Rsol(self,X, U,P_interp,time):
        rinit2 = np.zeros(X.shape[1])
        Qn = np.eye(X.shape[1])
        Rn = np.eye(U.shape[1])
        solver = ode(self.reqns).set_integrator('dopri5')
        solver.set_initial_value(rinit2,time[0]).set_f_params(self.A_interp,
            self.B_interp,self.a_interp,self.b_interp,P_interp, Rn,Qn)
        k =0
        t=time
        soln = [rinit2]
        while solver.successful() and solver.t < t[-1]:# 
            k +=1
            solver.integrate(t[k])
            soln.append(solver.y)
            
        soln.reverse()
        soln = np.array(soln)
        return soln

    # pointwise dynamics linearizations
    def fofx_pointwise(self,X,U):
        return U

    def fofx(self,t,X,U):
        return U(t)

    def dfdx_pointwise(self,x,u):
        xdim=x.shape[0]
        return np.zeros((xdim,xdim))

    def dfdx(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dfdxl=np.empty((time.shape[0],X.shape[1],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dfdxl[tindex,:,:]=self.dfdx_pointwise(X[tindex],U[tindex])
        self.A_current=dfdxl
        return dfdxl

    def dfdu_pointwise(self,x,u):
        udim=u.shape[0]
        return np.eye(udim)

    def dfdu(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dfdul=np.empty((time.shape[0],U.shape[1],U.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dfdul[tindex,:,:]=self.dfdu_pointwise(X[tindex],U[tindex])
        self.B_current=dfdul
        return dfdul

    def cost_pointwise(self,x,u):
        R=self.R
        Q=self.Q
        return .5*matmult(u.T,R,u)+.5*matmult((x-self.xref).T,Q,(x-self.xref))
    
    def cost(self,X,U):
        cost=np.empty(self.time.shape[0])
        for tindex,t in np.ndenumerate(self.time):
            cost[tindex]=self.cost_pointwise(X[tindex],U[tindex])
        return trapz(cost,self.time)#+.5*matmult((X[tindex]-self.xref).T,K,(X[tindex]-self.xref))

    def eval_cost(self):
        # return the evaluated cost function
        return self.cost(self.X_current,self.U_current)
            
    def dldu_pointwise(self,x,u):
        # return the pointwise linearized cost WRT state
        return matmult(self.R,u)
    
    def dldx_pointwise(self,x,u):
        # return pointwise linearized cost WRT input
        return matmult(self.Q,x-self.xref)  

    def dldx(self):
        # evaluate linearized cost WRT state
        X=self.X_current
        U=self.U_current
        time=self.time
        dldxl=np.empty((time.shape[0],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dldxl[tindex,:]=self.dldx_pointwise(X[tindex],U[tindex])
        self.a_current=dldxl  #
        return self.a_current

    def dldu(self):
        # evaluate linearized cost WRT input
        X=self.X_current
        U=self.U_current
        time=self.time
        dldul=np.empty((time.shape[0],U.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dldul[tindex,:]=self.dldu_pointwise(X[tindex],U[tindex])
        self.b_current=dldul
        return dldul

    def dcost(self,descdir):
        # evaluate directional derivative
        dX=descdir[0]
        dU=descdir[1]
        time=self.time
        dc=np.empty(time.shape[0])
        for tindex,t in np.ndenumerate(time):
            dc[tindex]=matmult(self.a_current[tindex],dX[tindex])+matmult(self.b_current[tindex],dU[tindex])
        intdcost=trapz(dc,time)
        return intdcost
    
    def descentdirection(self):
        # solve for the descent direction by 
        X=self.X_current
        U=self.X_current
        time=self.time

        Ps=self.Psol(X, U, time)
        self.P_current=Ps
        P_interp = interp1d(time, Ps.T)

        Rs=self.Rsol(X, U, P_interp,time)
        self.R_current=Rs

        r_interp = interp1d(time, Rs.T)

        zinit = np.zeros(X.shape[1])
        #initialize the 4th order Runge-Kutta solver
        solver = ode(self.zeqns).set_integrator('dopri5')
        #initial value
        solver.set_initial_value(zinit,time[0]).set_f_params(self.A_interp, self.B_interp,
                                                            self.a_interp, self.b_interp,
                                                            P_interp, r_interp,
                                                            self.Rn, self.Qn)
        k = 0
        t=time
        zsoln = [zinit]
        while solver.successful() and solver.t < t[-1]:
           k += 1
           solver.integrate(t[k])
           zsoln.append(solver.y)

        #Convert the list to a numpy array.
        zsoln = np.array(zsoln)
        zsoln=zsoln.reshape(time.shape[0],X.shape[1])
        vsoln=np.empty(U.shape)
        for tindex,t in np.ndenumerate(time):
            vsoln[tindex]=self.veqns(zsoln[tindex],self.A_current[tindex],
                                     self.B_current[tindex],self.a_current[tindex],
                                     self.b_current[tindex],Ps[tindex],Rs[tindex],self.Rn,self.Qn)
        return np.array([zsoln,vsoln])
    
    def simulate(self,X0,U,time):
        U_interp = interp1d(time, U.T)
        # # initialize the 4th order Runge-Kutta solver
        solver = ode(self.fofx).set_integrator('dopri5')
        # # initial value
        solver.set_initial_value(X0,time[0]).set_f_params(U_interp)
        #ppsol = odeint(pkeqns,P1,time,args=(A_interp,B_interp))
        k = 0
        t=time
        xsoln = [X0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            xsoln.append(solver.y)

        # Convert the list to a numpy array.
        xsoln = np.array(xsoln)
        return xsoln
        
    def proj(self,t,X,K,mu,alpha):
        # print U(t)
        # print K(t)
        # print alpha(t)
        uloc =mu(t) +  matmult(K(t),(alpha(t).T - X.T))
        self.fofx_pointwise(X,uloc)
        return uloc

    def projcontrol(self,X,K,mu,alpha):
        uloc = mu +  matmult(K,(alpha.T - X.T))
        return uloc

    def project(self,X0,traj,time):
        alpha=traj[0]
        mu=traj[1]

        #solve for riccatti gain
        Ks=self.Ksol(alpha, mu)
        K_interp = interp1d(time, Ks.T)
        mu_interp = interp1d(time, mu.T)
        alpha_interp = interp1d(time, alpha.T)
        solver = ode(self.proj).set_integrator('dopri5')
        # # initial value
        solver.set_initial_value(X0,time[0]).set_f_params(K_interp,mu_interp,alpha_interp)
        #ppsol = odeint(pkeqns,P1,time,args=(A_interp,B_interp))
        k = 0
        t=time
        soln = [X0]
        while solver.successful() and solver.t < t[-1]:
            k += 1
            solver.integrate(t[k])
            soln.append(solver.y)

        # Convert the list to a numpy array.
        xsoln = np.array(soln)
        usoln=np.empty(mu.shape)
        for tindex,t in np.ndenumerate(time):
            usoln[tindex,:]=self.projcontrol(xsoln[tindex],Ks[tindex],mu[tindex],alpha[tindex])
        return np.array([xsoln,usoln])
        
    def update_traj(self,X,U,time):
        self.time=time
        self.X_current=X
        self.U_current=U
        self.dfdx()
        self.dfdu()
        self.dldx()
        self.dldu()
        self.A_interp = interp1d(time, self.A_current.T)
        self.B_interp = interp1d(time, self.B_current.T)
        self.a_interp = interp1d(time, self.a_current.T)
        self.b_interp = interp1d(time, self.b_current.T)


class ErgodicOpt(ProjectionBasedOpt):

    def __init__(self, nx, nu, Nfourier=10, res = 400,  
                wlimit=float(1), dimw=2, barrcost=1, R=.05, Q=0, ergcost=10):

        super(ErgodicOpt,self).__init__(nx, nu, R=R, Q=Q) 

        self.barrcost=barrcost
        self.ergcost=ergcost
        self.Nfourier=Nfourier

        self.dimw = dimw # workspace dimension. current implementation only works for 2D
        self.wlimit = wlimit
        self.res = res

        # Check that the number of coefficients matches the workspace dimension
        if type(self.wlimit) == float:
            self.wlimit = [self.wlimit] * self.dimw
        elif len(self.wlimit) != self.dimw:
            raise Exception(
                "dimension of xmax_workspace \
                does not match dimension of workspace")
        if type(self.Nfourier) == int:
        # if a single number is given, create tuple of length dim_worksapce
            self.Nfourier = (self.Nfourier,) * self.dimw  # if tuples
        elif len(self.Nfourier) != self.dimw:
            raise Exception("dimension of Nfourier \
            does not match dimension of workspace")
                    # self.dimw=wdim
        # setup a grid over the workspace
        #xgrid=[]
        xgrid=[np.linspace(0, wlim, self.res) for wlim in self.wlimit]

        # just this part needs to be fixed for 2d vs 1d vs 3d
        self.xlist = np.meshgrid(xgrid[0], xgrid[1], indexing='ij')

        # set up a grid over the frequency
        klist = [np.arange(kd) for kd in self.Nfourier]
        klist = cartesian(klist)

        # do some ergodic stuff
        s = (float(self.dimw) + 1)/2;
        self.Lambdak = 1/(1 + np.linalg.norm(klist,axis=1)**2)**s
        self.klist = klist/self.wlimit * np.pi
        self.hk=np.zeros(self.Nfourier).flatten()
        for index,val in np.ndenumerate(self.hk):
            hk_interior=1
            for n_dim in range(0,self.dimw):
                integ=quad(lambda x: (np.cos(x*self.klist[index][n_dim]))**2,
                           0, float(self.wlimit[n_dim]))
                hk_interior=hk_interior*integ[0]
            self.hk[index]=np.sqrt(hk_interior)

    def normalize_pdf(self):
        # function to normalize a pdf
        sz=np.prod([n/float(self.res) for n in self.wlimit])
        summed = sz * np.sum(self.pdf.flatten())
        self.pdf= self.pdf/summed      

    def set_pdf(self, pdf):
        # input pdf 
        self.pdf = pdf
        self.normalize_pdf()
        self.calculate_uk(self.pdf)
        pass  

    def calculate_uk(self, pdf):
        # calculate Fourier coefficients of the distribution
        self.pdf=pdf
        self.uk=np.zeros(self.Nfourier).flatten()
        for index,val in np.ndenumerate(self.uk):
            uk_interior=1/self.hk[index] * pdf
            for n_dim in range(0,self.dimw):
                basis_part = np.cos(self.klist[index][n_dim] \
                                    * self.xlist[n_dim])
                uk_interior = self.wlimit[n_dim]/self.res \
                              * uk_interior*basis_part
            self.uk[index] = np.sum(uk_interior) #sum over # XXX:
        pass
    
    def calculate_ergodicity(self):
        # evaluate the ergodic metric (ck, uk, need to be calculated already)
        self.erg = np.sum(self.Lambdak * (self.ck - self.uk)**2)
        return self.erg

    def config_to_workspace(self,X):
        # transformation from statespace to workspace 
        # e.g. from joint angles to XY position on a table top
        XT=X.T
        W=np.array([XT[0],XT[1]])
        return W.T

    def DWDX(self,xk):
        # derivative of the workspace to statespace transformation
        x_workspace=np.array([[1,0,0,0],[0,1,0,0]])
        return (x_workspace).T

    def barrier(self,xk):
        barr_cost=np.zeros(xk.shape[0])
        
        xk=self.config_to_workspace(xk)
        xk=xk.T

        for n in range(0,self.dimw):
            too_big = xk[n][np.where(xk[n]>self.wlimit[n])]
            barr_cost[np.where(xk[n]>self.wlimit[n])]+=np.square(too_big-self.wlimit[n])
            too_small = xk[n][np.where(xk[n]<0)]
            barr_cost[np.where(xk[n]<0)]+=np.square(too_small-0)
            #barr_cost+=(too_small)**2
        barr_cost=trapz(barr_cost,self.time)
        return barr_cost
    
    def Dbarrier(self,xk):
        xk=self.config_to_workspace(xk)
        xk=xk.T
        dbarr_cost=np.zeros(xk.shape)
        # for n in range(0,self.dimw):
        #     if xk[n]>self.wlimit[n]:
        #         dbarr_cost[n]=2*(xk[n]-self.wlimit[n])
        #     if xk[n]<0:
        #         dbarr_cost[n]=2*(xk[n]-0)
        # dbarr_cost=np.zeros(xk.shape[0])
        for n in range(0,self.dimw):
            too_big = xk[n][np.where(xk[n]>self.wlimit[n])]
            dbarr_cost[n,np.where(xk[n]>self.wlimit[n])]=2*(too_big-self.wlimit[n])
            too_small = xk[n][np.where(xk[n]<0)]
            dbarr_cost[n,np.where(xk[n]<0)]=2*(too_small-0)
            #barr_cost+=(too_small)**2
        return dbarr_cost.T
    
    def ckeval(self):
        X=self.X_current
        time=self.time
        T=time[-1]
        # change coordinates from configuration to ergodic workspace
        W = self.config_to_workspace(X).T
        self.ck=np.zeros(self.Nfourier).flatten()
        #xlist = tj.T
        for index,val in np.ndenumerate(self.ck):
            ck_interior=1/self.hk[index]* 1/(float(T))
            for n_dim in range(0,self.dimw):
                basis_part = np.cos(self.klist[index][n_dim] * W[n_dim])
                ck_interior = ck_interior*basis_part
                self.ck[index]=trapz(ck_interior,time)#np.sum(self.dt*ck_interior)

    def akeval(self):
        X=self.X_current
        time=self.time
        T=time[-1]
        xlist = X.T
        outerchain = 2 * 1/self.hk * 1/(float(T)) * self.Lambdak \
                     * (self.ck-self.uk)
        x_in_w=self.config_to_workspace(X).T
        ak = []
        for index,val in np.ndenumerate(outerchain):
            DcDX = []
            # these are chain rule terms, get added
            for config_dim in range(0,self.nx):
                Dcdx=0
                for term_dim in range(0,self.dimw):
                    term=outerchain[index]
                    for prod_dim in range(0,self.dimw):
                        if term_dim == prod_dim:
                            basis_part=-self.klist[index][prod_dim]*np.sin(
                                self.klist[index][prod_dim] * x_in_w[prod_dim])\
                                *self.DWDX(xlist)[config_dim][prod_dim]
                        else:
                            basis_part = np.cos(self.klist[index][prod_dim] \
                                            * x_in_w[prod_dim])
                        term*=basis_part
                    Dcdx=Dcdx+term
                DcDX.append(Dcdx)
            ak.append(DcDX)
            
        summed_ak=np.sum(np.array(ak),axis=0)
        
        self.ak = summed_ak.T#self.workspace_to_config(summed_ak).T
        return  self.ak

    def evalcost(self):
        cost=self.cost(self.X_current,self.U_current)
        barr_cost=self.barrcost*self.barrier(self.X_current)
        erg_cost=self.ergcost*self.calculate_ergodicity()
        #print "barrcost=", barr_cost
        #print "contcost=", cont_cost
        #print "ergcost=", erg_cost
        #print "J=", barr_cost+erg_cost+cont_cost
        return barr_cost+erg_cost+cost

    def dldx(self):
        X=self.X_current
        U=self.U_current
        time=self.time
        dldxl=np.empty((time.shape[0],X.shape[1]))
        for tindex,t in np.ndenumerate(time):
            dldxl[tindex,:]=self.dldx_pointwise(X[tindex],U[tindex])
        self.a_current=dldxl+self.ergcost*self.ak+self.barrcost*self.Dbarrier(X)  #
        return self.a_current

    def update_traj(self,X,U,time):
        self.time=time
        self.X_current=X
        self.U_current=U
        self.ckeval()
        self.akeval()
        self.dfdx()
        self.dfdu()
        self.dldx()
        self.dldu()
        self.A_interp = interp1d(time, self.A_current.T)
        self.B_interp = interp1d(time, self.B_current.T)
        self.a_interp = interp1d(time, self.a_current.T)
        self.b_interp = interp1d(time, self.b_current.T)
