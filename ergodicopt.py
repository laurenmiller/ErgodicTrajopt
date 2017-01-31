import time
from matplotlib import cm
import matplotlib.pyplot as plt
import GP 
from lqr_solver import *


def ergoptimize(solver, pdf, state_init,
                control_init=np.array([0,0]),
                t0=0, tf=10, dt=100,
                plot=True,
                maxsteps=20):
    plt.close("all")
    #initialize system
    xdim=2
    udim=2

    solver.set_pdf(pdf)
    time=np.linspace(t0,tf,dt)
    
    #state_init=np.array(state_init)
    U0 = np.array([np.linspace(control_init[0],control_init[0],dt),np.linspace(control_init[1],control_init[1],dt)]).T
    X0 =solver.simulate(state_init,U0,time)
    solver.update_traj(X0,U0,time)

    #set up some containers
    costs=[solver.evalcost()]
    trajlist=[np.array([X0,U0])]
    descdirlist=[]
    dcosts=[]

    #linesearch parameters
    alpha=.001
    beta=.5

    #print "init ck=", solver.ck
    #print "uk=", solver.uk
    #print "init J", costs[0]
    # Setup Plots
    if plot==True:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2,aspect=maxsteps/costs[0])
        ax3 = fig.add_subplot(1, 3, 3,aspect=abs(maxsteps/12.0))
        ax2.set_xlim([0,maxsteps])
        ax2.set_title('J')
        ax2.set_ylim([0,costs[0]])
        ax3.set_xlim([0,maxsteps])
        ax3.set_ylim([-10.0,2])
        ax3.set_title('DJ')
        #pdf=LQ.pdf
        #wlim=LQ.wlimit
        q=solver.X_current.T
        ax1.imshow(np.squeeze(solver.pdf.T),extent=(0, solver.wlimit[0], 0, solver.wlimit[1]),
               cmap=cm.binary, aspect='equal',origin="lower")
        ax1.plot(q[0], q[1], '-c', label='q',lw=1.5)
        ax2.plot(costs, '*', label='q',lw=1.5)
        ax3.plot(dcosts, '-', label='q',lw=1.5)

        plt.draw()
        plt.pause(.01)
    k=0
    for k in range(0,maxsteps,1):
        print "*****************k=", k
        descdir=solver.descentdirection()
        newdcost=solver.dcost(descdir)
        print "DJ=", newdcost
        gamma=1
        newtraj=solver.project(state_init,trajlist[k]+gamma*descdir,time)
        solver.update_traj(newtraj[0],newtraj[1],time)
        newcost=solver.evalcost()
        while newcost > (costs[k] + alpha*gamma*newdcost) and gamma>.00000000001:
            gamma=beta*gamma
            print gamma
            newtraj=solver.project(state_init,trajlist[k]+gamma*descdir,time)
            solver.update_traj(newtraj[0],newtraj[1],time)
            newcost=solver.evalcost()
        print "gamma=", gamma
        print "new J=", newcost
        if plot==True:
            q=newtraj[0].T
            ax1.plot(q[0], q[1],  label='q',
                     lw=1,alpha=0.4)
            ax2.plot(costs, '-k', label='q', lw=1)
            ax2.plot(k,costs[-1], '*', label='q', lw=1)
            ax3.plot(dcosts, '-', label='q',lw=1.5)
            plt.draw()
            plt.pause(.01)
    
        costs.append(newcost)
        descdirlist.append(descdir)
        dcosts.append(np.log10(np.abs(newdcost)))
        trajlist.append(np.array([solver.X_current,solver.U_current]))
    if plot==True:
        q=newtraj[0].T
        ax1.plot(q[0], q[1],  label='q',
                 lw=3,alpha=0.4)
    return solver.X_current

X = .2*np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [0.00000000, -0.50000000],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [-5.03823144, 3.10584743],
              [-2.87600388, 6.74310541],
              [5.21301203, 4.26386883]])

xdim=2
udim=2

LQ=ErgodicOpt(xdim, udim, Nfourier=10, res=100,barrcost=50,R=.1,ergcost=10)

#bimodal_gaussian_pdf(LQ.xlist)

GP=GP.GaussianProcessModel(res=LQ.res)
GP.update_GP(X)
pdf=bimodal_gaussian_pdf(LQ.xlist)#GP.uncertainty

xinit=np.array([0.501,.1])
U0=np.array([0,0])
#pdf=uniform_pdf(LQ.xlist)
trajtotal=np.array([])
for j in range (1,10,1):
    traj=ergoptimize(LQ, pdf, xinit,control_init=U0, maxsteps=20)
    if j>1:
        trajtotal=np.concatenate((trajtotal,traj),axis=0)
    else:
        trajtotal=traj    
    GP.update_GP(trajtotal)
    pdf=GP.uncertainty
    xinit=trajtotal[-1]
