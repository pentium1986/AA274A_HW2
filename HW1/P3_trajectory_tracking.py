import numpy as np
from numpy import linalg

V_PREV_THRES = 0.0001

class TrajectoryTracker:
    """ Trajectory tracking controller using differential flatness """
    def __init__(self, kpx, kpy, kdx, kdy, V_max=0.5, om_max=1):
        self.kpx = kpx
        self.kpy = kpy
        self.kdx = kdx
        self.kdy = kdy

        self.V_max = V_max
        self.om_max = om_max

        self.coeffs = np.zeros(8) # Polynomial coefficients for x(t) and y(t) as
                                  # returned by the differential flatness code

    def reset(self):
        self.V_prev = 0
        self.om_prev = 0
        self.t_prev = 0

    def load_traj(self, times, traj):
        """ Loads in a new trajectory to follow, and resets the time """
        self.reset()
        self.traj_times = times
        self.traj = traj

    def get_desired_state(self, t):
        """
        Input:
            t: Current time
        Output:
            x_d, xd_d, xdd_d, y_d, yd_d, ydd_d: Desired state and derivatives
                at time t according to self.coeffs
        """
        x_d = np.interp(t,self.traj_times,self.traj[:,0])
        y_d = np.interp(t,self.traj_times,self.traj[:,1])
        xd_d = np.interp(t,self.traj_times,self.traj[:,3])
        yd_d = np.interp(t,self.traj_times,self.traj[:,4])
        xdd_d = np.interp(t,self.traj_times,self.traj[:,5])
        ydd_d = np.interp(t,self.traj_times,self.traj[:,6])
        
        return x_d, xd_d, xdd_d, y_d, yd_d, ydd_d

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time
        Outputs: 
            V, om: Control actions
        """

        dt = t - self.t_prev
        x_d, xd_d, xdd_d, y_d, yd_d, ydd_d = self.get_desired_state(t)

        ########## Code starts here ##########
        # Reset to nominal velocity if V is too small.
        if self.V_prev >= V_PREV_THRES: 
            V_p = self.V_prev
        else:
            V_p = np.sqrt(xd_d**2 + yd_d**2)
        
        xd = V_p * np.cos(th)
        yd = V_p * np.sin(th)
        u1 = xdd_d + self.kpx * (x_d - x) + self.kdx * (xd_d - xd)
        u2 = ydd_d + self.kpy * (y_d - y) + self.kdy * (yd_d - yd)
        
        J = np.array([[np.cos(th), -V_p * np.sin(th)], [np.sin(th), V_p * np.cos(th)]])
        u = np.array([u1, u2])
        a, om = np.linalg.solve(J, u)
        V = V_p + a * dt
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        return V, om