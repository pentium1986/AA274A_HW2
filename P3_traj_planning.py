import numpy as np
from P1_astar import DetOccupancyGrid2D, AStar
from P2_rrt import *
import scipy.interpolate
import matplotlib.pyplot as plt
from HW1.P1_differential_flatness import *
from HW1.P2_pose_stabilization import *
from HW1.P3_trajectory_tracking import *

class SwitchingController(object):
    """
    Uses one controller to initially track a trajectory, then switches to a 
    second controller to regulate to the final goal.
    """
    def __init__(self, traj_controller, pose_controller, t_before_switch):
        self.traj_controller = traj_controller
        self.pose_controller = pose_controller
        self.t_before_switch = t_before_switch

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            (x,y,th): Current state 
            t: Current time

        Outputs:
            V, om: Control actions
        """
        ########## Code starts here ##########
        if t < self.traj_controller.traj_times[-1] - self.t_before_switch:
            return self.traj_controller.compute_control(x,y,th,t)
        else:
            return self.pose_controller.compute_control(x,y,th,t)
        ########## Code ends here ##########

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    # input is a list of tuples, converts to array
    path = np.array(path)
    x_init = path[:,0]
    y_init = path[:,1]
    t_delta = np.sqrt(np.diff(x_init)**2 + np.diff(y_init)**2)/V_des
    t = np.append([0], np.cumsum(t_delta))
    t_smoothed = np.arange(0, t[-1], dt)
    tck_x = scipy.interpolate.splrep(t, x_init, s=alpha)
    x = scipy.interpolate.splev(t_smoothed, tck_x)
    xd = scipy.interpolate.splev(t_smoothed, tck_x, der=1)
    xdd = scipy.interpolate.splev(t_smoothed, tck_x, der=2)
    tck_y = scipy.interpolate.splrep(t, y_init, s=alpha)
    y = scipy.interpolate.splev(t_smoothed, tck_y)
    yd = scipy.interpolate.splev(t_smoothed, tck_y, der=1)
    ydd = scipy.interpolate.splev(t_smoothed, tck_y, der=2)
    th = np.arctan2(yd, xd)
    traj_smoothed = np.vstack([x,y,th,xd,yd,xdd,ydd]).transpose()
    ########## Code ends here ##########

    return traj_smoothed, t_smoothed

def modify_traj_with_limits(traj, t, V_max, om_max, dt):
    """
    Modifies an existing trajectory to satisfy control limits and
    interpolates for desired timestep.

    Inputs:
        traj (np.array [N,7]): original trajecotry
        t (np.array [N]): original trajectory times
        V_max, om_max (float): control limits
        dt (float): desired timestep
    Outputs:
        t_new (np.array [N_new]) new timepoints spaced dt apart
        V_scaled (np.array [N_new])
        om_scaled (np.array [N_new])
        traj_scaled (np.array [N_new, 7]) new rescaled traj at these timepoints
    Hint: This should almost entirely consist of calling functions from Problem Set 1
    """
    ########## Code starts here ##########
    V,om = compute_controls(traj=traj)
    s = compute_arc_length(V, t)
    V_tilde = rescale_V(V, om, V_max, om_max)
    tau = compute_tau(V_tilde, s)
    om_tilde = rescale_om(V, om, V_tilde)
    s_f = State(x=traj[0,0], y=traj[0,1], V=min(V_tilde[-1],V_max), th=traj[0,2])
    t_new, V_scaled, om_scaled, traj_scaled = interpolate_traj(traj, tau, V_tilde, om_tilde, dt, s_f)
    ########## Code ends here ##########

    return t_new, V_scaled, om_scaled, traj_scaled
