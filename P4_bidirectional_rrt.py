import numpy as np
import matplotlib.pyplot as plt
from dubins import path_length, path_sample
from utils import plot_line_segments, line_line_intersection

# Represents a motion planning problem to be solved using the RRT algorithm
class RRTConnect(object):

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRTConnect")

    def find_nearest_forward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering forward from x
        """
        raise NotImplementedError("find_nearest_forward must be overriden by a subclass of RRTConnect")

    def find_nearest_backward(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the forward steering distance (subject to robot dynamics)
        from x to V[i] is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V steering backward from x
        """
        raise NotImplementedError("find_nearest_backward must be overriden by a subclass of RRTConnect")

    def steer_towards_forward(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRTConnect")

    def steer_towards_backward(self, x1, x2, eps):
        """
        Steers backward from x2 towards x1 along the shortest path (subject
        to robot dynamics). Returns x1 if the length of this shortest path is
        less than eps, otherwise returns the point at distance eps along the
        path backward from x2 to x1.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards_backward must be overriden by a subclass of RRTConnect")

    def solve(self, eps, max_iters = 1000):
        """
        Uses RRT-Connect to perform bidirectional RRT, with a forward tree
        rooted at self.x_init and a backward tree rooted at self.x_goal, with
        the aim of producing a dynamically-feasible and obstacle-free trajectory
        from self.x_init to self.x_goal.

        Inputs:
            eps: maximum steering distance
            max_iters: maximum number of RRT iterations (early termination
                is possible when a feasible solution is found)
                
        Output:
            None officially (just plots), but see the "Intermediate Outputs"
            descriptions below
        """
        
        state_dim = len(self.x_init)

        V_fw = np.zeros((max_iters, state_dim))     # Forward tree
        V_bw = np.zeros((max_iters, state_dim))     # Backward tree

        n_fw = 1    # the current size of the forward tree
        n_bw = 1    # the current size of the backward tree

        P_fw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the forward tree
        P_bw = -np.ones(max_iters, dtype=int)       # Stores the parent of each state in the backward tree

        success = False

        ## Intermediate Outputs
        # You must update and/or populate:
        #    - V_fw, V_bw, P_fw, P_bw, n_fw, n_bw: the represention of the
        #           planning trees
        #    - success: whether or not you've found a solution within max_iters
        #           RRT-Connect iterations
        #    - self.path: if success is True, then must contain list of states
        #           (tree nodes) [x_init, ..., x_goal] such that the global
        #           trajectory made by linking steering trajectories connecting
        #           the states in order is obstacle-free.
        # Hint: Use your implementation of RRT as a reference

        ########## Code starts here ##########
        def reconstruct_path():
            # forward path
            self.path = [x_new_connect]
            idx = n_fw - 1
            while idx != 0:
                parent = P_fw[idx]
                self.path.append(V_fw[parent])
                idx = parent
            self.path.reverse()
            # backward path
            idx = n_bw - 1
            while idx != 0:
                parent = P_bw[idx]
                self.path.append(V_bw[parent])
                idx = parent
        
        V_fw[0,:] = self.x_init
        V_bw[0,:] = self.x_goal
        while n_fw < max_iters or n_bw < max_iters:
            x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi, len(self.x_init))
            x_near_idx = self.find_nearest_forward(V_fw[:n_fw], x_rand)
            x_near = V_fw[x_near_idx]
            x_new = self.steer_towards_forward(x_near, x_rand, eps)
            if n_fw < max_iters and self.is_free_motion(self.obstacles, x_near, x_new):
                V_fw[n_fw] = x_new
                P_fw[n_fw] = x_near_idx
                n_fw += 1
                x_connect_idx = self.find_nearest_backward(V_bw[:n_bw], x_new)
                x_connect = V_bw[x_connect_idx]
                while n_bw < max_iters:
                    x_new_connect = self.steer_towards_backward(x_new, x_connect, eps)
                    if self.is_free_motion(self.obstacles, x_new_connect, x_connect):
                        V_bw[n_bw] = x_new_connect
                        P_bw[n_bw] = x_connect_idx
                        n_bw += 1
                        if np.array_equal(x_new_connect, x_new):
                            reconstruct_path()
                            success = True
                            break
                        x_connect = x_new_connect
                        x_connect_idx = n_bw - 1
                    else:
                        break
                if success:
                    break
            x_rand = np.random.uniform(self.statespace_lo, self.statespace_hi, len(self.x_init))
            x_near_idx = self.find_nearest_backward(V_bw[:n_bw], x_rand)
            x_near = V_bw[x_near_idx]
            x_new = self.steer_towards_backward(x_rand, x_near, eps)
            if n_bw < max_iters and self.is_free_motion(self.obstacles, x_new, x_near):
                V_bw[n_bw] = x_new
                P_bw[n_bw] = x_near_idx
                n_bw += 1
                x_connect_idx = self.find_nearest_forward(V_fw[:n_fw], x_new)
                x_connect = V_fw[x_connect_idx]
                while n_fw < max_iters:
                    x_new_connect = self.steer_towards_forward(x_connect, x_new, eps)
                    if self.is_free_motion(self.obstacles, x_connect, x_new_connect):
                        V_fw[n_fw] = x_new_connect
                        P_fw[n_fw] = x_connect_idx
                        n_fw += 1
                        if np.array_equal(x_new_connect, x_new):
                            reconstruct_path()
                            success = True
                            break
                        x_connect = x_new_connect
                        x_connect_idx = n_fw - 1
                    else:
                        break
                if success:
                    break
        ########## Code ends here ##########

        plt.figure()
        self.plot_problem()
        self.plot_tree(V_fw, P_fw, color="blue", linewidth=.5, label="RRTConnect forward tree")
        self.plot_tree_backward(V_bw, P_bw, color="purple", linewidth=.5, label="RRTConnect backward tree")
        
        if success:
            self.plot_path(color="green", linewidth=2, label="solution path")
            plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
            plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")
        plt.scatter(V_fw[:n_fw,0], V_fw[:n_fw,1], color="blue")
        plt.scatter(V_bw[:n_bw,0], V_bw[:n_bw,1], color="purple")

        plt.show()

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

class GeometricRRTConnect(RRTConnect):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        # Hint: This should take one line.
        return np.argmin(np.sum((V - x)**2, axis=1))
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        return self.find_nearest_forward(V, x)

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        # Hint: This should take one line.
        dist = np.linalg.norm(x1-x2)
        if dist <= eps:
            return x2
        else:
            return x1 + (x2-x1) * eps / dist
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        return self.steer_towards_forward(x2, x1, eps)

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_tree_backward(self, V, P, **kwargs):
        self.plot_tree(V, P, **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)

class DubinsRRTConnect(RRTConnect):
    """
    Represents a planning problem for the Dubins car, a model of a simple
    car that moves at a constant speed forward and has a limited turning
    radius. We will use this v0.9.2 of the package at
    https://github.com/AndrewWalker/pydubins/blob/0.9.2/dubins/dubins.pyx
    to compute steering distances and steering trajectories. In particular,
    note the functions dubins.path_length and dubins.path_sample (read
    their documentation at the link above). See
    http://planning.cs.uiuc.edu/node821.html
    for more details on how these steering trajectories are derived.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles, turning_radius):
        self.turning_radius = turning_radius
        super(self.__class__, self).__init__(statespace_lo, statespace_hi, x_init, x_goal, obstacles)

    def reverse_heading(self, x):
        """
        Reverses the heading of a given pose.
        Input: x (np.array [3]): Dubins car pose
        Output: x (np.array [3]): Pose with reversed heading
        """
        theta = x[2]
        if theta < np.pi:
            theta_new = theta + np.pi
        else:
            theta_new = theta - np.pi
        return np.array((x[0], x[1], theta_new))

    def find_nearest_forward(self, V, x):
        ########## Code starts here ##########
        return np.argmin([path_length(row, x, self.turning_radius) for row in V])
        ########## Code ends here ##########

    def find_nearest_backward(self, V, x):
        ########## Code starts here ##########
        return np.argmin([path_length(x, row, self.turning_radius) for row in V])
        ########## Code ends here ##########

    def steer_towards_forward(self, x1, x2, eps):
        ########## Code starts here ##########
        from dubins import path_sample
        samples = path_sample(x1, x2, 1.001*self.turning_radius, eps)[0]
        # Samples always include x1
        if len(samples) > 1:
            return samples[1]
        else:
            return x2
        ########## Code ends here ##########

    def steer_towards_backward(self, x1, x2, eps):
        ########## Code starts here ##########
        x1_new = self.reverse_heading(x1)
        x2_new = self.reverse_heading(x2)
        return self.reverse_heading(self.steer_towards_forward(x2_new, x1_new, eps))
        ########## Code ends here ##########

    def is_free_motion(self, obstacles, x1, x2, resolution = np.pi/6):
        pts = path_sample(x1, x2, self.turning_radius, self.turning_radius*resolution)[0]
        pts.append(x2)
        for i in range(len(pts) - 1):
            for line in obstacles:
                if line_line_intersection([pts[i][:2], pts[i+1][:2]], line):
                    return False
        return True

    def plot_tree(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[P[i],:], V[i,:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[i,:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_tree_backward(self, V, P, resolution = np.pi/24, **kwargs):
        line_segments = []
        for i in range(V.shape[0]):
            if P[i] >= 0:
                pts = path_sample(V[i,:], V[P[i],:], self.turning_radius, self.turning_radius*resolution)[0]
                pts.append(V[P[i],:])
                for j in range(len(pts) - 1):
                    line_segments.append((pts[j], pts[j+1]))
        plot_line_segments(line_segments, **kwargs)

    def plot_path(self, resolution = np.pi/24, **kwargs):
        pts = []
        path = np.array(self.path)
        for i in range(path.shape[0] - 1):
            pts.extend(path_sample(path[i], path[i+1], self.turning_radius, self.turning_radius*resolution)[0])
        plt.plot([x for x, y, th in pts], [y for x, y, th in pts], **kwargs)
