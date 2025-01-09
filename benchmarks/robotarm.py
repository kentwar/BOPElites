from benchmarks.BaseBenchmark import BaseExperiment
import math
import numpy as np
import torch

class RobotArm(BaseExperiment):
    def __init__(self, feature_resolution , seed = 100):
        
        kwargs =    {
            'example_x' : [0,0,0,0] ,
            'Xconstraints' : [[0, 1], [0, 1], [0, 1], [0, 1] ] ,
            'featmins' : [0,0] ,
            'featmaxs' : [1,1] ,
            'lowestvalue' : 0 ,
            'maxvalue' : 1 ,
            }
        self._set_Xconstraints(np.array(kwargs['Xconstraints']))    #input x ranges [min,max]
        self.example_x = kwargs['example_x']
        self.xdims = len(kwargs['example_x']) 
        self.fdims = len(kwargs['featmins']) 
        self.featmins = kwargs['featmins']
        self.featmaxs = kwargs['featmaxs']
        self.feature_resolution = feature_resolution
        self.lowestvalue = kwargs['lowestvalue']
        self.maxvalue = kwargs['maxvalue']
        self.seed = seed
        self.name = 'robotarm'
        self.desc1name = 'x position of joints'
        self.desc2name = 'y position of joints'

    def forward_kinematics(self, joint_positions, link_lengths):
        '''
        Compute the forward kinematics of a planar robotic arm:
        given the joint positions and the link lengths, returns the 2D 
        Cartesian position of the end-effector (the hand)
        '''
        assert(len(joint_positions) == len(link_lengths))
        
        # some init
        p = np.append(joint_positions, 0) # end-effector has no angle
        l = np.concatenate(([0], link_lengths)) # first link has no length
        joint_xy = np.zeros((len(p), 2)) # Cartesian positions of the joints
        mat = np.matrix(np.identity(4)) # 2D transformation matrix

        # compute the position of each joint
        for i in range(0, len(l)):
            m = [[math.cos(p[i]), -math.sin(p[i]), 0, l[i]],
                [math.sin(p[i]),  math.cos(p[i]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            joint_xy[i,:] = np.array(v[0:2].A.flatten())
        return(joint_xy) # return the position of the joints

    def forward_kinematics_vectorized(self, joint_positions, link_lengths):
        '''
        Compute the forward kinematics of multiple planar robotic arms:
        given the joint positions and the link lengths for each arm, returns the 2D 
        Cartesian positions of the end-effectors (the hands)
        '''
        assert joint_positions.shape == link_lengths.shape
        
        n_arms, n_joints = joint_positions.shape
        joint_positions = np.hstack((joint_positions, np.zeros((n_arms, 1)))) # add zero angle for end-effector
        link_lengths = np.hstack((np.zeros((n_arms, 1)), link_lengths)) # add zero length for first link
        
        joint_xy = np.zeros((n_arms, n_joints + 1, 2)) # Cartesian positions of the joints
        
        for arm_index in range(n_arms):
            mat = np.matrix(np.identity(4)) # 2D transformation matrix for each arm
            for joint_index in range(n_joints + 1):
                angle = joint_positions[arm_index, joint_index]
                length = link_lengths[arm_index, joint_index]
                m = np.matrix([
                    [math.cos(angle), -math.sin(angle), 0, length],
                    [math.sin(angle), math.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
                mat = mat * m
                v = mat * np.matrix([0, 0, 0, 1]).T
                joint_xy[arm_index, joint_index, :] = v[0:2].A1
        
        return joint_xy # return the positions of the joints for all arms

    def forward_kinematics_vectorized_torch(self, joint_positions, link_lengths):
        '''
        Compute the forward kinematics of multiple planar robotic arms using PyTorch:
        given the joint positions and the link lengths for each arm, returns the 2D 
        Cartesian positions of the end-effectors (the hands)
        '''
        assert joint_positions.shape == link_lengths.shape
        
        n_arms, n_joints = joint_positions.shape
        device = joint_positions.device # to ensure tensors are on the same device

        # Initialize the transformation matrix as the identity matrix for each arm
        mat = torch.eye(4, device=device).repeat(n_arms, 1, 1)  # Shape: (n_arms, 4, 4)
      
        joint_positions = torch.cat((joint_positions, torch.zeros(n_arms, 1, device=device)), dim=1) # add zero angle for end-effector
        link_lengths = torch.cat((torch.zeros(n_arms, 1, device=device), link_lengths), dim=1) # add zero length for first link
        
        joint_xy = torch.zeros((n_arms, n_joints + 1, 2), device=device)# Cartesian positions of the joints
        
        for arm_index in range(n_arms):
            mat = torch.eye(4, device=device, dtype = torch.double) # 2D transformation matrix for each arm
            for joint_index in range(n_joints + 1):
                angle = joint_positions[arm_index, joint_index]
                length = link_lengths[arm_index, joint_index]
                # Preallocate m with the correct shape, device, and dtype, ensuring it supports gradients
                m = torch.zeros((n_arms, 4, 4), device=device, dtype=torch.double)

                # Assign values to m while preserving the computational graph
                cos_angle = torch.cos(angle)
                sin_angle = torch.sin(angle)
                m[:, 0, 0] = cos_angle
                m[:, 0, 1] = -sin_angle
                m[:, 1, 0] = sin_angle
                m[:, 1, 1] = cos_angle
                m[:, 0, 3] = length * cos_angle
                m[:, 1, 3] = length * sin_angle
                m[:, 2, 2] = 1.0
                m[:, 3, 3] = 1.0
                mat = torch.bmm(mat, m)
                #v = torch.mm(mat, torch.tensor([[0], [0], [0], [1]], device=device, dtype= torch.double))
                #joint_xy[arm_index, joint_index, :] = v[:2].squeeze()
        end_effector_positions = mat[:, :2, 3]

        return joint_xy # return the positions of the joints for all arms

    def forward_kinematics_end_effector_torch(self, joint_positions, link_lengths):
        assert joint_positions.shape == link_lengths.shape, "Shapes of joint_positions and link_lengths must match."
        
        n_arms, n_joints = joint_positions.shape
        device = joint_positions.device  # Ensure all operations are on the same device as input tensors
        
        # Initialize the transformation matrix as the identity matrix for each arm
        mat = torch.eye(4, device=device, requires_grad=True).repeat(n_arms, 1, 1)  # Shape: (n_arms, 4, 4)
        
        # Iterate through joints to apply transformations
        for i in range(n_joints):
            angle = joint_positions[:, i]
            length = link_lengths[:, i]
            
            # Calculate transformation matrix for the current joint
            cos, sin = torch.cos(angle), torch.sin(angle)
            m = torch.zeros((n_arms, 4, 4), device=device)
            m[:, 0, 0] = cos
            m[:, 0, 1] = -sin
            m[:, 1, 0] = sin
            m[:, 1, 1] = cos
            m[:, 0, 3] = length * cos  # Translation in x
            m[:, 1, 3] = length * sin  # Translation in y
            m[:, 2, 2] = m[:, 3, 3] = 1  # Homogeneous coordinates
            
            # Apply transformation
            mat = torch.bmm(mat, m)
            
        # Extract end-effector positions from the last column of the transformation matrices
        end_effector_positions = mat[:, :2, 3]  # Shape: (n_arms, 2)
        
        return end_effector_positions

    def torch_fitness(self, genotype):
        '''
        fitness is the standard deviation of joint angles (Smoothness)
        (As it is negative, we want to maximize it)
        '''
        genotype = genotype
        fit = 1 - torch.std(genotype, dim = 1 , unbiased = False)

        # now compute the behavior
        #   scale to [0,2pi]
        g = self.torch_interp(genotype, torch.tensor([0, 1]), torch.tensor([0, 2 * math.pi]))
        j = self.forward_kinematics_end_effector_torch(g, torch.ones(g.shape[0], g.shape[1]))
        #  normalize behavior in [0,1]
        n_configs = genotype.shape[1]
        # Extract the end-effector positions (last joint positions) for each configuration
        #end_effector_positions = j[:, -1, :]
        # Normalize the end-effector positions
        b = (j) / (2 * n_configs) + 0.5
        return fit , b

    def fitness(self, genotype):
        '''
        fitness is the standard deviation of joint angles (Smoothness)
        (As it is negative, we want to maximize it)
        '''
        if type(genotype) == torch.Tensor:
            return(self.torch_fitness(genotype)[0])
        
        fit = 1 - np.std(genotype)

        # now compute the behavior
        #   scale to [0,2pi]
        g = np.interp(genotype, (0, 1), (0, 2 * math.pi))
        j = self.forward_kinematics(g, [1]*len(g))
        #  normalize behavior in [0,1]
        b = (j[-1,:]) / (2 * len(g)) + 0.5
        return fit , b# the fitness and the position of the last joint

    def fitness_fun(self, X):
        '''Function wrapper
        '''
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        if t == torch.Tensor:
            return(self.torch_fitness(X)[0])
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        assert (X >= self.Xconstraints[:,0]).all() , 'The point is outside the box constraints (lower bound)'
        assert (X <= self.Xconstraints[:,1]).all() , 'The point is outside the box constraints (Upper bound)'
        ## Single Point    
        if s == (1,self.xdims):
            return(self.fitness(X[0])[0] )
        else:
            return([self.fitness(x)[0] for x in X])

    def feature_fun(self, X):
        '''Function wrapper
        '''
        if type(X) == torch.Tensor:
            return(self.torch_fitness(X)[1])
        t = type(X)
        s = np.shape(X)
        ms = np.shape(X[0])
        assert t == np.ndarray, 'Input to the fitness function must be an array'
        assert s == (1,self.xdims) or ms == (self.xdims,), 'genome is the wrong shape'
        ## Single Point    
        if s == (1,self.xdims):
            return(self.fitness(X[0])[1] )
        else:
            return([self.fitness(x)[1] for x in X])

    def feat_fun(self, X):
        '''Function wrapper
        '''
        return(self.feature_fun(X))

    def torch_interp(self, x, xp, fp):
        # Find the indices of the two points in xp that surround each x value
        idxs = torch.searchsorted(xp, x)
        idxs = idxs.clamp(1, len(xp) - 1)
        left = xp[idxs - 1]
        right = xp[idxs]
        left_val = fp[idxs - 1]
        right_val = fp[idxs]

        # Compute the slope of the line between the points
        # and use it to interpolate the y value
        slope = (right_val - left_val) / (right - left)
        return left_val + slope * (x - left)
        


#domain = RobotArm(**kwargs)

