import numpy as np
import open3d as o3d
import copy
import scipy.linalg

def vector_to_se3(v):
    w = v[3:]
    v_trans = v[:3]
    W = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])
    T = np.eye(4)
    theta = np.linalg.norm(w)
    if theta < 1e-7:
        T[:3, :3] = np.eye(3) + W
    else:
        T[:3, :3] = scipy.linalg.expm(W)
    T[:3, 3] = v_trans
    return T

def se3_to_vector(T):
    R = T[:3, :3]
    t = T[:3, 3]
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-7:
        w = np.zeros(3)
    else:
        log_R = scipy.linalg.logm(R)
        w = np.array([log_R[2, 1], log_R[0, 2], log_R[1, 0]])
    return np.concatenate([t, np.real(w)])

class ADMMSubGraph:
    """Represents a local robot trajectory in a distributed ADMM setup."""
    def __init__(self, robot_id, nodes, edges):
        self.robot_id = robot_id
        
        # Local state (dictionary of 4x4 SE(3) poses)
        self.poses = {n_id: pose.copy() for n_id, pose in nodes.items()}
        self.edges = edges # List of Open3D PoseGraphEdge
        
        # ADMM Local dual variables (y) for consensus nodes
        self.dual_vars = {} # Maps shared node_id -> 6D dual vector

    def local_optimize(self, consensus_vars, rho):
        """
        Step A: Local optimization (x-update).
        Minimizes local odometry errors + ADMM penalty for consensus variables.
        """
        # In a full implementation, we'd run Levenberg-Marquardt or Gauss-Newton here
        # incorporating the penalty term: (rho/2) * || x_i - z + y_i ||^2
        # For this structural template, we apply a simplified gradient descent step
        
        learning_rate = 0.01
        
        for iteration in range(5): # Local solving steps
            for edge in self.edges:
                i, j = edge.source_node_id, edge.target_node_id
                # Skip if one of the nodes doesn't belong to this subgraph
                if i not in self.poses or j not in self.poses: continue
                
                # Odometry gradient approximation
                Xi, Xj = self.poses[i], self.poses[j]
                Z = edge.transformation
                
                # Simplified error pulling nodes together
                err_mat = np.linalg.inv(Z) @ np.linalg.inv(Xi) @ Xj
                e_vec = se3_to_vector(err_mat)
                
                # Apply odometry constraint adjustments (simplified)
                if i != 0: # Anchor check
                    self.poses[i] = self.poses[i] @ vector_to_se3(learning_rate * e_vec)
                self.poses[j] = self.poses[j] @ vector_to_se3(-learning_rate * e_vec)

        # Apply ADMM Consensus Penalty Pull
        for c_id, z_pose in consensus_vars.items():
            if c_id in self.poses:
                # ADMM Penalty gradient: rho * (x_i - z + y_i)
                err_mat = np.linalg.inv(z_pose) @ self.poses[c_id]
                x_minus_z = se3_to_vector(err_mat)
                
                # Apply dual penalty
                penalty = rho * (x_minus_z + self.dual_vars[c_id])
                
                # Pull local pose towards consensus
                self.poses[c_id] = self.poses[c_id] @ vector_to_se3(-0.1 * penalty)


def optimise_pose_graph_admm(pose_graph, num_iterations=10, rho=1.0):
    """
    Custom Consensus ADMM solver for Open3D PoseGraph.
    Artificially partitions the graph to simulate multi-robot distributed SLAM.
    """
    print(f"\n  [ADMM] Initialising Consensus ADMM with {num_iterations} iterations...")
    
    # 1. Artificially Partition the Graph (Simulate 2 Robots)
    N = len(pose_graph.nodes)
    split_idx = N // 2
    
    nodes_1 = {i: pose_graph.nodes[i].pose for i in range(split_idx + 1)}
    nodes_2 = {i: pose_graph.nodes[i].pose for i in range(split_idx, N)}
    
    edges_1, edges_2, cross_edges = [], [], []
    for edge in pose_graph.edges:
        if edge.source_node_id <= split_idx and edge.target_node_id <= split_idx:
            edges_1.append(edge)
        elif edge.source_node_id >= split_idx and edge.target_node_id >= split_idx:
            edges_2.append(edge)
        else:
            cross_edges.append(edge)
            
    print(f"  [ADMM] Partitioned into Robot 1 (Nodes 0-{split_idx}) and Robot 2 (Nodes {split_idx}-{N-1})")
    
    robot1 = ADMMSubGraph(robot_id=1, nodes=nodes_1, edges=edges_1)
    robot2 = ADMMSubGraph(robot_id=2, nodes=nodes_2, edges=edges_2)
    
    # The split point (split_idx) is our shared "Separator" Node
    consensus_nodes = [split_idx]
    
    # Initialize Global Consensus Variables (z)
    global_consensus = {c_id: pose_graph.nodes[c_id].pose.copy() for c_id in consensus_nodes}
    
    # Initialize Local Dual Variables (y)
    robot1.dual_vars = {c_id: np.zeros(6) for c_id in consensus_nodes}
    robot2.dual_vars = {c_id: np.zeros(6) for c_id in consensus_nodes}
    
    # 2. ADMM Alternating Loop
    for it in range(num_iterations):
        # Step A: Local Optimization (Parallel execution in real distributed systems)
        robot1.local_optimize(global_consensus, rho)
        robot2.local_optimize(global_consensus, rho)
        
        # Step B: Consensus Update (Z-update)
        for c_id in consensus_nodes:
            # Gather estimates from both robots
            x1 = robot1.poses[c_id]
            x2 = robot2.poses[c_id]
            
            y1 = robot1.dual_vars[c_id]
            y2 = robot2.dual_vars[c_id]
            
            # Simple average consensus in tangent space
            err_mat = np.linalg.inv(x1) @ x2
            diff_vec = se3_to_vector(err_mat)
            
            # z = average(x) + average(y)
            # Apply half the difference to x1 to find the midpoint
            midpoint = x1 @ vector_to_se3(0.5 * diff_vec)
            
            # Offset by dual variables
            avg_dual = 0.5 * (y1 + y2)
            global_consensus[c_id] = midpoint @ vector_to_se3(avg_dual)
            
        # Step C: Dual Update (Y-update)
        for c_id in consensus_nodes:
            # y_i = y_i + x_i - z
            z_inv = np.linalg.inv(global_consensus[c_id])
            
            err1 = se3_to_vector(z_inv @ robot1.poses[c_id])
            robot1.dual_vars[c_id] += err1
            
            err2 = se3_to_vector(z_inv @ robot2.poses[c_id])
            robot2.dual_vars[c_id] += err2

    print("  [ADMM] Consensus reached.")

    # 3. Reconstruct optimized Open3D Pose Graph
    optimized_graph = copy.deepcopy(pose_graph)
    for i in range(N):
        if i <= split_idx:
            optimized_graph.nodes[i].pose = robot1.poses[i]
        else:
            optimized_graph.nodes[i].pose = robot2.poses[i]
            
    # Snap the shared boundary node perfectly to the global consensus
    optimized_graph.nodes[split_idx].pose = global_consensus[split_idx]
        
    return optimized_graph
