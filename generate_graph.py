# Load point cloud
# generate graph 


import numpy as np
import pyvista as pv
import open3d as o3d

import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from graph_nets import utils_np
from graph_nets import blocks
from graph_nets import utils_tf

class GenerateGraphTuple(object):
    def __init__(self):
        print("generate_graphs")

    def read_point_cloud(self, path):
        pcd = o3d.io.read_point_cloud(path)
        xyz_load = np.asarray(pcd.points)
        alpha = .9
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        mesh_smp = mesh.simplify_quadric_decimation(target_number_of_triangles=100)
        mesh_smp.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])
        # o3d.visualization.draw_geometries([mesh_smp])
        return np.asarray(mesh_smp.vertices), mesh_smp

    def mesh_triangles_to_edges(self, mesh_traingles):
        mesh_triangle_array = np.asarray(mesh_traingles)
        mesh_edges = []
        for i in range(len(mesh_triangle_array)):
            mesh_edges.append(np.delete(mesh_triangle_array[i],2))
            mesh_edges.append(np.delete(mesh_triangle_array[i],0))
            mesh_edges.append(np.delete(mesh_triangle_array[i],1))
        return mesh_edges

    def base_graph(self, xyz_load, mesh_edges, prev_xyz, step, spring_constant=np.random.randint(5)):
        # Nodes
        # Generate initial position and velocity for all masses.
        # The left-most mass has is at position (0, 0); other masses (ordered left to
        # right) have x-coordinate d meters apart from their left neighbor, and
        # y-coordinate 0. All masses have initial velocity 0m/s.
        num_nodes = len(xyz_load)
        nodes = np.zeros((num_nodes,6) , dtype=np.float32)
        rest_length=np.zeros((len(mesh_edges),3))
        for i in range(0,3):
            nodes[:,i] = xyz_load[:,i]
            if step != 0:
                nodes[:,i+3] = (xyz_load[:,i]-prev_xyz[:,i])
    
        # rest_length contains the nodes connecting and the distance betwen them
        for n in range(0, len(mesh_edges)):
            rest_length[n] = [mesh_edges[n,0],mesh_edges[n,1],float(np.linalg.norm(nodes[mesh_edges[n,0]] - nodes[mesh_edges[n,1]]))]

        # Edges.
        edges, senders, receivers = [], [], []
        for i in range(len(rest_length)):
            left_node = int(rest_length[i,0])
            right_node = int(rest_length[i,1])
            # # The 'if' statements prevent incoming edges to fixed ends of the string.
            # if right_node < n - 1:
            # # Left incoming edge.
            edges.append([spring_constant, rest_length[i,2]])
            senders.append(left_node)
            receivers.append(right_node)
            # if left_node > 0:
            # # Right incoming edge.
            #     edges.append([50., d])
            #     senders.append(right_node)
            #     receivers.append(left_node)

        return {
            "globals": [0., 0.],
            "nodes": nodes,
            "edges": edges,
            "receivers": receivers,
            "senders": senders
        }

    def get_graph_batch(self, batch_num, batch_size=5):
        batch = np.random.randint(batch_num[0], batch_num[1], batch_size)
        step = np.random.randint(0,4)
        static_tr_t = []
        static_tr_t_1 = []
        for i in range(batch_size):
            if step == 0:
                prev_xyz = None
            else:
                prev_xyz, mesh_t=self.read_point_cloud('generated_data/3d_ball_iter'+str(batch[i])+'num_step'+str(step-1)+'.ply')
            xyz_t, mesh_t=self.read_point_cloud('generated_data/3d_ball_iter'+str(batch[i])+'num_step'+str(step)+'.ply')
            xyz_t_1, mesh_t_1=self.read_point_cloud('generated_data/3d_ball_iter'+str(batch[i])+'num_step'+str(step+1)+'.ply')
            mesh_edges_t = self.mesh_triangles_to_edges(mesh_t.triangles)
            mesh_edges_t_1 = self.mesh_triangles_to_edges(mesh_t_1.triangles)
            base_graph_t = self.base_graph(xyz_t, np.array(mesh_edges_t), prev_xyz, step)
            base_graph_t_1 = self.base_graph(xyz_t, np.array(mesh_edges_t_1), xyz_t, step)
            static_tr_t.append(base_graph_t)
            static_tr_t_1.append(base_graph_t_1)
        input_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_tr_t)
        target_graph_tr = utils_tf.data_dicts_to_graphs_tuple(static_tr_t_1)
        # print(input_graph_tr)
        return input_graph_tr, target_graph_tr, base_graph_t
            

class GenerateGraphVisual(object):

    def visualize_points(self, pos, edge_index=None, index=None):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(projection='3d')
        if edge_index is not None:
            for (src, dst) in edge_index.t().tolist():
                src = pos[src].tolist()
                dst = pos[dst].tolist()
                plt.plot([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]], linewidth=1, color='black')
        if index is None:
            ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], s=50, zorder=1000)
        else:
            mask = torch.zeros(pos.size(0), dtype=torch.bool)
            mask[index] = True
            ax.scatter3D(pos[~mask, 0], pos[~mask, 1], pos[~mask, 2], s=50, color='lightgray', zorder=1000)
            ax.scatter3D(pos[mask, 0], pos[mask, 1], pos[mask, 2], s=50, zorder=1000)
        plt.axis('off')
        ax.set_aspect('equal', 'box')
        plt.show()

    def pointcloud_mesh(self, path):
        pcd = o3d.io.read_point_cloud(path)
        # o3d.visualization.draw_geometries([pcd])
        alpha = .5
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        # print(np.asarray(mesh.vertices))
        # print(np.asarray(mesh.triangles))
        return mesh

        
    # this is a hack, might miss certain edge connections
    def mesh_triangles_to_edges(self, mesh_traingles):
        mesh_triangle_array = np.asarray(mesh_traingles)
        mesh_edges = []
        for i in range(len(mesh_triangle_array)):
            mesh_edges.append(np.delete(mesh_triangle_array[i],2))
            mesh_edges.append(np.delete(mesh_triangle_array[i],0))
            mesh_edges.append(np.delete(mesh_triangle_array[i],1))
        return mesh_edges


    def generate_graph_data(self, nodes, edges):
        edge_index = torch.tensor(np.transpose(np.asarray(edges)), dtype=torch.long)
        x = torch.tensor(nodes, dtype=torch.float)
        data = Data(pos=x, edge_index=edge_index)
        return data

if __name__ == "__main__":
    graph = GenerateGraphVisual()
    # points is a 3D numpy array (n_points, 3) coordinates of a sphere
    mesh = graph.pointcloud_mesh('generated__data/3d_ball_iter0num_step4.ply')
    mesh_edges = graph.mesh_triangles_to_edges(mesh.triangles)
    data = graph.generate_graph_data(np.asarray(mesh.vertices), mesh_edges)
    data.validate(raise_on_error=True)
    # graph.visualize_points(data.pos, edge_index=data.edge_index)
    graph_tuple = GenerateGraphTuple()

    # xyz=graph_tuple.read_point_cloud('generated__data/3d_ball_iter0num_step4.ply')
    # base_graph = graph_tuple.base_graph(xyz, np.array(mesh_edges))
    # base_graph_tr = utils_tf.data_dicts_to_graphs_tuple([base_graph])
    graph_tuple.get_graph_batch()
