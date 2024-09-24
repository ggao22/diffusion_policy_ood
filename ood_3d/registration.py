import click
import numpy as np
import matplotlib.pyplot as plt

from pycpd import RigidRegistration
from sklearn.datasets import make_spd_matrix

import open3d as o3d
import numpy as np

def to_voxel_centers(ptcloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcloud)
    voxel_size = 0.01
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    voxel_centers = [voxel_grid.get_voxel_center_coordinate(voxel.grid_index) for voxel in voxels]

    # o3d.visualization.draw_geometries([voxel_grid])

    return voxel_centers
    return ptcloud

@click.command()
@click.option('-p', '--ptcloud_path', required=True)
def main(ptcloud_path):
    ptcloud = np.load(ptcloud_path)

    voxel_ptclouds = []
    for k in range(len(ptcloud)):
        voxel_ptclouds.append(np.array(to_voxel_centers(ptcloud[k])))
        # np.random.shuffle(voxel_ptclouds[k])
    
    for k in range(len(voxel_ptclouds)-1):

        target = np.copy(voxel_ptclouds[k])
        TY = np.copy(voxel_ptclouds[k+1])
        iters = 150

        best_error = np.inf
        best_reg = None
        for _ in range(iters):
            try:
                R_init = make_spd_matrix(3)
                reg = RigidRegistration(X=target, Y=TY, R=R_init, w=0.4)
                TY, (s_reg, R_reg, t_reg) = reg.register()
                if reg.q < best_error:
                    best_error = reg.q
                    best_reg = TY, (s_reg, R_reg, t_reg)
            except:
                target = np.copy(voxel_ptclouds[k])
                TY = np.copy(voxel_ptclouds[k+1])
                # pass
        TY, (s_reg, R_reg, t_reg) = best_reg
        print(s_reg, R_reg, t_reg)

        data = [voxel_ptclouds[k], TY, voxel_ptclouds[k+1]]
        fig_lims = 1
        fig = plt.figure(figsize=(30,10))
        for d in range(len(data)):
            ax = fig.add_subplot(1, len(data), d+1, projection='3d')
            ax.set_xlim(-fig_lims/2, fig_lims/2)
            ax.set_ylim(-fig_lims/2, fig_lims/2)
            ax.set_zlim(0, fig_lims)
            ax.elev = 90
            for i in range(0,len(data[d]),2):
                ax.scatter(data[d][i,0],data[d][i,1],data[d][i,2], color=plt.cm.rainbow(i/len(data[d])), s=2)
        plt.show()

if __name__ == '__main__':
    main()
