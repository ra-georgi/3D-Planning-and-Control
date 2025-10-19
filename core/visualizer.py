import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects as pe

class Visualizer():

    def __init__(self, cfg, times, states, controls):
        self.params   = cfg
        self.times    = times
        self.states   = states
        self.controls = controls

    # Generate Plots
    def plot_states(self):
        """Plot state variable evolution with time"""

        t =  self.times
        state = self.states

        new_state = np.zeros([12, len(t)])
        new_state[0:3,:]  = state[0:3,:]
        new_state[6:9,:]  = state[7:10,:]
        new_state[9:12,:] = state[10:13,:]

        for i in range(len(t)):
                quaternion = R.from_quat(np.array(state[3:7,i]),scalar_first=True)
                rotation_matrix = quaternion.as_matrix()

                new_state[6:9,i] = rotation_matrix @ new_state[6:9,i]

                eul = quaternion.as_euler('zyx', degrees=True)
                new_state[3:6,i] = [eul[2],eul[1],eul[0]]
                # roll_x, pitch_y, yaw_z = self.euler_from_quaternion(state[4,i], state[5,i], state[6,i], state[3,i])
                # new_state[3:6,i] = [math.degrees(roll_x),math.degrees(pitch_y),math.degrees(yaw_z)]

        num_rows = 4
        num_cols = 3
        # Create a figure with subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 8))

        # Flatten the 2D array of subplots for easier indexing
        axs = axs.flatten()

        # Labels for the y-axis
        y_labels = [
        'x_pos', 'y_pos', 'z_pos',
        'phi_euler', 'theta_euler', 'psi_euler',
        'Vx', 'Vy', 'Vz',
        'AngVx_b', 'AngVy_b', 'AngVz_b'
        ]                

        for i, ax in enumerate(axs):
                color = 'blue'
                if i % num_cols == 1:
                        color = 'red'
                elif i % num_cols == 2:
                        color = 'green'                        
                ax.plot(t, new_state[i, :],color=color)
                ax.set_xlabel('Time')
                ax.set_ylabel(y_labels[i])

        # Add a single title for the entire window
        fig.suptitle('Quadcopter State Variables vs Time', fontsize=16)    
        plt.tight_layout()
        plt.show()  

    def animate_quadcopter(self):
        """Display animation of simulated quadcopter flight"""

        fig = plt.figure(figsize=(15,7),dpi=100)
        self.ax_anim = plt.axes(projection='3d')

        for a in (self.ax_anim.xaxis, self.ax_anim.yaxis, self.ax_anim.zaxis):
            a.pane.set_facecolor((0.8, 0.8, 0.8, 0.03))   # almost transparent white
            a.pane.set_edgecolor((1, 1, 1, 0.1))    # faint edges
        self.ax_anim.zaxis.pane.set_facecolor("#FFD39B") 

        # self.ax_anim.set_facecolor("skyblue")
        self.ax_anim.set_facecolor("#c1e4f5")
        self.ax_anim.tick_params(colors=(0,0,0), labelsize=5)
        self.ax_anim.grid(False)
        # self.ax_anim.grid(True, color="white", linewidth=0.2, alpha=0.2)

        # Adjust the size of the plot within the figure
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        self.ax2 = plt.axes([0.8, 0.8, 0.2, 0.2])
        self.ax2.axis('off')  # Hide the axes for the inset      

        # Optional Box for effect
        # card = FancyBboxPatch(
        #     (0.79, 0.895), 0.05, 0.05,  # (x, y, w, h) in figure coords
        #     transform=fig.transFigure,
        #     boxstyle="round,pad=0.05,rounding_size=0.015",
        #     # facecolor=(1, 1, 1, 0.88),     # soft white, slightly transparent
        #     facecolor= "lightgray",
        #     edgecolor=(0, 0, 0, 0.25),
        #     linewidth=1.0
        # )
        
        # # subtle shadow
        # card.set_path_effects([pe.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.25)])
        # fig.patches.append(card)

        # Monospaced, right-aligned numbers
        # self.hud_text = fig.text(
        #     0.85, 0.945, "", ha="left", va="top",
        #     family="DejaVu Sans Mono", fontsize=11, color="#222"
        # )

        hud_title = fig.text(0.5, 0.98, "Flight Animation",
                          ha="center", va="top", fontsize=12, weight="bold")


        self.hud_text = fig.text(
            0.621, 0.98, "", ha="left", va="top",
            family="DejaVu Sans Mono", fontsize=11, color="#222",
            bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.02",
                    facecolor=(1,1,1,0.90), edgecolor=(0,0,0,0.25)),
            zorder=10, clip_on=False
        )

        # thin stroke for legibility on any background
        self.hud_text.set_path_effects([pe.withStroke(linewidth=2, foreground=(1,1,1,0.7))])



        # # Initialize the text in the inset axes with a box around it
        # bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgray")
        # self.text_t = self.ax2.text(0.1, 0.8, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
        # self.text_x = self.ax2.text(0.1, 0.6, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)   
        # self.text_y = self.ax2.text(0.1, 0.4, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)
        # self.text_z = self.ax2.text(0.1, 0.2, '', transform=self.ax2.transAxes, fontsize=12, bbox=bbox_props)     

        state = self.states       
        l = self.params["quadcopter"]["arm_length"]                                 
        
        
        x0, y0, z0 = state[0,0], state[1,0], state[2,0]
        self.quad_Arm1 = self.ax_anim.plot3D([x0+l, x0-l], [y0, y0], [z0, z0], lw=3 )[0]
        self.quad_Arm2 = self.ax_anim.plot3D([x0, x0], [y0+l, y0-l], [z0, z0], lw=3 )[0]
        self.quad_traj = self.ax_anim.plot3D(x0, y0, z0, 'gray')[0] 


        # Box body parameters
        body_size = self.params["quadcopter"].get("body_size", 0.5*l)
        self.body_half = np.array([body_size/2, body_size/2, body_size/4])  # box half extents

        # Rotor disks parameters
        self.rotor_offsets = [
            np.array([ l, 0, 0]),
            np.array([-l, 0, 0]),
            np.array([ 0, l, 0]),
            np.array([ 0,-l, 0]),
        ]
        self.rotor_radius = self.params["quadcopter"].get("rotor_radius", 0.18*l)

        theta = np.linspace(0, 2*np.pi, 40)
        self._disk_c = np.cos(theta)
        self._disk_s = np.sin(theta)

        self.body_box = None
        self.rotor_disks = []
        for _ in self.rotor_offsets:
            disk = Poly3DCollection([], color='black', alpha=0.5)
            self.ax_anim.add_collection3d(disk)
            self.rotor_disks.append(disk)



        #To make quadcopter's arms look equal in animation
        # TODO: Make the limits to parameters that can be set
        self.ax_anim.set_xlim([-5,5])
        self.ax_anim.set_ylim([-5,5])
        self.ax_anim.set_zlim([-5,5])      
 
        # Plot start and goal points
        waypoints =  self.params["world"]["waypoints"]
        for wp in waypoints:
                # self.ax_anim.scatter(wp["pose"][0], wp["pose"][1], wp["pose"][2])   
                self.ax_anim.scatter(wp["pose"][0], 
                                     wp["pose"][1], 
                                     wp["pose"][2], 
                                     s=50, c='orange', edgecolors='black', alpha=0.8)  

        # Plot static obstacles
        obstacles =  self.params["obstacles"]["static"]
        for obstacle in obstacles:
            radius_obstacle = obstacle["radius"]
            u_grid, v_grid = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
            x_grid = radius_obstacle*np.cos(u_grid)*np.sin(v_grid)
            y_grid = radius_obstacle*np.sin(u_grid)*np.sin(v_grid) 
            z_grid = radius_obstacle*np.cos(v_grid)
            self.ax_anim.plot_surface(  x_grid + obstacle["pose"][0], 
                                        y_grid + obstacle["pose"][1], 
                                        z_grid + obstacle["pose"][2], color='r',alpha=0.8)                  


        
        #For Frame Update function
        self.arm1_start = np.array([ l,0,0])
        self.arm1_end   = np.array([-l,0,0])

        self.arm2_start = np.array([0, l,0])
        self.arm2_end   = np.array([0,-l,0])

        self.ani = FuncAnimation(fig=fig, func=self.update_anim_quad,frames=state.shape[1], fargs=(),interval=10)
        plt.show()

    def update_anim_quad(self,frame):
            # for each frame, update the data stored on each artist.

            time = self.times[frame]
            xt, yt, zt = self.states[0,frame], self.states[1,frame], self.states[2,frame]

            arm_length = self.params["quadcopter"]["arm_length"]                 
            quaternion = R.from_quat(np.array(self.states[3:7,frame]),scalar_first=True)
            rotation_matrix = quaternion.as_matrix()


            # === Update box ===
            if self.body_box is not None:
                self.body_box.remove()

            # Define 8 corners of the box in body frame
            hx, hy, hz = self.body_half
            corners = np.array([
                [ hx,  hy,  hz],
                [ hx, -hy,  hz],
                [-hx, -hy,  hz],
                [-hx,  hy,  hz],
                [ hx,  hy, -hz],
                [ hx, -hy, -hz],
                [-hx, -hy, -hz],
                [-hx,  hy, -hz],
            ]).T

            corners_w = (rotation_matrix @ corners).T + np.array([xt, yt, zt])

            faces = [
                [corners_w[j] for j in [0,1,2,3]],
                [corners_w[j] for j in [4,5,6,7]],
                [corners_w[j] for j in [0,1,5,4]],
                [corners_w[j] for j in [2,3,7,6]],
                [corners_w[j] for j in [1,2,6,5]],
                [corners_w[j] for j in [4,7,3,0]],
            ]
            self.body_box = Poly3DCollection(faces, facecolors='black', alpha=0.6)
            self.ax_anim.add_collection3d(self.body_box)

            # === Update rotor disks ===
            ex = rotation_matrix @ np.array([1,0,0])
            ey = rotation_matrix @ np.array([0,1,0])

            for i, offset in enumerate(self.rotor_offsets):
                c = rotation_matrix @ offset + np.array([xt,yt,zt])

                # Compute rotor circle points
                circle_pts = (
                    c.reshape(3,1)
                    + self.rotor_radius * (np.outer(ex, self._disk_c) + np.outer(ey, self._disk_s))
                ).T  # shape (N,3)

                # Update disk polygon
                self.rotor_disks[i].set_verts([circle_pts])



            Arm1_Start = rotation_matrix @ self.arm1_start
            Arm1_End   = rotation_matrix @ self.arm1_end
            Arm2_Start = rotation_matrix @ self.arm2_start
            Arm2_End   = rotation_matrix @ self.arm2_end              

            self.quad_Arm1.set_data_3d([xt+Arm1_Start[0], xt+Arm1_End[0]], 
                                    [yt+Arm1_Start[1], yt+Arm1_End[1]], 
                                    [zt+Arm1_Start[2], zt+Arm1_End[2]])
            
            self.quad_Arm2.set_data_3d([xt+Arm2_Start[0], xt+Arm2_End[0]], 
                                    [yt+Arm2_Start[1], yt+Arm2_End[1]], 
                                    [zt+Arm2_Start[2], zt+Arm2_End[2]])
            
            self.quad_traj.set_data_3d(self.states[0,:frame],
                                    self.states[1,:frame],
                                    self.states[2,:frame])

            # self.text_t.set_text(f't = {time:.2f} s')
            # self.text_x.set_text(f'x = {xt:.2f} m')
            # self.text_y.set_text(f'y = {yt:.2f} m')
            # self.text_z.set_text(f'z = {zt:.2f} m')

            control = self.params.get("controller", {}).get("name", "PID")

            self.hud_text.set_text(
                "Control Method: TBD\n t  = {:>7.2f} s\n x  = {:>7.2f} m\n y  = {:>7.2f} m\n z  = {:>7.2f} m"
                .format(time, xt, yt, zt)
)


            return 
    


