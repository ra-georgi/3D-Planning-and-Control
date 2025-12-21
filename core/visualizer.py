import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib import patheffects as pe

class Visualizer():

    def __init__(self, cfg, times, states, controls, controller, planner, trajectory_object):
        self.params   = cfg
        self.times    = times
        self.states   = states
        self.controls = np.insert(controls, 0, 0, axis=1)   #To make the same size as states 
        self.controller_name = controller.controller_name
        self.controller_dt = controller.controller_dt
        if planner is not None:
            self.planner_name = planner.planner_name
            self.interpolator_name = planner.interpolator_name
            self.trajectory_object = trajectory_object
            self.planner_res = planner.voxel_resolution

            # === Precompute planned trajectory positions ===
            pos_des_all = np.zeros((3, len(times)))
            for i, t in enumerate(times):
                pos_des, vel_des, acc_des = self.trajectory_object.evaluate_trajectory(t)
                # assume pos_des is [x, y, z]
                pos_des_all[:, i] = pos_des
                self.pos_des_all = pos_des_all        

        else:
              self.planner_name = " - "
              self.interpolator_name = " - "
              self.planner_res = " - "
              self.pos_des_all = None

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

        # ---- TITLE BANNER -------------------------------------------------
        # Banner background (rounded rectangle)
        banner = FancyBboxPatch(
            (0.2795, 0.96), 0.441, 0.01,   # (x, y, width, height) in figure coords
            transform=fig.transFigure,
            boxstyle="square,pad=0.02",
            facecolor="#1f4e79",        # dark blue
            edgecolor="none",
            alpha=0.95
        )
        fig.patches.append(banner)

        # Main title text
        fig.text(
            0.5, 0.965, "Flight Animation",
            ha="center", va="center",
            fontsize=16, color="white",
            weight="bold", family="Arial", style="italic",
            path_effects=[pe.withSimplePatchShadow(offset=(1,-1), alpha=0.4)]
        )

        # ------------------------------------------------------------------

        self.ax_anim = plt.axes(projection='3d')

        for a in (self.ax_anim.xaxis, self.ax_anim.yaxis, self.ax_anim.zaxis):
            a.pane.set_facecolor((0.8, 0.8, 0.8, 0.03))   # almost transparent white
            a.pane.set_edgecolor((1, 1, 1, 0.1))    # faint edges
        self.ax_anim.zaxis.pane.set_facecolor("#FFD39B") 

        self.ax_anim.set_facecolor("#c1e4f5")
        self.ax_anim.tick_params(colors=(0,0,0), labelsize=5)
        self.ax_anim.grid(False)
        # self.ax_anim.grid(True, color="white", linewidth=0.2, alpha=0.2)

        # Adjust the size of the plot within the figure
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

        self.ax_side = plt.axes([0.74, 0.0, 0.22, 1.0])
        self.ax_side.axis('off')

        # --- RIGHT CARDS (two boxes) ---------------------------------------------
        # Card 1: Data (dynamic)

        # ---- RIGHT BANNERS -------------------------------------------------
        # Banner background (rounded rectangle)
        banner_data = FancyBboxPatch(
            (0.76, 0.96), 0.147, 0.01,   # (x, y, width, height) in figure coords
            transform=fig.transFigure,
            boxstyle="square,pad=0.02",
            facecolor="#B8B1D9",       
            edgecolor="none",
            alpha=0.95
        )
        fig.patches.append(banner_data)

        banner_legend = FancyBboxPatch(
            (0.76, 0.589), 0.147, 0.01,   # (x, y, width, height) in figure coords
            transform=fig.transFigure,
            boxstyle="square,pad=0.02",
            facecolor="#9FB2A6",       
            edgecolor="none",
            alpha=0.95
        )
        fig.patches.append(banner_legend)

         # Two rounded boxes inside ax_side (coords are in ax_side Axes coordinates)
        self.box_data = FancyBboxPatch(
            (0.0, 0.6), 0.85, 0.39,  # x, y, w, h
            transform=self.ax_side.transAxes,
            boxstyle="round,pad=0.0,rounding_size=0.00",
            facecolor="#EFECF8", edgecolor=(0, 0, 0, 0.25), linewidth=1.0
        )        
        self.ax_side.add_patch(self.box_data)

        self.box_legend = FancyBboxPatch(
            (0.0, 0.01), 0.85, 0.61,
            transform=self.ax_side.transAxes,
            boxstyle="round,pad=0.0,rounding_size=0.0",
            facecolor="#D8E2DC", edgecolor=(0, 0, 0, 0.25), linewidth=1.0
        )
        self.ax_side.add_patch(self.box_legend)

        # --- Titles for the two boxes ---
        fig.text(0.41, 0.978, "Data",
                          transform=self.ax_side.transAxes,
                          ha="center", va="top", fontsize=16, weight="bold",
                          family="Arial", style="italic",color="white",
                          path_effects=[pe.withSimplePatchShadow(offset=(1,-1), alpha=0.4)]
                          )

        fig.text(0.4, 0.607, "Legend",
                          transform=self.ax_side.transAxes,
                          ha="center", va="top", fontsize=16, weight="bold",
                          family="Arial", style="italic",color="white",
                          path_effects=[pe.withSimplePatchShadow(offset=(1,-1), alpha=0.4)]
                          )
        
        # --- Dynamic text that will be updated each frame (inside Data box) ---
        self.txt_data = self.ax_side.text(
            0.18, 0.92, "", transform=self.ax_side.transAxes,
            ha="left", va="top", fontsize=13, color="#222", style="italic", linespacing=1.5   
        )


        # --- Add symbols for legend items (adjust colors to match your plot) ---
        legend_items = [
            ("Quadcopter", "k", "x"),      # black cross
            ("Waypoints", "orange", "o"),  # orange circles
            ("Obstacle", "red", "o"),      # red sphere
            ("Planned Trajectory", "blue", "."), # dashed line 
            ("Actual Trajectory", "gray", "."),      
        ]
    
        y_pos = 0.5
        for label, color, marker in legend_items:
            self.ax_side.plot(
                [0.2], [y_pos], marker=marker,
                color=color, markersize=8, markeredgecolor="black", transform=self.ax_side.transAxes
            )
            self.ax_side.text(
                0.3, y_pos, label,
                transform=self.ax_side.transAxes,
                ha="left", va="center", fontsize=10, color="#222"
            )
            y_pos -= 0.10


        # --- RIGHT CARDS  ---------------------------------------------

        banner_settings = FancyBboxPatch(
            (0.103, 0.96), 0.1361, 0.01,   # (x, y, width, height) in figure coords
            transform=fig.transFigure,
            boxstyle="square,pad=0.02",
            facecolor="#B8A096",       
            edgecolor="none",
            alpha=0.95
        )
        fig.patches.append(banner_settings)

        # --- Left sidebar for Legend ---
        self.ax_left = plt.axes([0.039, 0.01, 0.22, 0.98])
        self.ax_left.axis('off')

        # Legend box
        self.box_legend = FancyBboxPatch(
            (0.2, 0.0), 0.906, 1,
            transform=self.ax_left.transAxes,
            boxstyle="round,pad=0.0,rounding_size=0.00",
            facecolor="#EFDFD8", edgecolor=(0, 0, 0, 0.25), linewidth=1.0,
        )
        self.ax_left.add_patch(self.box_legend)

        fig.text(0.6, 0.988, "Settings",
                            transform=self.ax_left.transAxes,
                            ha="center", va="top", fontsize=16, weight="bold",
                            family="Arial", style="italic",color="white",
                            path_effects=[pe.withSimplePatchShadow(offset=(1,-1), alpha=0.4)]
                            )

        # --- Static settings text (computed once) ---

        # dt        = self.params["time"]["dt"]
        # mass = self.params["quadcopter"]["mass"]
        # arm_length = self.params["quadcopter"]["arm_length"]
        # if (self.params["world"]["wind"]["active"] == "True"):
        #       wind = "Active"
        # else:
        #       wind = "Off"
              
        # input_delay = dt * self.params["time"]["delay_time_step"]

        # settings_str = (
        #     f" Mass: {mass} Kg\n"
        #     f" Arm Length: {arm_length} m\n"            
        #     f" Controller: {self.controller_name}\n"
        #     f" Planner: {self.planner_name}\n"
        #     f" Interpolator: {self.interpolator_name}\n"
        #     f" Sim time step: {dt} s\n" 
        #     f" Controller time step: {self.controller_dt} s\n"
        #     f" Control Input delay: {input_delay} s\n"
        #     f" Wind Disturbances: {wind} \n"
        # )

        # self.ax_left.text(
        #     0.22, 0.92, settings_str, transform=self.ax_left.transAxes,
        #     ha="left", va="top", family="DejaVu Sans Mono", fontsize=10, color="#222", style="italic", linespacing=2
        # )

        # --- Static settings text (computed once) ---

        dt = self.params["time"]["dt"]
        mass = self.params["quadcopter"]["mass"]
        arm_length = self.params["quadcopter"]["arm_length"]

        input_delay = dt * self.params["time"]["delay_time_step"]

        # wind = "On" if self.params["world"]["wind"]["active"] == "True" else "Off"
        if (self.params["world"]["wind"]["active"] == "True"):
              wind = self.params["world"]["wind"]["type"]
        else:
              wind = "Off"


        I_xx = self.params["quadcopter"]["I_xx"]
        I_yy = self.params["quadcopter"]["I_yy"]
        I_zz = self.params["quadcopter"]["I_zz"]
        kf = self.params["quadcopter"]["motor"]["kf"]
        km = self.params["quadcopter"]["motor"]["km"]


        planner_type = self.planner_name
        planner_res  = self.planner_res

        controller_type = self.controller_name
        controller_freq = 1 / self.controller_dt if self.controller_dt else "N/A"

        actuator_limit = self.params["quadcopter"]["limits"]["clip_factor"]

        # ---------- HEADER STYLE ----------
        def draw_header(y, text):
            self.ax_left.text(
                0.25, y, text,
                transform=self.ax_left.transAxes,
                ha="left", va="top",
                fontsize=12,
                weight="bold",
                family="Arial",
                color="#3b3b3b"
            )

        # ---------- CONTENT STYLE ----------
        def draw_block(y, text):
            self.ax_left.text(
                0.27, y, text,
                transform=self.ax_left.transAxes,
                ha="left", va="top",
                family="DejaVu Sans Mono",
                fontsize=9.5,
                color="#222",
                linespacing=1.6
            )


        y = 0.92

        # === SIM PARAMETERS ===
        draw_header(y, "Sim Parameters")
        y -= 0.04
        draw_block(y,
            f"Time Step: {dt} s\n"
            f"Input Delay: {input_delay} s\n"
            f"Actuator limit factor: {actuator_limit}\n"
            f"Wind: {wind}"
        )

        y -= 0.15

        # === QUADCOPTER PARAMETERS ===
        draw_header(y, "Quadcopter Parameters")
        y -= 0.04
        draw_block(y,
            f"Mass: {mass} kg\n"
            f"I_xx: {I_xx}\n"
            f"I_yy: {I_yy}\n"
            f"I_zz: {I_zz}\n"                        
            f"Arm Length: {arm_length} m\n"
            f"Kf: {kf}\n"
            f"Km: {km}"
        )

        y -= 0.25

        # === PLANNER PARAMETERS ===
        draw_header(y, "Planner Parameters")
        y -= 0.04
        draw_block(y,
            f"Type: {planner_type}\n"
            f"Resolution: {planner_res}"
        )

        y -= 0.15

        # === CONTROLLER PARAMETERS ===
        draw_header(y, "Controller Parameters")
        y -= 0.04
        draw_block(y,
            f"Type: {controller_type}\n"
            f"Update Frequency: {controller_freq if controller_freq == 'N/A' else f'{controller_freq:.1f} Hz'}"
        )




        # --- ---------------------------------------------


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


        # === Plot planned trajectory (static) ===
        if self.pos_des_all is not None:
            self.quad_traj_planned = self.ax_anim.plot3D(
                self.pos_des_all[0, 0],   #self.pos_des_all[0, :],  Uncomment last line at end too
                self.pos_des_all[1, 0],
                self.pos_des_all[2, 0],
                'b',  # blue dashed planned path
                linewidth=1.0
            )[0]


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
            u1, u2, u3, u4 = self.controls[0,frame], self.controls[1,frame], self.controls[2,frame], self.controls[3,frame]

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

            control = self.params.get("controller", {}).get("name", "PID")
            
            # Dynamic "Data" box update
            # self.txt_data.set_text(
            #     "  t  = {:>5.2f} s \n  x  = {:>5.2f} m\n  y  = {:>5.2f} m\n  z  = {:>5.2f} m\n u1  = {:>5.2f} (rad/s)2 \n u2  = {:>5.2f} (rad/s)2\n u3  = {:>5.2f} (rad/s)2\n u4  = {:>5.2f} (rad/s)2\n"
            #     .format(time, xt, yt, zt, u1, u2, u3, u4)
            # )            

            self.txt_data.set_text(
                (
                    f" t   = {time:6.2f} s\n"
                    f" x   = {xt:6.2f} m\n"
                    f" y   = {yt:6.2f} m\n"
                    f" z   = {zt:6.2f} m\n"
                    f"u₁  = {u1:6.2f} (rad/s)²\n"
                    f"u₂  = {u2:6.2f} (rad/s)²\n"
                    f"u₃  = {u3:6.2f} (rad/s)²\n"
                    f"u₄  = {u4:6.2f} (rad/s)²"
                )
            )

            if self.pos_des_all is not None:
                self.quad_traj_planned.set_data_3d(self.pos_des_all[0, :frame],
                                        self.pos_des_all[1, :frame],
                                        self.pos_des_all[2, :frame])          


            return 
    


