import pygame
import numpy as np
import math,os
from swerve_drive import SwerveDrive
from PIDController import PIDController
from TrajectoryParser import AutoParser
class SwerveEnvironment:
    """Pygame environment for visualizing and simulating a swerve drive robot."""
    
    def __init__(self, 
                 width=800, 
                 height=600,
                 dt=0.02,
                 pixels_per_meter=100,
                 render_mode=True,
                 start_point = (0,0,0),
                 HW = None):
        """
        Initialize the swerve drive environment.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels
            dt: Time step for simulation (seconds)
            pixels_per_meter: Scale factor for rendering
            render_mode: Whether to initialize pygame rendering
        """
        self.width = width
        self.height = height
        self.dt = dt
        self.pixels_per_meter = pixels_per_meter
        self.render_mode = render_mode
        self.start_point = start_point  # (x, y, theta)
        if HW is not None:
            field_height, field_width = HW
            self.pixels_per_meter = min(self.width/field_width, self.height/field_height)
            self.width = int(field_width * self.pixels_per_meter)
            self.height = int(field_height * self.pixels_per_meter)
        if os.path.exists("pathplanner/field25.png"):
            self.field_image = pygame.image.load("pathplanner/field25.png")
            self.field_image = pygame.transform.scale(self.field_image, (self.width, self.height))
        else:
            self.field_image = None
        
        # Create swerve drive robot
        self.robot = SwerveDrive(
            wheelbase=0.6,
            track_width=0.6,
            mass=50.0,
            moment_of_inertia=5.0,
            drive_tau=0.1,
            steer_tau=0.05,
            max_speed=4.0,
            max_angular_velocity=2*math.pi
        )
        self.robot.reset(*self.start_point)
        
        # Initialize pygame if rendering
        if self.render_mode:
            pygame.init()
            self.screen = pygame.Surface((self.width, self.height))
            # pygame.display.set_caption("Swerve Drive Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (128, 128, 128)
        self.LIGHT_GRAY = (200, 200, 200)
        
        # Trajectory tracking
        self.trajectory = []
        self.max_trajectory_points = 500
        
        # Simulation state
        self.time = 0.0
        self.running = True
        
    def reset(self, x=0.0, y=0.0, theta=0.0):
        """
        Reset the environment to initial state.
        
        Args:
            x: Initial x position (meters)
            y: Initial y position (meters)
            theta: Initial orientation (radians)
            
        Returns:
            Initial state dictionary
        """
        self.robot.reset(x, y, theta)
        self.trajectory = []
        self.time = 0.0
        return self.get_state()
    
    def step(self, vx, vy, omega):
        """
        Step the simulation forward one time step.
        
        Args:
            vx: Desired velocity in x direction (m/s, forward)
            vy: Desired velocity in y direction (m/s, left)
            omega: Desired angular velocity (rad/s, counter-clockwise)
            
        Returns:
            state: Dictionary containing robot state
        """
        # Set desired velocity
        self.robot.set_desired_velocity_world(vx, vy, omega)
        
        # Update robot
        self.robot.update(self.dt)
        
        # Update time
        self.time += self.dt
        
        # Add to trajectory
        robot_state = self.robot.get_state()
        self.trajectory.append((robot_state['x'], robot_state['y']))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
        
        return self.get_state()
    
    def get_state(self):
        """
        Get current state of the robot.
        
        Returns:
            Dictionary with robot state information
        """
        robot_state = self.robot.get_state()
        module_states = self.robot.get_module_states()
        
        return {
            'x': robot_state['x'],
            'y': robot_state['y'],
            'theta': robot_state['theta'],
            'vx': robot_state['vx'],
            'vy': robot_state['vy'],
            'omega': robot_state['omega'],
            'time': self.time,
            'modules': module_states
        }
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x =  x * self.pixels_per_meter
        screen_y = self.height - y * self.pixels_per_meter
        return int(screen_x), int(screen_y)
    
    def draw_grid(self):
        """Draw a grid on the screen."""
        # Draw grid lines every meter
        for i in range(-10, 11):
            # Vertical lines
            x_world = i
            x_screen, y_top = self.world_to_screen(x_world, 10)
            _, y_bottom = self.world_to_screen(x_world, -10)
            if 0 <= x_screen < self.width:
                color = self.GRAY if i != 0 else self.WHITE
                pygame.draw.line(self.screen, color, (x_screen, 0), (x_screen, self.height), 1)
            
            # Horizontal lines
            y_world = i
            x_left, y_screen = self.world_to_screen(-10, y_world)
            x_right, _ = self.world_to_screen(10, y_world)
            if 0 <= y_screen < self.height:
                color = self.GRAY if i != 0 else self.WHITE
                pygame.draw.line(self.screen, color, (0, y_screen), (self.width, y_screen), 1)
    
    def draw_robot(self):
        """Draw the robot and its modules."""
        state = self.robot.get_state()
        x, y, theta = state['x'], state['y'], state['theta']
        
        # Get robot corners
        half_wb = self.robot.wheelbase / 2
        half_tw = self.robot.track_width / 2
        
        corners = [
            (half_wb, half_tw),    # Front left
            (half_wb, -half_tw),   # Front right
            (-half_wb, -half_tw),  # Back right
            (-half_wb, half_tw)    # Back left
        ]
        
        # Rotate and translate corners
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        screen_corners = []
        for cx, cy in corners:
            # Rotate
            rx = cx * cos_theta - cy * sin_theta
            ry = cx * sin_theta + cy * cos_theta
            # Translate
            wx = x + rx
            wy = y + ry
            screen_corners.append(self.world_to_screen(wx, wy))
        
        # Draw robot body
        pygame.draw.polygon(self.screen, self.BLUE, screen_corners, 0)
        pygame.draw.polygon(self.screen, self.WHITE, screen_corners, 2)
        
        # Draw direction indicator (front of robot)
        center_screen = self.world_to_screen(x, y)
        front_x = x + half_wb * cos_theta
        front_y = y + half_wb * sin_theta
        front_screen = self.world_to_screen(front_x, front_y)
        pygame.draw.line(self.screen, self.YELLOW, center_screen, front_screen, 3)
        
        # Draw swerve modules
        module_states = self.robot.get_module_states()
        for name, module in self.robot.modules.items():
            # Module position in robot frame
            mx_robot = module.x_pos
            my_robot = module.y_pos
            
            # Rotate to world frame
            mx_world = x + mx_robot * cos_theta - my_robot * sin_theta
            my_world = y + mx_robot * sin_theta + my_robot * cos_theta
            
            module_screen = self.world_to_screen(mx_world, my_world)
            
            # Draw module base
            pygame.draw.circle(self.screen, self.WHITE, module_screen, 8)
            pygame.draw.circle(self.screen, self.BLACK, module_screen, 8, 2)
            
            # Draw module direction
            module_state = module_states[name]
            module_angle = module_state['angle'] + theta  # Convert to world frame
            module_length = 0.15  # meters
            
            end_x = mx_world + module_length * math.cos(module_angle)
            end_y = my_world + module_length * math.sin(module_angle)
            end_screen = self.world_to_screen(end_x, end_y)
            
            # Color based on speed
            speed_ratio = abs(module_state['speed']) / self.robot.max_speed
            color_intensity = int(255 * min(speed_ratio, 1.0))
            module_color = (color_intensity, 255 - color_intensity, 0)
            
            pygame.draw.line(self.screen, module_color, module_screen, end_screen, 3)
    
    def draw_trajectory(self):
        """Draw the robot's trajectory."""
        if len(self.trajectory) < 2:
            return
        
        screen_points = [self.world_to_screen(x, y) for x, y in self.trajectory]
        pygame.draw.lines(self.screen, self.GREEN, False, screen_points, 2)
    
    def draw_info(self):
        """Draw information overlay."""
        state = self.robot.get_state()
        real_velocity_world = self.robot.velocity_robot_to_world(state['vx'], state['vy'])
        comanded_velocity_world = self.robot.velocity_robot_to_world(self.robot.desired_vx, self.robot.desired_vy)
        info_lines = [
            f"Time: {self.time:.2f}s",
            f"Position: ({state['x']:.2f}, {state['y']:.2f})m",
            f"Heading: {math.degrees(state['theta']):.1f}°",
            f"Velocity: ({real_velocity_world[0]:.2f}, {real_velocity_world[1]:.2f})m/s",
            f"Angular: {state['omega']:.2f}rad/s",
            f"",
            f"Commands:",
            f"Vx: {comanded_velocity_world[0]:.2f}m/s",
            f"Vy: {comanded_velocity_world[1]:.2f}m/s",
            f"Omega: {self.robot.desired_omega:.2f}rad/s",
            f"Accumulated Error: {getattr(self, 'accumulated_error', 0.0):.2f}"
        ]
        
        y_offset = 10
        for line in info_lines:
            text_surface = self.font.render(line, True, self.WHITE)
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25
    
    def render(self):
        """Render the current state."""
        if not self.render_mode:
            return
        
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw elements
        if self.field_image is not None:
            self.screen.blit(self.field_image, (0, 0))
        else:
            self.draw_grid()
        self.draw_trajectory()
        self.draw_robot()
        self.draw_info()
        return pygame.surfarray.array3d(self.screen)
        # Update display
        # pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events. Returns False if window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.reset()
        return True
    
    def close(self):
        """Close the environment."""
        if self.render_mode:
            pygame.quit()

    def run_trajectory(self, trajectory, trans_pidx, trans_pidy, angular_pid, tolerance=0.05, angular_tolerance=0.1):
        """
        Run the robot through a trajectory using PID control.
        
        Args:
            trajectory: List of tuples (x, y, theta, x_dot, y_dot, theta_dot)
                        where positions are in meters/radians and velocities are in m/s or rad/s
            trans_pid: PID controller for translational control (should have update method)
            angular_pid: PID controller for angular control (should have update method)
            tolerance: Position error tolerance in meters
            angular_tolerance: Angular error tolerance in radians
            
        Returns:
            success: Boolean indicating if trajectory was completed
            trajectory_data: List of dictionaries containing state at each step
        """
        time = 0.0
        trajectory_data = []
        
        for waypoint in trajectory:
            # Get current state
            while time<waypoint['time']:
                time += self.dt
                state = self.get_state()
                current_x = state['x']
                current_y = state['y']
                current_theta = state['theta']
                
                # Calculate position errors
                error_x = waypoint['x'] - current_x
                error_y = waypoint['y'] - current_y
                trans_error = math.sqrt(error_x**2 + error_y**2)
                
                # Calculate angular error (shortest angle)
                error_theta = waypoint['theta'] - current_theta
                error_theta = math.atan2(math.sin(error_theta), math.cos(error_theta))
                
                # Convert translational error to robot frame
                cos_theta = math.cos(current_theta)
                sin_theta = math.sin(current_theta)
               
                
                # PID control for translational motion
                vx_correction = trans_pidx.update(error_x,self.dt)
                vy_correction = trans_pidy.update(error_y,self.dt)
                # print(vx_correction,vy_correction, "|",error_x,error_y)
                # PID control for angular motion
                omega_correction = angular_pid.update(error_theta,self.dt)
                
                # Combine feedforward and feedback
                vx_cmd = vx_correction
                vy_cmd = vy_correction
                omega_cmd = omega_correction
                
                # Step simulation
                step_state = self.step(vx_cmd, vy_cmd, omega_cmd)
                trajectory_data.append({
                    'state': step_state,
                    'target': (waypoint['x'], waypoint['y'], waypoint['theta']),
                    'error': (trans_error, error_theta),
                    'command': (vx_cmd, vy_cmd, omega_cmd)
                })
                setattr(self, 'accumulated_error', getattr(self, 'accumulated_error', 0.0) + abs(trans_error) + abs(error_theta))
                # Render if enabled
                if self.render_mode:
                    self.render()
                    if not self.handle_events():
                        return False, trajectory_data
                    self.clock.tick(int(1.0 / self.dt))
                
                # Check if target reached within tolerance
                if trans_error < tolerance and abs(error_theta) < angular_tolerance:
                    continue
        
        return True, trajectory_data
# Example usage
if __name__ == "__main__":
    # Create environment
    start_point = (4,4,0)
    env_HW = (8.21, 16.54)  # FRC field dimensions in meters (height, width)
    env = SwerveEnvironment(width=1000, height=800, dt=0.02,HW = env_HW, start_point = start_point)
    
    # Reset to starting position
    env.reset(start_point[0],start_point[1],start_point[2])
    
    # Simulation loop
    running = True
    t = 0.0

    auto_parser = AutoParser(
        r"c:\Users\chris.reckner\Documents\pythontests\Machine Learning\pathplanner\autos\Wonky 3.auto",
        paths_directory=r"c:\Users\chris.reckner\Documents\pythontests\Machine Learning\pathplanner\paths"
    )
    
    print(f"\nPath names in auto: {auto_parser.get_path_names()}")
    
    # Get summary
    summary = auto_parser.get_trajectory_summary()
    print(f"\nAuto Summary:")
    print(f"  Number of paths: {summary['num_paths']}")
    print(f"  Total duration: {summary['duration']:.2f} seconds")
    print(f"  Total distance: {summary['total_distance']:.2f} meters")
    print(f"  Start position: ({summary['start_position'][0]:.2f}, {summary['start_position'][1]:.2f})")
    print(f"  End position: ({summary['end_position'][0]:.2f}, {summary['end_position'][1]:.2f})")
    
    # Generate combined trajectory
    combined_trajectory = auto_parser.generate_combined_trajectory(dt=0.25)
    
    # Create PID controllers (assuming they are defined elsewhere)
    trans_pidx = PIDController(1,0,0)  # Replace with actual PID controller initialization
    trans_pidy = PIDController(1,0,0)  # Replace with actual PID controller initialization
    angular_pid = PIDController(1,0,0)  # Replace with actual PID controller initialization
    
    # Run the trajectory
    while running:
        success, trajectory_data = env.run_trajectory(combined_trajectory, trans_pidx, trans_pidy, angular_pid)
        # Handle events
        # env.step(.1,.1,.5)
        
        # env.render()
        env.reset(start_point[0],start_point[1],start_point[2])
        # running = env.handle_events()
        if not success: 
            running = False
       
    
    env.close()
