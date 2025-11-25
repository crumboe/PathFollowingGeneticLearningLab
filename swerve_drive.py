import numpy as np
import math


class SwerveModule:
    """Simulates a single swerve drive module with first-order response."""
    
    def __init__(self, x_pos, y_pos, drive_tau=0.5, steer_tau=0.05):
        """
        Initialize a swerve module.
        
        Args:
            x_pos: X position of module relative to robot center (meters)
            y_pos: Y position of module relative to robot center (meters)
            drive_tau: Time constant for drive motor first-order response (seconds)
            steer_tau: Time constant for steering motor first-order response (seconds)
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.drive_tau = drive_tau
        self.steer_tau = steer_tau
        
        # Current states
        self.current_speed = 0.0  # m/s
        self.current_angle = 0.0  # radians
        
        # Target states
        self.target_speed = 0.0
        self.target_angle = 0.0
    
    def set_target(self, speed, angle):
        """Set target speed and angle for the module."""
        self.target_speed = speed
        self.target_angle = angle
    
    def update(self, dt):
        """Update module state using first-order response."""
        # First-order response: dx/dt = (target - current) / tau
        # Solution: current = current + (target - current) * (1 - e^(-dt/tau))
        
        drive_alpha = 1 - math.exp(-dt / self.drive_tau)
        steer_alpha = 1 - math.exp(-dt / self.steer_tau)
        # print(self.current_speed, self.target_speed, drive_alpha)
        self.current_speed += (self.target_speed - self.current_speed) * drive_alpha
        # print(self.current_angle, self.target_angle, steer_alpha)
        # Handle angle wrapping for shortest path
        angle_diff = self.target_angle - self.current_angle
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        self.current_angle += angle_diff * steer_alpha
        
        # Normalize angle to [-pi, pi]
        self.current_angle = math.atan2(math.sin(self.current_angle), math.cos(self.current_angle))
    
    def get_velocity_contribution(self):
        """Get the velocity contribution of this module in robot frame."""
        vx = self.current_speed * math.cos(self.current_angle)
        vy = self.current_speed * math.sin(self.current_angle)
        return vx, vy
    def get_acceleration_contribution(self, dt):
        """Get the acceleration contribution of this module in robot frame."""
        # Assuming constant acceleration over dt
        ax = (self.target_speed - self.current_speed) / dt * math.cos(self.current_angle)
        ay = (self.target_speed - self.current_speed) / dt * math.sin(self.current_angle)
        return ax, ay
    
   

class SwerveDrive:
    """Simulates a 4-wheel swerve drive robot with momentum."""
    
    def __init__(self, 
                 wheelbase=0.6,  # meters, front-back distance
                 track_width=0.6,  # meters, left-right distance
                 mass=50.0,  # kg
                 moment_of_inertia=5.0,  # kg*m^2
                 drive_tau=0.1,  # drive motor time constant
                 steer_tau=0.05,  # steering motor time constant
                 max_speed=4.0,  # m/s
                 max_angular_velocity=2*math.pi):  # rad/s
        """
        Initialize swerve drive robot.
        
        Args:
            wheelbase: Distance between front and back wheels (meters)
            track_width: Distance between left and right wheels (meters)
            mass: Robot mass (kg)
            moment_of_inertia: Rotational inertia (kg*m^2)
            drive_tau: Drive motor time constant (seconds)
            steer_tau: Steering motor time constant (seconds)
            max_speed: Maximum module speed (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        self.wheelbase = wheelbase
        self.track_width = track_width
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.max_speed = max_speed
        self.max_angular_velocity = max_angular_velocity
        
        # Create four swerve modules: FL, FR, BL, BR
        half_wb = wheelbase / 2
        half_tw = track_width / 2
        
        self.modules = {
            'FL': SwerveModule(half_wb, half_tw, drive_tau, steer_tau),
            'FR': SwerveModule(half_wb, -half_tw, drive_tau, steer_tau),
            'BL': SwerveModule(-half_wb, half_tw, drive_tau, steer_tau),
            'BR': SwerveModule(-half_wb, -half_tw, drive_tau, steer_tau)
        }
        
        # Robot state
        self.x = 0.0  # meters
        self.y = 0.0  # meters
        self.theta = 0.0  # radians
        
        # Velocities (with momentum)
        self.vx = 0.0  # m/s in robot frame
        self.vy = 0.0  # m/s in robot frame
        self.omega = 0.0  # rad/s
        
        # Desired velocities
        self.desired_vx = 0.0
        self.desired_vy = 0.0
        self.desired_omega = 0.0
    def set_desired_velocity_world(self, world_vx, world_vy, omega):
        """
        Set desired robot velocities in world frame.
        
        Args:
            world_vx: Desired velocity in world x direction (m/s)
            world_vy: Desired velocity in world y direction (m/s)
            omega: Desired angular velocity (rad/s, counter-clockwise is positive)
        """
        # Convert world frame velocities to robot frame
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        robot_vx = world_vx * cos_theta + world_vy * sin_theta
        robot_vy = -world_vx * sin_theta + world_vy * cos_theta
        
        self.set_desired_velocity(robot_vx, robot_vy, omega)

    def set_desired_velocity(self, vx, vy, omega):
        """
        Set desired robot velocities.
        
        Args:
            vx: Desired velocity in x direction (m/s, forward is positive)
            vy: Desired velocity in y direction (m/s, left is positive)
            omega: Desired angular velocity (rad/s, counter-clockwise is positive)
        """
        self.desired_vx = vx
        self.desired_vy = vy
        self.desired_omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)
    def velocity_robot_to_world(self, vx, vy):
        """Convert robot frame velocities to world frame."""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        world_vx = vx * cos_theta - vy * sin_theta
        world_vy = vx * sin_theta + vy * cos_theta
        
        return world_vx, world_vy
    def _inverse_kinematics(self, vx, vy, omega):
        """
        Calculate module states from desired chassis velocities.
        
        Args:
            vx: Robot velocity in x direction (m/s)
            vy: Robot velocity in y direction (m/s)
            omega: Robot angular velocity (rad/s)
        
        Returns:
            Dictionary of (speed, angle) tuples for each module
        """
        module_states = {}
        
        for name, module in self.modules.items():
            # Calculate velocity at module position due to rotation
            module_angle = math.atan2(module.y_pos, module.x_pos)
            module_radius = math.sqrt(module.x_pos**2 + module.y_pos**2)
            tangential_velocity = omega * module_radius
            rotation_vx = tangential_velocity * -math.sin(module_angle)
            rotation_vy = tangential_velocity * math.cos(module_angle)
            
            # Total velocity at module
            module_vx = vx + rotation_vx
            module_vy = vy + rotation_vy
            
            # Convert to speed and angle
            speed = math.sqrt(module_vx**2 + module_vy**2)
            angle = math.atan2(module_vy, module_vx)
            
            # Limit speed
            if speed > self.max_speed:
                speed = self.max_speed
            
            module_states[name] = (speed, angle)
        
        return module_states
    
    def _calculate_forces(self, dt):
        """Calculate forces and torque from module accelerations."""
        total_fx = 0.0
        total_fy = 0.0
        total_torque = 0.0
        
        for name, module in self.modules.items():
            # Get module's velocity contribution
            module_ax, module_ay = module.get_acceleration_contribution(dt)
            
            # Calculate force (simplified: proportional to velocity difference)
            # In reality, this would involve traction models
            force_x = module_ax * self.mass/len(self.modules.items())  # Scaling factor
            force_y = module_ay * self.mass/len(self.modules.items())
            
            total_fx += force_x
            total_fy += force_y
            
            # Calculate torque contribution
            # torque = r × F
            torque = module.x_pos * force_y - module.y_pos * force_x
            total_torque += torque
        
        return total_fx, total_fy, total_torque
    
    def update(self, dt):
        """
        Update robot state for one time step.
        
        Args:
            dt: Time step (seconds)
        """
        # Calculate desired module states
        module_states = self._inverse_kinematics(
            self.desired_vx, 
            self.desired_vy, 
            self.desired_omega
        )
        
        # Set targets for each module
        for name, (speed, angle) in module_states.items():
            self.modules[name].set_target(speed, angle)
        
        vx = 0.0
        vy = 0.0
        omega = 0.0
        # Update each module (first-order response)
        for module in self.modules.values():
            module.update(dt)
            vx += module.current_speed * math.cos(module.current_angle)
            vy += module.current_speed * math.sin(module.current_angle)
            # Calculate angular velocity contribution accounting for module's distance from center
            module_radius = math.sqrt(module.x_pos**2 + module.y_pos**2)
            module_angle = math.atan2(module.y_pos, module.x_pos)
            tangential_velocity = (module.current_speed * math.cos(math.radians(90)+module.current_angle - module_angle))
            if module_radius > 0:
                omega += tangential_velocity / module_radius
        self.vx = vx / len(self.modules)
        self.vy = vy / len(self.modules)
        self.omega = omega / len(self.modules)

        # self.vx = (vx - self.vx) * 0.1 + self.vx
        # self.vy = (vy - self.vy) * 0.1 + self.vy
        # self.omega = (omega - self.omega) * 0.1 + self.omega
        # Apply damping to simulate friction
        damping = 1
        self.vx *= damping
        self.vy *= damping
        self.omega *= damping
        
        # Convert robot frame velocities to world frame
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        
        world_vx = self.vx * cos_theta - self.vy * sin_theta
        world_vy = self.vx * sin_theta + self.vy * cos_theta
        
        # Update position and orientation
        self.x += world_vx * dt
        self.y += world_vy * dt
        self.theta += self.omega * dt
        
        # Normalize theta to [-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
    
    def get_state(self):
        """Get current robot state."""
        return {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'vx': self.vx,
            'vy': self.vy,
            'omega': self.omega
        }
    
    def get_module_states(self):
        """Get current state of all modules."""
        states = {}
        for name, module in self.modules.items():
            states[name] = {
                'speed': module.current_speed,
                'angle': module.current_angle,
                'target_speed': module.target_speed,
                'target_angle': module.target_angle
            }
        return states
    
    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Reset robot to initial state."""
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.desired_vx = 0.0
        self.desired_vy = 0.0
        self.desired_omega = 0.0
        
        for module in self.modules.values():
            module.current_speed = 0.0
            module.current_angle = 0.0
            module.target_speed = 0.0
            module.target_angle = 0.0


# Example usage
if __name__ == "__main__":
    # Create swerve drive robot
    robot = SwerveDrive(
        wheelbase=0.6,
        track_width=0.6,
        mass=50.0,
        moment_of_inertia=5.0
    )
    
    # Simulate for 5 seconds
    dt = 0.02  # 50 Hz
    time = 0.0
    duration = 5.0
    
    # Command: move forward and rotate
    robot.set_desired_velocity(vx=1.0, vy=0.5, omega=0.5)
    
    print(f"{'Time':<8} {'X':<10} {'Y':<10} {'Theta':<10} {'Vx':<10} {'Vy':<10} {'Omega':<10}")
    print("-" * 70)
    
    while time < duration:
        robot.update(dt)
        state = robot.get_state()
        
        if time % 0.5 < dt:  # Print every 0.5 seconds
            print(f"{time:<8.2f} {state['x']:<10.3f} {state['y']:<10.3f} "
                  f"{state['theta']:<10.3f} {state['vx']:<10.3f} "
                  f"{state['vy']:<10.3f} {state['omega']:<10.3f}")
        
        time += dt
    
    print("\nFinal module states:")
    module_states = robot.get_module_states()
    for name, state in module_states.items():
        print(f"{name}: speed={state['speed']:.3f} m/s, angle={math.degrees(state['angle']):.1f}°")
