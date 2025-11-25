import json
import numpy as np
import math
from typing import List, Dict, Tuple


class BezierCurve:
    """Represents a cubic Bezier curve segment."""
    
    def __init__(self, p0, p1, p2, p3):
        """
        Initialize a cubic Bezier curve.
        
        Args:
            p0: Start point (x, y)
            p1: First control point (x, y)
            p2: Second control point (x, y)
            p3: End point (x, y)
        """
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
    
    def point_at(self, t):
        """
        Get point on curve at parameter t.
        
        Args:
            t: Parameter from 0 to 1
            
        Returns:
            (x, y) tuple
        """
        t = np.clip(t, 0, 1)
        return (1-t)**3 * self.p0 + \
               3*(1-t)**2*t * self.p1 + \
               3*(1-t)*t**2 * self.p2 + \
               t**3 * self.p3
    
    def derivative_at(self, t):
        """
        Get derivative (tangent) at parameter t.
        
        Args:
            t: Parameter from 0 to 1
            
        Returns:
            (dx, dy) tuple
        """
        t = np.clip(t, 0, 1)
        return 3*(1-t)**2 * (self.p1 - self.p0) + \
               6*(1-t)*t * (self.p2 - self.p1) + \
               3*t**2 * (self.p3 - self.p2)
    
    def length(self, num_samples=100):
        """
        Approximate arc length of the curve.
        
        Args:
            num_samples: Number of samples for approximation
            
        Returns:
            Approximate arc length
        """
        length = 0.0
        prev_point = self.point_at(0)
        
        for i in range(1, num_samples + 1):
            t = i / num_samples
            curr_point = self.point_at(t)
            length += np.linalg.norm(curr_point - prev_point)
            prev_point = curr_point
        
        return length


class TrajectoryParser:
    """Parses PathPlanner .path files and generates trajectories."""
    
    def __init__(self, path_file: str):
        """
        Initialize trajectory parser.
        
        Args:
            path_file: Path to the .path JSON file
        """
        with open(path_file, 'r') as f:
            self.path_data = json.load(f)
        
        self.waypoints = self.path_data['waypoints']
        self.constraints = self.path_data['globalConstraints']
        self.goal_state = self.path_data['goalEndState']
        self.ideal_start = self.path_data.get('idealStartingState', {'velocity': 0, 'rotation': 0.0})
        # print(self.ideal_start)
        self.point_towards_zones = self.path_data.get('pointTowardsZones', [])
        # print(self.ideal_start.get('rotation'), self.goal_state.get('rotation'))
        # Extract constraints
        self.max_velocity = self.constraints['maxVelocity']
        self.max_acceleration = self.constraints['maxAcceleration']
        self.max_angular_velocity = math.radians(self.constraints['maxAngularVelocity'])
        self.max_angular_acceleration = math.radians(self.constraints['maxAngularAcceleration'])
        
        # Build Bezier curves from waypoints
        self.curves = self._build_bezier_curves()
        
    def _build_bezier_curves(self) -> List[BezierCurve]:
        """Build Bezier curve segments from waypoints."""
        curves = []
        
        for i in range(len(self.waypoints) - 1):
            start_wp = self.waypoints[i]
            end_wp = self.waypoints[i + 1]
            
            # Start point
            p0 = (start_wp['anchor']['x'], start_wp['anchor']['y'])
            
            # First control point (from start waypoint's nextControl)
            if start_wp['nextControl'] is not None:
                p1 = (start_wp['nextControl']['x'], start_wp['nextControl']['y'])
            else:
                p1 = p0
            
            # Second control point (from end waypoint's prevControl)
            if end_wp['prevControl'] is not None:
                p2 = (end_wp['prevControl']['x'], end_wp['prevControl']['y'])
            else:
                p2 = (end_wp['anchor']['x'], end_wp['anchor']['y'])
            
            # End point
            p3 = (end_wp['anchor']['x'], end_wp['anchor']['y'])
            
            curves.append(BezierCurve(p0, p1, p2, p3))
        
        return curves
    
    def _calculate_total_path_length(self) -> float:
        """Calculate total path length."""
        return sum(curve.length() for curve in self.curves)
    
    def _get_point_at_distance(self, distance: float) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Get point, tangent, and waypoint index at a given distance along the path.
        
        Args:
            distance: Distance along path
            
        Returns:
            (point, tangent, waypoint_index) tuple
        """
        accumulated_length = 0.0
        
        for i, curve in enumerate(self.curves):
            curve_length = curve.length()
            
            if accumulated_length + curve_length >= distance:
                # Point is on this curve
                target_length = distance - accumulated_length
                t = target_length / curve_length  # Approximate parameter
                
                point = curve.point_at(t)
                tangent = curve.derivative_at(t)
                tangent /= np.linalg.norm(tangent)  # Normalize
                return point, tangent, i,t
            
            accumulated_length += curve_length
        
        # Return last point if distance exceeds path length
        last_curve = self.curves[-1]
        last_point = last_curve.point_at(1.0)
        last_tangent = last_curve.derivative_at(1.0)
        return last_point, last_tangent / np.linalg.norm(last_tangent), len(self.curves) - 1,1
    def _get_rotation_at_distance(self, distance: float, relative_pos: float, default_heading: float,total_length: float) -> float:
        """
        Get rotation at a given distance, considering rotation targets and point-towards zones.
        
        Args:
            distance: Distance along path
            relative_pos: Relative position (0 to 1) along entire path
            default_heading: Default heading from tangent
            
        Returns:
            Rotation in radians
        """
        # Check if we're in a point-towards zone first (higher priority)
        next_zone = None
        min_next_zone_pos = float('inf')
        last_zone = None
        max_last_zone_pos = float('-inf')
        
        for zone in self.point_towards_zones:
            min_pos = zone.get('minWaypointRelativePos', 0.0)/total_length
            max_pos = zone.get('maxWaypointRelativePos', 1.0)/total_length
            
            if min_pos <= relative_pos <= max_pos:
                # Get current position
                point, _, _, _ = self._get_point_at_distance(distance)
                
                # Get target position
                target_x = zone['fieldPosition']['x']
                target_y = zone['fieldPosition']['y']
                
                # Calculate angle to target
                dx = target_x - point[0]
                dy = target_y - point[1]
                angle = math.atan2(dy, dx)
                
                # Apply rotation offset
                offset = math.radians(zone.get('rotationOffset', 0.0))
                return angle + offset
            # If we haven't reached the first zone yet, get the angle command at its minPos
            if relative_pos < min_pos:
                if min_pos < min_next_zone_pos:
                    min_next_zone_pos = min_pos
                    next_zone = zone
            if relative_pos > max_pos:
                if max_pos > max_last_zone_pos:
                    max_last_zone_pos = max_pos
                    last_zone = zone
          
        # Check for rotation targets at waypoints
        point, _, waypoint_idx,relative_pos_on_curve = self._get_point_at_distance(distance)
        
        # Check if we need to interpolate between waypoint rotations
        if waypoint_idx < len(self.waypoints) - 1:
            
            start_wp = self.waypoints[waypoint_idx]
            end_wp = self.waypoints[waypoint_idx + 1]
            
            start_rotation = None
            end_rotation = None
            
            if waypoint_idx == 0:
                if self.ideal_start.get('rotation') is not None:
                    start_rotation = math.radians(self.ideal_start.get('rotation'))
            if waypoint_idx == len(self.waypoints) -2:
                if self.goal_state.get('rotation') is not None:
                    end_rotation = math.radians(self.goal_state.get('rotation'))

            # Get rotation targets if they exist
            if 'rotationTarget' in start_wp and start_wp['rotationTarget'] is not None and start_rotation is None:
                start_rotation = math.radians(start_wp['rotationTarget'].get('rotation', None) or start_wp['rotationTarget'].get('degrees', 0.0))
            
            if 'rotationTarget' in end_wp and end_wp['rotationTarget'] is not None and end_rotation is None:
                end_rotation = math.radians(end_wp['rotationTarget'].get('rotation', None) or end_wp['rotationTarget'].get('degrees', 0.0))
            
            # If both waypoints have rotation targets, interpolate between them
            if start_rotation is not None and end_rotation is not None:
                # Calculate how far we are along this curve segment
                
                        # Interpolate rotation (handle angle wrapping)
                angle_diff = end_rotation - start_rotation
                # Get the current point, tangent, and waypoint index
                return angle_diff*relative_pos_on_curve +start_rotation
            
            # If only end has rotation target and we're close to end, use it
            elif end_rotation is not None:
                # Check if we're in the last portion of the segment
                angle_diff = end_rotation - default_heading
                
        
                # Default to tangent-based heading
                return default_heading + angle_diff*relative_pos_on_curve
        
        return default_heading
    
    def generate_trajectory(self, dt: float = 0.02) -> List[Dict]:
        """
        Generate trajectory based on waypoints and velocity profile.
        
        Args:
            dt: Time step for trajectory points
            
        Returns:
            List of trajectory points with time, position, velocity, and heading
        """
        total_length = self._calculate_total_path_length()
        start_vel = self.ideal_start['velocity']
        end_vel = self.goal_state.get('velocity', 0.0)
        max_vel = self.max_velocity
        max_accel = self.max_acceleration

        # Generate velocity profile
        velocity_profile = self._generate_velocity_profile(total_length, start_vel, end_vel, max_vel, max_accel, dt)
        
        trajectory = []
        accumulated_length = 0.0
        time = 0.0
        
        # Track previous values for calculating omega and angular acceleration
        prev_velocity = start_vel
        prev_theta = math.radians(self.ideal_start.get('rotation', 0.0))
        prev_omega = 0.0
        
        while accumulated_length < total_length:
            # Get current velocity based on the distance traveled
            current_velocity = velocity_profile(accumulated_length)
            
            # Get the current point and tangent
            point, tangent, _, relative_pos_on_curve = self._get_point_at_distance(accumulated_length)
            
            # Calculate relative position along path
            relative_pos = accumulated_length / total_length if total_length > 0 else 0.0
            
            # Get heading from tangent as default
            default_heading = prev_theta
            
            # Get desired rotation considering rotation targets and point-towards zones
            desired_theta = self._get_rotation_at_distance(accumulated_length, relative_pos, default_heading, total_length)
            desired_theta = math.atan2(math.sin(desired_theta), math.cos(desired_theta))  # Normalize angle
            
            # Calculate desired angular velocity
            angle_diff = desired_theta - prev_theta
            # Normalize angle difference to [-pi, pi]
            angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
            desired_omega = angle_diff / dt if dt > 0 else 0.0
            
            # Apply angular acceleration constraint
            max_omega_change = self.max_angular_acceleration * dt
            omega_change = desired_omega - prev_omega
            
            # Limit angular acceleration
            if abs(omega_change) > max_omega_change:
                omega_change = math.copysign(max_omega_change, omega_change)
            
            omega = prev_omega + omega_change
            
            # Apply max angular velocity constraint
            omega = np.clip(omega, -self.max_angular_velocity, self.max_angular_velocity)
            
            # Calculate actual theta based on constrained omega
            theta = prev_theta + omega * dt
            theta = math.atan2(math.sin(theta), math.cos(theta))  # Normalize angle
            
            # Calculate linear acceleration
            acceleration = (current_velocity - prev_velocity) / dt if dt > 0 else 0.0
            
            # Update the trajectory point
            trajectory.append({
                'time': time,
                'x': point[0],
                'y': point[1],
                'vx': tangent[0] * current_velocity,
                'vy': tangent[1] * current_velocity,
                'theta': theta,
                'omega': omega,
                'acceleration': acceleration
            })
            
            # Update for next iteration
            prev_velocity = current_velocity
            prev_theta = theta
            prev_omega = omega
            accumulated_length += current_velocity * dt
            time += dt

        return trajectory
    
    def _generate_velocity_profile(self, total_length: float, start_vel: float, 
                                   end_vel: float, max_vel: float, max_accel: float, dt: float):
        """
        Generate a trapezoidal velocity profile.
        
        Returns:
            Function that takes distance and returns velocity
        """
        # Calculate acceleration and deceleration distances
        accel_dist = (max_vel**2 - start_vel**2) / (2 * max_accel)
        decel_dist = (max_vel**2 - end_vel**2) / (2 * max_accel)
        # print(accel_dist,decel_dist,total_length)
        # print(max_accel,max_vel,start_vel,end_vel)
        # Check if we can reach max velocity
        if accel_dist + decel_dist > total_length:
            # Triangular profile - can't reach max velocity
            # Calculate peak velocity
            peak_vel = math.sqrt((2 * max_accel * total_length + start_vel**2 + end_vel**2) / 2)
            accel_dist = (peak_vel**2 - start_vel**2) / (2 * max_accel)
            decel_dist = (peak_vel**2 - end_vel**2) / (2 * max_accel)
            cruise_dist = 0
            cruise_vel = peak_vel
        else:
            # Trapezoidal profile
            cruise_dist = total_length - accel_dist - decel_dist
            cruise_vel = max_vel
        
        def velocity_at_distance(d):
            if d < 0:
                return start_vel
            elif d < accel_dist:
                # Acceleration phase
                
                return math.sqrt(start_vel**2 + 2 * max_accel * d + 0.01)
            elif d < accel_dist + cruise_dist:
                # Cruise phase
                return cruise_vel
            elif d < total_length:
                # Deceleration phase
                remaining = total_length - d
                return math.sqrt(end_vel**2 + 2 * max_accel * remaining)
            else:
                return end_vel
        
        return velocity_at_distance
    
    def get_trajectory_duration(self) -> float:
        """Get the total duration of the trajectory."""
        trajectory = self.generate_trajectory()
        return trajectory[-1]['time'] if trajectory else 0.0
    
    def save_trajectory(self, output_file: str, dt: float = 0.02):
        """Save trajectory to JSON file."""
        trajectory = self.generate_trajectory(dt)
        
        with open(output_file, 'w') as f:
            json.dump({
                'trajectory': trajectory,
                'metadata': {
                    'duration': trajectory[-1]['time'] if trajectory else 0.0,
                    'length': self._calculate_total_path_length(),
                    'num_points': len(trajectory),
                    'dt': dt,
                    'constraints': self.constraints
                }
            }, f, indent=2)


class AutoParser:
    """Parses PathPlanner .auto files and sequences multiple trajectories."""
    
    def __init__(self, auto_file: str, paths_directory: str = None):
        """
        Initialize auto parser.
        
        Args:
            auto_file: Path to the .auto JSON file
            paths_directory: Directory containing .path files. If None, assumes paths are in ../paths relative to auto file
        """
        with open(auto_file, 'r') as f:
            self.auto_data = json.load(f)
        
        # Determine paths directory
        if paths_directory is None:
            import os
            auto_dir = os.path.dirname(auto_file)
            self.paths_directory = os.path.join(auto_dir, '..', 'paths')
        else:
            self.paths_directory = paths_directory
        
        self.reset_odom = self.auto_data.get('resetOdom', True)
        self.commands = self._parse_commands(self.auto_data.get('command', {}))
        
    def _parse_commands(self, command_node: Dict) -> List[Dict]:
        """
        Recursively parse command structure to extract path commands.
        
        Args:
            command_node: Command node from auto file
            
        Returns:
            List of path names in execution order
        """
        commands = []
        
        if not command_node:
            return commands
        
        cmd_type = command_node.get('type', '')
        cmd_data = command_node.get('data', {})
        
        if cmd_type == 'path':
            # Single path command
            path_name = cmd_data.get('pathName', '')
            if path_name:
                commands.append({'type': 'path', 'name': path_name})
        
        elif cmd_type == 'sequential':
            # Sequential group - execute in order
            sub_commands = cmd_data.get('commands', [])
            for sub_cmd in sub_commands:
                commands.extend(self._parse_commands(sub_cmd))
        
        elif cmd_type == 'parallel':
            # Parallel group - for now, we'll just execute sequentially
            # In a full implementation, you'd need to handle parallel execution
            sub_commands = cmd_data.get('commands', [])
            for sub_cmd in sub_commands:
                commands.extend(self._parse_commands(sub_cmd))
        
        elif cmd_type == 'race':
            # Race group - first command to finish wins
            # For simplicity, execute first command only
            sub_commands = cmd_data.get('commands', [])
            if sub_commands:
                commands.extend(self._parse_commands(sub_commands[0]))
        
        elif cmd_type == 'deadline':
            # Deadline group - execute with time limit
            sub_commands = cmd_data.get('commands', [])
            for sub_cmd in sub_commands:
                commands.extend(self._parse_commands(sub_cmd))
        
        # Ignore other command types (wait, named, etc.) for trajectory purposes
        
        return commands
    
    def get_path_names(self) -> List[str]:
        """Get list of all path names in execution order."""
        return [cmd['name'] for cmd in self.commands if cmd['type'] == 'path']
    
    def generate_combined_trajectory(self, dt: float = 0.02) -> List[Dict]:
        """
        Generate a single combined trajectory from all paths in the auto.
        
        Args:
            dt: Time step for trajectory points
            
        Returns:
            List of trajectory points for the entire autonomous routine
        """
        import os
        
        combined_trajectory = []
        time_offset = 0.0
        last_position = (0.0, 0.0)
        last_theta = 0.0
        last_velocity = 0.0
        
        path_names = self.get_path_names()
        
        for i, path_name in enumerate(path_names):
            # Construct path file path
            path_file = os.path.join(self.paths_directory, f"{path_name}.path")
            
            if not os.path.exists(path_file):
                print(f"Warning: Path file not found: {path_file}")
                continue
            
            # Parse the path
            try:
                parser = TrajectoryParser(path_file)
                trajectory = parser.generate_trajectory(dt)
                
                if not trajectory:
                    continue
                
                # For the first path or if resetOdom is true, use trajectory as-is
                if i == 0 or self.reset_odom:
                    # Add trajectory points with time offset
                    for point in trajectory:
                        combined_point = point.copy()
                        combined_point['time'] += time_offset
                        combined_point['path_name'] = path_name
                        combined_point['path_index'] = i
                        combined_trajectory.append(combined_point)
                else:
                    # Adjust trajectory to start from last position
                    # Calculate offset needed
                    first_point = trajectory[0]
                    x_offset = last_position[0] - first_point['x']
                    y_offset = last_position[1] - first_point['y']
                    theta_offset = last_theta - first_point['theta']
                    
                    # Transform all points
                    cos_offset = math.cos(theta_offset)
                    sin_offset = math.sin(theta_offset)
                    
                    for point in trajectory:
                        # Rotate and translate position
                        rel_x = point['x'] - first_point['x']
                        rel_y = point['y'] - first_point['y']
                        
                        rot_x = rel_x * cos_offset - rel_y * sin_offset
                        rot_y = rel_x * sin_offset + rel_y * cos_offset
                        
                        new_x = last_position[0] + rot_x
                        new_y = last_position[1] + rot_y
                        
                        # Rotate velocities
                        rot_vx = point['vx'] * cos_offset - point['vy'] * sin_offset
                        rot_vy = point['vx'] * sin_offset + point['vy'] * cos_offset
                        
                        combined_point = {
                            'time': point['time'] + time_offset,
                            'x': new_x,
                            'y': new_y,
                            'theta': point['theta'] + theta_offset,
                            'vx': rot_vx,
                            'vy': rot_vy,
                            'omega': point.get('omega', 0.0),
                            'path_name': path_name,
                            'path_index': i
                        }
                        combined_trajectory.append(combined_point)
                
                # Update offsets for next path
                if trajectory:
                    last_point = combined_trajectory[-1]
                    time_offset = last_point['time']
                    last_position = (last_point['x'], last_point['y'])
                    last_theta = last_point['theta']
                    last_velocity = math.sqrt(last_point['vx']**2 + last_point['vy']**2)
                
            except Exception as e:
                print(f"Error processing path '{path_name}': {e}")
                continue
        
        return combined_trajectory
    
    def save_combined_trajectory(self, output_file: str, dt: float = 0.02):
        """Save combined trajectory to JSON file."""
        trajectory = self.generate_combined_trajectory(dt)
        
        with open(output_file, 'w') as f:
            json.dump({
                'trajectory': trajectory,
                'metadata': {
                    'duration': trajectory[-1]['time'] if trajectory else 0.0,
                    'num_points': len(trajectory),
                    'num_paths': len(self.get_path_names()),
                    'path_names': self.get_path_names(),
                    'dt': dt,
                    'reset_odom': self.reset_odom
                }
            }, f, indent=2)
    
    def get_trajectory_summary(self) -> Dict:
        """Get summary information about the autonomous routine."""
        trajectory = self.generate_combined_trajectory()
        
        if not trajectory:
            return {
                'duration': 0.0,
                'num_paths': 0,
                'path_names': [],
                'total_distance': 0.0
            }
        
        # Calculate total distance
        total_distance = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i]['x'] - trajectory[i-1]['x']
            dy = trajectory[i]['y'] - trajectory[i-1]['y']
            total_distance += math.sqrt(dx**2 + dy**2)
        
        return {
            'duration': trajectory[-1]['time'],
            'num_paths': len(self.get_path_names()),
            'path_names': self.get_path_names(),
            'total_distance': total_distance,
            'start_position': (trajectory[0]['x'], trajectory[0]['y']),
            'end_position': (trajectory[-1]['x'], trajectory[-1]['y'])
        }


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Parse an auto file with multiple paths
    print("=" * 60)
    print("Auto Routine Parser - Multiple Paths")
    print("=" * 60)
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
    
    # Save combined trajectory
    # auto_parser.save_combined_trajectory("auto_trajectory_output.json")
    print("\nCombined trajectory saved to auto_trajectory_output.json")
    
    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data from combined trajectory
    auto_times = [p['time'] for p in combined_trajectory]
    auto_xs = [p['x'] for p in combined_trajectory]
    auto_ys = [p['y'] for p in combined_trajectory]
    auto_thetas = [math.degrees(p['theta']) for p in combined_trajectory]
    auto_velocities = [math.sqrt(p['vx']**2 + p['vy']**2) for p in combined_trajectory]
    
    # Plot 1: Combined path trajectory with color coding by path
    path_names = auto_parser.get_path_names()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(path_names)))
    
    for i, path_name in enumerate(path_names):
        path_points = [p for p in combined_trajectory if p.get('path_name') == path_name]
        if path_points:
            path_xs = [p['x'] for p in path_points]
            path_ys = [p['y'] for p in path_points]
            axes[0, 0].plot(path_xs, path_ys, '-', linewidth=2, color=colors[i], label=path_name)
    
    axes[0, 0].plot(auto_xs[0], auto_ys[0], 'go', markersize=12, label='Start', zorder=5)
    axes[0, 0].plot(auto_xs[-1], auto_ys[-1], 'ro', markersize=12, label='End', zorder=5)
    axes[0, 0].set_xlabel('X (meters)')
    axes[0, 0].set_ylabel('Y (meters)')
    axes[0, 0].set_title('Combined Auto Trajectory')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Plot 2: Full trajectory (single line)
    axes[0, 1].plot(auto_xs, auto_ys, 'b-', linewidth=2)
    axes[0, 1].plot(auto_xs[0], auto_ys[0], 'go', markersize=12, label='Start')
    axes[0, 1].plot(auto_xs[-1], auto_ys[-1], 'ro', markersize=12, label='End')
    
    # Mark transitions between paths
    for i in range(1, len(path_names)):
        transition_points = [p for p in combined_trajectory if p.get('path_index') == i]
        if transition_points:
            tp = transition_points[0]
            axes[0, 1].plot(tp['x'], tp['y'], 'ko', markersize=6)
    
    axes[0, 1].set_xlabel('X (meters)')
    axes[0, 1].set_ylabel('Y (meters)')
    axes[0, 1].set_title('Full Auto Path')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Plot 3: Velocity profile over time
    axes[1, 0].plot(auto_times, auto_velocities, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    axes[1, 0].set_title('Velocity Profile')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Mark path transitions
    for i in range(1, len(path_names)):
        transition_points = [p for p in combined_trajectory if p.get('path_index') == i]
        if transition_points:
            axes[1, 0].axvline(x=transition_points[0]['time'], color='gray', linestyle='--', alpha=0.5)
    
    # Plot 4: Heading over time
    axes[1, 1].plot(auto_times, auto_thetas, 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Heading (degrees)')
    axes[1, 1].set_title('Heading Profile')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark path transitions
    for i in range(1, len(path_names)):
        transition_points = [p for p in combined_trajectory if p.get('path_index') == i]
        if transition_points:
            axes[1, 1].axvline(x=transition_points[0]['time'], color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualization complete!")
