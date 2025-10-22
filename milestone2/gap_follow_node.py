"""
Gap Following Algorithm Implementation for F1/10 Autonomous Racing

This module implements a gap-following algorithm that processes LIDAR data to identify
the largest navigable gap and steers the vehicle through it. The algorithm includes
disparity extension for obstacle avoidance and dynamic speed control.

Key Features:
- Real-time LIDAR data processing for gap detection
- Disparity extension to handle sudden depth changes (obstacles)
- Smooth steering control to prevent oscillations
- Dynamic speed adjustment based on available clearance
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from math import *


class GapFollowing(Node):
    # GAP FOLLOWER VARIABLES
    car_width = 0.3  # meters
    lookahead_distance = 1.0  # meters
    safety_buffer = 0.1  # meters
    speed = 1.0 #meters / second
    disparity_threshold = 3.0 # meters
    disparity_factor = 2.5 # scaling factor for disparity extension
    dynamic_speed_enabled = 0 # when enabled the car speeds up dynamically based off amount of free space available
    last_angle = 0.0 # the last published steering angle, for smooth driving
    disparity_safety_buffer = 15 # degrees
    max_speed = 1.5 # m/s
    min_speed = 1.0 # m/s

    def __init__(self):
        """
        Initialize the Gap Following ROS2 Node
        
        Sets up:
        - Subscription to LIDAR scan data on /scan topic
        - Publisher for vehicle control commands on /drive_raw topic
        - Node logging for debugging and monitoring
        
        The node operates in a reactive control loop: receive LIDAR data -> 
        process gaps -> compute steering angle -> publish control commands
        """
        super().__init__('gap_follow_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive_raw', 10)
        self.get_logger().info("GapFollowing node initialized and subscribed to /scan")

    def listener_callback(self, msg):
        """
        Main LIDAR Data Processing Callback
        
        This is the core control loop that executes every time new LIDAR data arrives.
        
        Process Flow:
        1. Receive LaserScan message with range data
        2. Apply disparity extension to handle obstacles and sudden depth changes
        3. Run gap-following algorithm to find the best navigable gap
        4. If a valid gap is found, compute steering angle and publish control commands
        5. Log results for monitoring and debugging
        
        Args:
            msg (LaserScan): ROS2 LaserScan message containing range data from LIDAR sensor
        """
        #When we get LIDAR data, calculate the best gap
        self.get_logger().info(f"Received LaserScan with {len(msg.ranges)} ranges")
        disparity_ranges = self.disparity_extender(msg.ranges, 5)
        steering_angle = self.gap_following(disparity_ranges)

        #Ensure there's a gap to follow
        if steering_angle is not None:
            self.vehicleControl(steering_angle, disparity_ranges)
            self.get_logger().info(f"Steering Angle Command: {steering_angle:.2f} rad")
        else:
            self.get_logger().warn("No valid gap found!")

    def gap_following(self, lidar_ranges):
        """
        Core Gap Detection and Navigation Algorithm
        
        This function implements the main gap-following logic to find the largest
        navigable gap and compute an appropriate steering angle.
        
        Algorithm Steps:
        1. Scan through LIDAR ranges to identify potential gaps (distances > lookahead_distance)
        2. Group consecutive points into gap segments
        3. Filter gaps by minimum width requirements (car_width + safety_buffer)
        4. Calculate gap width in angular space and verify it's wide enough for the vehicle
        5. Select the largest gap by number of LIDAR points
        6. Target the middle of the selected gap
        7. Compute steering angle and apply smoothing to prevent oscillations
        
        Gap Width Calculation:
        - Converts LIDAR indices to angular positions (-135° to +135°, 270° FOV)
        - Uses geometric relationship: min_width_angle = atan2(physical_width, distance)
        - Ensures gap is physically wide enough for the car to pass through
        
        Steering Smoothing:
        - Limits steering angle changes per cycle to prevent jerky motion
        - Maintains smooth trajectory while following the gap
        
        Args:
            lidar_ranges (list): Processed LIDAR range data (after disparity extension)
            
        Returns:
            float: Smoothed steering angle in radians (None if no valid gap found)
        """
        min_gap_width = self.car_width + 2 * self.safety_buffer
        gaps = []
        current_gap = []

        for angle_idx, distance in enumerate(lidar_ranges):
            if distance > self.lookahead_distance:
                # we store angle index of gap, not distance
                # because we know the range array properties and can get the distance back
                current_gap.append(angle_idx)
            else:
                if current_gap: #If we have a gap
                    #270 degrees over 1080 LIDAR indices - 135 (half 270) such that center is 0
                    start_angle = current_gap[0] * 270 / 1080 - 135 
                    end_angle = current_gap[-1] * 270 / 1080 - 135
                    gap_width = abs(end_angle - start_angle)

                    # check if the gap is wide enough
                    if gap_width > degrees(atan2(min_gap_width, min(lidar_ranges[current_gap[0]], lidar_ranges[current_gap[-1]]))):
                        gaps.append(current_gap)
                    current_gap = []

        # Handle last gap if still open
        if current_gap:
            #270 degrees over 1080 LIDAR indices - 135 (half 270) such that center is 0
            start_angle = current_gap[0] * 270 / 1080 - 135
            end_angle = current_gap[-1] * 270 / 1080 - 135
            gap_width = abs(end_angle - start_angle)

            #Checks that the gap width (in degrees) is large enough for the car. The min width angle is calculated by finding the angle between the shorter gap edge distance and the minimum width 
            if gap_width > degrees(atan2(min_gap_width, min(lidar_ranges[current_gap[0]], lidar_ranges[current_gap[-1]]))):
                gaps.append(current_gap)

        if not gaps: #If no gaps found
            return None

        # Pick the largest gap
        best_gap = max(gaps, key=len)
        middle_idx = best_gap[len(best_gap) // 2] #Target the middle of the largest gap

        # Convert index to angle in radians
        angle = (middle_idx * 270 / 1080 - 135) * pi / 180
        distance = lidar_ranges[middle_idx]
        lookahead_distance = min(distance, self.lookahead_distance)

        #steering_angle = atan2(2 * self.car_width * sin(angle), lookahead_distance)

        steering_angle = 0.5 * angle

        # experiment: try smoothening steering angle to prevent snake-like motion
        max_delta = 0.05  # rad per cycle
        delta = steering_angle - self.last_angle
        delta = max(min(delta, max_delta), -max_delta)
        smoothed_angle = self.last_angle + delta

        self.last_angle = smoothed_angle
       

        #Log some useful info
        self.get_logger().info(
            f"\nGap count: {len(gaps)}"
            f"\nBest gap size: {len(best_gap)}"
            f"\nMidpoint angle: {degrees(angle):.2f} deg"
            f"\nDistance: {distance:.2f} m"
            f"\nSteering Angle: {degrees(smoothed_angle):.2f} deg"
            # f"\nSteering Angle: {degrees(steering_angle):.2f} deg"
        )

        return smoothed_angle
        # return steering_angle

    def disparity_extender(self, lidar_ranges, disparity_threshold=0.5):
        """
        Disparity Extension Algorithm for Obstacle Safety
        
        This function processes LIDAR data to extend the apparent size of obstacles
        when there are sudden depth changes (disparities). This prevents the vehicle
        from attempting to navigate through gaps that appear open but are actually
        blocked by obstacles at different distances.
        
        Problem Solved:
        - LIDAR beam spacing creates "false gaps" near obstacle edges
        - Vehicle might try to thread between obstacles that are too close
        - Sudden depth changes indicate obstacle boundaries that need safety margins
        
        Algorithm:
        1. Scan adjacent LIDAR points for sudden distance changes (> disparity_threshold)
        2. Identify which side has the closer obstacle (potential collision risk)
        3. Calculate angular width needed for safe passage based on car dimensions
        4. Extend the closer distance value to neighboring beams to "grow" the obstacle
        5. Apply additional safety buffer around extended regions
        
        Mathematical Approach:
        - Angular extension width = atan2(car_width/disparity_factor, closer_distance)
        - Number of beams to extend = (angular_width * total_beams) / FOV_angle
        - Direction of extension depends on which side has the closer obstacle
        
        Args:
            lidar_ranges (list): Raw LIDAR range measurements
            disparity_threshold (float): Minimum distance change to trigger extension (default: 0.5m)
            
        Returns:
            list: Modified range data with extended obstacle regions for safer navigation
        """
        extended_ranges = list(lidar_ranges)  # copy of original values

        for i in range(len(lidar_ranges) - 1):
            d1 = lidar_ranges[i]
            d2 = lidar_ranges[i + 1]

            # check for sudden jumps 
            if abs(d1 - d2) > self.disparity_threshold:
                # Find which side is closer (the obstacle)
                closer_distance = min(d1, d2)

                # Calculate how many beams to extend based on disparity factor
                # note: for now this will be half the cars width, but we can change it later if it doesnt work well
                angle_width = atan2(self.car_width / self.disparity_factor, closer_distance)  
                num_beams_to_extend = int((angle_width * 1080) / (270 * pi/180))

                # Apply safety buffer
                num_beams_to_extend += int((self.disparity_safety_buffer * 1080) / 270)

                if d1 < d2:
                    # Obstacle is on the left side of the disparity
                    for k in range(1, num_beams_to_extend + 1):
                        if i + k < len(extended_ranges):
                            extended_ranges[i + k] = min(extended_ranges[i + k], closer_distance)
                else:
                    # Obstacle is on the right side of the disparity
                    for k in range(1, num_beams_to_extend + 1):
                        if i - k >= 0:
                            extended_ranges[i - k] = min(extended_ranges[i - k], closer_distance)

        return extended_ranges



    def vehicleControl(self, angle, lidar_ranges=None):
        """
        Vehicle Control Command Generation
        
        Converts the computed steering angle into actual vehicle control commands
        with safety limits and dynamic speed adjustment.
        
        Control Strategy:
        1. Clamp steering angle within vehicle's physical limits (±24°)
        2. Calculate dynamic speed based on:
           - Steering angle (slow down for sharp turns)
           - Forward clearance (slow down if obstacles ahead)
           - User-defined speed settings and limits
        3. Publish Ackermann steering commands to the vehicle
        
        Speed Control Logic:
        - Base speed from configuration
        - Angle factor: reduce speed proportionally with steering angle magnitude
        - Distance factor: increase speed if forward path is clear, reduce if obstacles close
        - Dynamic mode: applies both factors, Static mode: uses minimum safe speed
        - Final speed clamped between min_speed and max_speed limits
        
        Forward Clearance Assessment:
        - Examines central LIDAR beams (±20 beams from center ≈ ±5°)
        - Uses minimum distance in forward window for conservative obstacle detection
        - Scales speed based on available forward clearance distance
        
        Args:
            angle (float): Desired steering angle in radians
            lidar_ranges (list, optional): LIDAR data for forward clearance assessment
        
        Publishes:
            AckermannDriveStamped message with steering_angle and speed commands
        """
        max_angle = 0.4189  # rad (~24°)
        newSteeringAngle = max(min(angle, max_angle), -max_angle)

        # speed scaling from research paper, dk if this works on car but does well in sim
        speed = self.speed  

        # Slow down if turning sharply
        angle_factor = max(0.5, 1.0 - abs(newSteeringAngle) / max_angle)

        # Check forward clearance (~ use the central lidar beam and a bit of a deviation)
        if lidar_ranges:
            center_idx = len(lidar_ranges) // 2
            forward_window = 20  # ~5° each side
            forward_dist = min(lidar_ranges[center_idx - forward_window : center_idx + forward_window])
            distance_factor = min(1.5, forward_dist / 3.0)  # scale up if >5m clear
        else:
            distance_factor = 1.0

        # dynamic speed calc
        if self.dynamic_speed_enabled:
            dynamic_speed = speed * angle_factor * distance_factor
            dynamic_speed = max(self.min_speed, min(dynamic_speed, self.max_speed))  # clamp between 0.8 and 1.5 m/s
        else: 
            dynamic_speed = self.min_speed

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = newSteeringAngle
        drive_msg.drive.speed = dynamic_speed
        self.publisher.publish(drive_msg)

        self.get_logger().info(
            f"Published Steering: {newSteeringAngle:.2f} rad, "
            f"Speed: {dynamic_speed:.2f} m/s"
        )


def main(args=None):
    """
    Main Entry Point for Gap Following Node
    
    Standard ROS2 node lifecycle:
    1. Initialize ROS2 communications
    2. Create and configure the GapFollowing node
    3. Enter the ROS2 event loop (spin) to process callbacks
    4. Clean shutdown when interrupted (Ctrl+C)
    
    Args:
        args: Command line arguments (passed from ROS2 launch system)
    """
    rclpy.init(args=args)
    gap_following = GapFollowing()
    rclpy.spin(gap_following)
    gap_following.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
