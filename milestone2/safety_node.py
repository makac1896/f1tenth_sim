import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
import math


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')

         # Store latest odometry and scan data
        self.latest_odom = None
        self.latest_scan = None
        self.latest_drive_raw = None
        self.linear_x_speed = 0
        
        # Subscribers -------------
        self.subscription = self.create_subscription(
            LaserScan, 
            '/scan', # topic
            self.scan_callback,
            10 # QoS, Quality of Service, limits the amount of queued msgs
        )

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            '/drive_raw', # subscribe to drive topic from car control
            self.drive_raw_callback,
            10
        )

        # Publishers -------------
        self.publisher = self.create_publisher(
            AckermannDriveStamped,
            '/drive', # topic
            10 
        ) 




    def scan_callback(self, msg: LaserScan):
        """
        Processes incoming LaserScan messages.
        """
        self.latest_scan = msg
        self.check_for_collision() 


    def odom_callback(self, msg: Odometry):
        """
        Processes incoming Odometry messages.
        """
        self.latest_odom = msg
        self.check_for_collision() 

    
    def drive_raw_callback(self, msg: AckermannDriveStamped):
        """
        Processes incoming driving control messages.
        """
        self.latest_drive_raw = msg
        self.check_for_collision() 




    def check_for_collision(self):
 
        # Ensure both messages have been received at least once
        if self.latest_scan is None:
            return
        
        if self.latest_odom is None:
            self.get_logger().info('no ODOM')
            self.linear_x_speed = 1
        else:
            self.linear_x_speed = self.latest_odom.twist.twist.linear.x
            self.get_logger().info(f'getting a Real ODOM VALUE: {self.linear_x_speed}')
  

        # Collision logic here
        time_to_collision = self.calculate_time_to_collision(self.latest_odom, self.latest_scan)
        
        # Check for incoming collisions (TODO: define threshold)
        if time_to_collision < 0.4: # make 0.4
            self.publish_stop()
            self.get_logger().warn("Crashing...")
        elif time_to_collision < 0.5:
            self.get_logger().warn("Obstacle encountered, STOPPING...")
            self.publish_brake()
        elif self.latest_drive_raw is None:
            self.get_logger().info("Youre fine")
        else:
            self.publisher.publish(self.latest_drive_raw)


    

    def calculate_time_to_collision(self, odom_msg:Odometry, scan_msg: LaserScan):
        
        range_min_tts = float('inf')
        car_radius = 0.30

        window = 620 - 460

        if scan_msg.ranges: # check that we have ranges data
            for i in range(window):
                curr_range = scan_msg.ranges[i+460] - car_radius

                if curr_range > scan_msg.range_min and curr_range < scan_msg.range_max:

                    # calculate relevant parametes for calculating TTS
                    beam_angle = (-540 + i) * 4 # get the beam angle in degrees from vehicles x axis
                    linear_speed = self.linear_x_speed
                    range_rate = linear_speed * math.cos(beam_angle * 2 * math.pi / 360)
                    
                    # Ignore range rates of <= 0
                    if range_rate <= 0:
                        time_to_collision = float('inf')
                    else:
                        time_to_collision = curr_range / range_rate

                    # return only the minimum TTS for the range
                    if time_to_collision < range_min_tts:
                        range_min_tts = time_to_collision
                    

        self.get_logger().info(f'Time to collision is: {range_min_tts}')
        return range_min_tts 
            

        

    def publish_stop(self):
        rclpy.shutdown()
    
    def publish_brake(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.get_logger().info("Stopping with Ackermannn")
        self.publisher.publish(msg)

        


def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode() 
    rclpy.spin(safety_node)
    safety_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
