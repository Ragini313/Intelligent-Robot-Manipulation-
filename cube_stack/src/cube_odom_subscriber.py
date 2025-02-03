#!/usr/bin/python3

import rospy
from nav_msgs.msg import Odometry
from cube_msgs.msg import Cube  



# Callback function to handle odometry messages
def odom_callback(msg, cube_name):
    """Callback function to handle odometry messages."""
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation

    # 创建 Cube 消息并填充数据
    cube_message = Cube()
    cube_message.position = position
    cube_message.orientation = orientation
    cube_message.size = 0.045
    cube_message.name = cube_name  # 添加物体名称

    # 发布 Cube 消息
    #rospy.loginfo(f"Publishing {cube_name} position: {cube_message}")
    cube_publisher.publish(cube_message)



def main():
    global cube_publisher
    rospy.init_node('cube_odom_subscriber')

    cube_publisher = rospy.Publisher('/cube', Cube, queue_size=10)

    # Get a list of all 28 cube topics
    topics = [
        "/cube_0_odom", "/cube_1_odom", "/cube_2_odom", "/cube_3_odom", "/cube_4_odom",
        "/cube_5_odom", "/cube_6_odom", "/cube_7_odom", "/cube_8_odom", "/cube_9_odom",
        "/cube_10_odom", "/cube_11_odom", "/cube_12_odom", "/cube_13_odom", "/cube_14_odom",
        "/cube_15_odom", "/cube_16_odom", "/cube_17_odom", "/cube_18_odom", "/cube_19_odom",
        "/cube_20_odom", "/cube_21_odom", "/cube_22_odom", "/cube_23_odom", "/cube_24_odom",
        "/cube_25_odom", "/cube_26_odom", "/cube_27_odom"
    ]

    # Create subscribers for each topic
    for topic in topics:
        rospy.Subscriber(topic, Odometry, odom_callback, callback_args=topic)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Shutting down cube_odom_subscriber node.")


