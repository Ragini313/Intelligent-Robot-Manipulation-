#!/usr/bin/env python3

from robot_commander import RobotCommander
import rospy


def main():
    robot_commander = RobotCommander()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass