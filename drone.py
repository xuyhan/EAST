from dronekit import *
from pymavlink import mavutil
import time, sys, argparse, math

connection_string = "127.0.0.1:14540"
MAV_MODE_AUTO = 4

print("Connecting")
vehicle = connect(connection_string, wait_ready=True)


def PX4setMode(mavMode):
    vehicle._master.mav.command_long_send(vehicle._master.target_system,
                                          vehicle._master.target_component,
                                          mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                          mavMode,
                                          0, 0, 0, 0, 0, 0)


def get_location_offset_meters(original_location, dNorth, dEast, alt):
    earth_radius = 6378137.0
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius * math.cos(math.pi * original_location.lat / 180))

    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return LocationGlobal(newlat, newlon, original_location.alt + alt)


home_position_set = False


@vehicle.on_message('HOME_POSITION')
def listener(self, name, home_position):
    global home_position_set
    home_position_set = True


while not home_position_set:
    print("Waiting for home position ...")
    time.sleep(1)

print(" Type: %s" % vehicle._vehicle_type)
print(" Armed: %s" % vehicle.armed)
print(" System status: %s" % vehicle.system_status.state)
print(" GPS: %s" % vehicle.gps_0)
print(" Alt: %s" % vehicle.location.global_relative_frame.alt)

PX4setMode(MAV_MODE_AUTO)
time.sleep(1)

cmds = vehicle.commands
cmds.clear()

home = vehicle.location.global_relative_frame

# Takeoff to 10 meters
wp = get_location_offset_meters(home, 0, 0, 10)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

# Move 10 meters North
wp = get_location_offset_meters(wp, 10, 0, 0)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

# Move 10 meters East
wp = get_location_offset_meters(wp, 0, 10, 0)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

# Move 10 meters South
wp = get_location_offset_meters(wp, -10, 0, 0)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

# Move 10 meters West
wp = get_location_offset_meters(wp, 0, -10, 0)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

# Land
wp = get_location_offset_meters(home, 0, 0, 0)
cmd = Command(0,0,0,
              mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
              mavutil.mavlink.MAV_CMD_NAV_LAND,
              0, 1, 0, 0, 0, 0,
              wp.lat, wp.lon, wp.alt)
cmds.add(cmd)

cmds.upload()
time.sleep(2)

vehicle.armed = True

nextwaypoint = vehicle.commands.next
while nextwaypoint < len(vehicle.commands):
    if vehicle.commands.next > nextwaypoint:
        display_seq = vehicle.commands.next + 1
        print("Moving to waypoint %s" % display_seq)
        nextwaypoint = vehicle.commands.next
    time.sleep(1)

while vehicle.commands.next > 0:
    time.sleep(1)

vehicle.armed = False
time.sleep(1)

vehicle.close()
time.sleep(1)
