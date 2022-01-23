import threading
from queue import Queue
from dronekit import *
from pymavlink import mavutil

print_lock = threading.Lock()
MAV_MODE_AUTO = 4
class Drone(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.queue = queue
        self.daemon = True

        print("Connecting")
        self.vehicle = connect("127.0.0.1:14540", wait_ready=True)
        self.home_position_set = False

        @self.vehicle.on_message('HOME_POSITION')
        def listener(elf, name, home_position):
            self.home_position_set = True

        while not self.home_position_set:
            print("Waiting for home position ...")
            time.sleep(1)

        print(" Type: %s" % self.vehicle._vehicle_type)
        print(" Armed: %s" % self.vehicle.armed)
        print(" System status: %s" % self.vehicle.system_status.state)
        print(" GPS: %s" % self.vehicle.gps_0)
        print(" Alt: %s" % self.vehicle.location.global_relative_frame.alt)

        self.PX4setMode(MAV_MODE_AUTO)
        time.sleep(1)

        cmds = self.vehicle.commands
        cmds.clear()

        self.home = self.vehicle.location.global_relative_frame
        self.vehicle.home_location = self.vehicle.location.global_frame

        # Takeoff to 10 meters
        wp = self.get_location_offset_meters(self.home, 0, 0, 10)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        # Move 10 meters North
        wp = self.get_location_offset_meters(wp, 10, 0, 0)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        # Move 10 meters East
        wp = self.get_location_offset_meters(wp, 0, 10, 0)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        # Move 10 meters South
        wp = self.get_location_offset_meters(wp, -10, 0, 0)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        # Move 10 meters West
        wp = self.get_location_offset_meters(wp, 0, -10, 0)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        # Land
        wp = self.get_location_offset_meters(self.vehicle.home_location, 0, 0, 0)
        cmd = Command(0, 0, 0,
                      mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_LAND,
                      0, 1, 0, 0, 0, 0,
                      wp.lat, wp.lon, wp.alt)
        cmds.add(cmd)

        cmds.upload()
        time.sleep(2)

    def run(self):
        self.vehicle.armed = True

        nextwaypoint = self.vehicle.commands.next
        while nextwaypoint < len(self.vehicle.commands):
            command = self.queue.get()
            print("Received command {}".format(command))

            if self.vehicle.commands.next > nextwaypoint:
                display_seq = self.vehicle.commands.next + 1
                print("Moving to waypoint %s" % display_seq)
                nextwaypoint = self.vehicle.commands.next

            time.sleep(1)

        while self.vehicle.commands.next > 0:
            time.sleep(1)

        self.vehicle.armed = False
        time.sleep(1)

        self.vehicle.close()
        time.sleep(1)

    def add_command(self):
        pass

    def PX4setMode(self, mavMode):
        self.vehicle._master.mav.command_long_send(self.vehicle._master.target_system,
                                                   self.vehicle._master.target_component,
                                                   mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                                   mavMode,
                                                   0, 0, 0, 0, 0, 0)

    def get_location_offset_meters(self, original_location, dNorth, dEast, alt):
        earth_radius = 6378137.0
        dLat = dNorth / earth_radius
        dLon = dEast / (earth_radius * math.cos(math.pi * original_location.lat / 180))

        newlat = original_location.lat + (dLat * 180 / math.pi)
        newlon = original_location.lon + (dLon * 180 / math.pi)
        return LocationGlobal(newlat, newlon, original_location.alt + alt)


if __name__ == '__main__':
    q = Queue()
    drone = Drone(q)
    drone.start()