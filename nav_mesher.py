import pyzed.sl as sl
import time
import ogl_viewer.viewer as gl
import numpy as np


def main():
    height_threshold = 0.3
    normal_threshold = 0.1

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode (default fps: 60)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    camera_infos = zed.get_camera_information()

    # Enable positional tracking with default parameters
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable spatial mapping with default parameters
    spatial_mapping_parameters = sl.SpatialMappingParameters()
    err = zed.enable_spatial_mapping(spatial_mapping_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create a Mesh object
    mesh = sl.Mesh()

    # Create a Pose object
    last_time = time.time()
    pose = sl.Pose()
    plane = sl.Plane()
    image = sl.Mat()
    normals = sl.Mat()
    xyz = sl.Mat()

    # Create a PyOpenGl viewer
    has_imu = camera_infos.sensors_configuration.gyroscope_parameters.is_available
    viewer = gl.GLViewer()
    viewer.init(camera_infos.camera_configuration.calibration_parameters.left_cam, has_imu)
    user_action = gl.UserAction()
    user_action.clear()

    zed.enable_positional_tracking()

    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD
    runtime_parameters.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    floor_normal = floor_point = None
    while viewer.is_available():
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            # grab image, normals, and xyz data
            zed.retrieve_image(image, sl.VIEW.LEFT)
            zed.retrieve_measure(normals, sl.MEASURE.NORMALS)
            zed.retrieve_measure(xyz, sl.MEASURE.XYZ)
            
            # Get the pose of the left eye of the camera with reference to the world frame
            track_state = zed.get_position(pose)

            # check for user input
            if user_action.increase_height:
                height_threshold += 0.05 if height_threshold < 5 else 0

            if user_action.decrease_height:
                height_threshold -= 0.05 if height_threshold > 0.05 else 0

            if user_action.increase_normal:
                normal_threshold += 0.05 if normal_threshold < 1 else 0

            if user_action.decrease_normal:
                normal_threshold -= 0.05 if normal_threshold > 0.05 else 0

            if track_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Update the viewer

                # get the current time
                duration = time.time() - last_time

                # relocate floor plane
                if duration > 1:
                    last_time = time.time()
                    reset_tracking_floor_frame = sl.Transform()
                    find_plane_success = zed.find_floor_plane(plane, reset_tracking_floor_frame)

                # update floor plane if found
                if find_plane_success == sl.ERROR_CODE.SUCCESS:
                    floor_normal = plane.get_normal()
                    floor_point = plane.get_center()

                # once floor is found, compute how many pixels are close to the floor
                if floor_normal is not None:
                    # find the normals of pixels that are close to [0,1,0]
                    normal_mask = np.isclose(normals.get_data()[:,:,1], floor_normal[1], atol=normal_threshold)
                    # find the heights of pixels that are close to the floor point
                    height_mask = np.isclose(xyz.get_data()[:,:,1], floor_point[1], atol=height_threshold)
                    # combine the two masks to get a mask of pixels that are close to the floor
                    floor_mask = np.logical_and(normal_mask, height_mask)

                    # get the image data
                    image_val = image.get_data(deep_copy=False)

                    # add a green tint to all the pixels in floor_mask
                    gr_inc = 75
                    image_val[floor_mask] = np.where(
                        np.add(image_val[floor_mask], np.array([0,gr_inc,0,0])) > 255, 
                        255, 
                        np.add(image_val[floor_mask], np.array([0,gr_inc,0,0]))
                    )

            # update the viewer and get user input
            user_action = viewer.update_view(image, pose.pose_data(), track_state, height_threshold, normal_threshold)

    viewer.exit()
    image.free(sl.MEM.CPU)
    mesh.clear()

    # Disable modules and close the camera
    zed.disable_positional_tracking()
    zed.close()

if __name__ == "__main__":
    main()
