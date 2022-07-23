import math
import sys
import numpy as np
import pyzed.sl as sl
import cv2
import imutils
  
# left in if wanting to test/save images
def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    
def get_depth_info(cx, cy, point_cloud, tr_np):
    # Get and print distance value in mm at the center of the image
    # We measure the distance camera - object using Euclidean distance
    x = cx
    y = cy
    err, point_cloud_value = point_cloud.get_value(x, y)

    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                        point_cloud_value[1] * point_cloud_value[1] +
                        point_cloud_value[2] * point_cloud_value[2])

    point_cloud_np = point_cloud.get_data()
    point_cloud_np.dot(tr_np)

    if not np.isnan(distance) and not np.isinf(distance):
        print("Orange object x y coords ({}, {}): Distance {:1.3} m".format(x, y, distance), end="\r")
        # return [x, y, distance]
    else:
        print("Can't estimate distance at this position.")
        print("Your camera is probably too close to the scene, please move it backwards.\n")


def draw_shapes_on_screen(image_ocv, cx, cy, depth_image_ocv, centers):
    cv2.circle(image_ocv, (cx,cy), 3, (255, 255, 255), -1)
    cv2.circle(depth_image_ocv, (cx,cy), 3, (255, 255, 255), -1)

    # adding center point to the contour  
    cv2.putText(image_ocv,"Centre", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255), 1)
    cv2.putText(image_ocv,"("+str(cx)+","+str(cy)+")", (cx+10,cy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 255),1)
    # adding contour to the centers list, to create a list of all contour objects
    centers.append([cx, cy])
    # return centers


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)


    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

    key = ' '
    # while key != q
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv (ocv), use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            depth_image_ocv = depth_image_zed.get_data()


            # applying a gaussian blur to the cam so that it does not pick up background noise
            blurred_frame = cv2.GaussianBlur(image_ocv, (5, 5), 0)
            hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
            
            # defining the upper and lower bounds for color
            lower_orange = np.array([10, 156, 73])

            upper_orange = np.array([73, 255, 255]) # this value has been tested with our underwater footage and should be good for up to 8 about feet away
            # upper_orange = np.array([31, 255, 255])


            # creating a orange hsv_frame mask so that only the specified range shows up
            orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)
            orange_frame = cv2.bitwise_and(image_ocv, image_ocv, mask=orange_mask)

            # defining contours and setting imutils.grab_contours to get the already created contours
            contours = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = imutils.grab_contours(contours)
            centers = []
                
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # only display for orange things that are a specifed size to decrease false positives. this number may need to be + or - depending on how our sub works
                if area > 600:
                    cv2.drawContours(image_ocv, [contour], -1, (0,255, 0), 3)
                    cv2.drawContours(depth_image_ocv, [contour], -1, (0,255, 0), 3)

                    # movements is used to plot points 
                    M = cv2.moments(contour)
                    
                    #coordinate x and coordinate y 
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # creating the center point circle and displaying it on screen
                    # you could comment out draw_shapes_on_screen() and it would still work
                    draw_shapes_on_screen(image_ocv, cx, cy, depth_image_ocv, centers)
                    # print(len(centers))
                
                    # [x, y, distance] distance is in meters from the camera. 
                    x_y_coordinate_and_depth_array = get_depth_info(cx, cy, point_cloud, tr_np)
                    # print(x_y_coordinate_and_depth_array)

                    
                    
            # display what the camera sees, a mask layer with only orange, and the depth view
            # can likely be taken out 
            cv2.imshow("Image", image_ocv)
            cv2.imshow("Mask", orange_frame)
            cv2.imshow("Depth", depth_image_ocv)

            key = cv2.waitKey(10)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()