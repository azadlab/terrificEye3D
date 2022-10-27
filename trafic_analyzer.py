import numpy as np
import math
from random import randint
import statistics
from collections import OrderedDict
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import sys
from centroidtracker import CentroidTracker
import pyzed.sl as sl
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parser():
    parser = argparse.ArgumentParser(description="Traffic Analyzer 3D")
    parser.add_argument("--input", type=str, default=0,
                        help="SVO source. If empty, uses zedcam 0 stream")
    parser.add_argument("--out_filename", type=str, default=None,
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--start_frame", default=0 , help="starting position of .svo file.")
    parser.add_argument("--end_frame", default=1000000 , help="end position of .svo file.")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--confidence", type=float, default=.65,
                        help="remove detections with confidence below this value")

    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 30
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def project_pt(image, point):
    
    image_h, image_w, __ = image.shape
    x = int((point[0]/darknet_width) * image_w)
    y = int((point[1]/darknet_height) * image_h)

    return x,y



def generate_analytics(frame, tobjects):
    global inbound_count
    global outbound_count
    global IDs_counted
    global class_counts
    global inbound_counted
    global outbound_counted

    analytics_text = []

    for tid, tobj in tobjects.items():
        
        cur_pos = project_pt(frame, tobj.cur_position)
        starting_pos = project_pt(frame, tobj.starting_position)

        #if(tid not in IDs_counted):
        #    class_counts[tobj.label]+=1
        #    IDs_counted.add(tid)
        
        if(starting_pos[1] < Boundary_Line[0][1]) :  #if the object was detected above boundary line (INBOUND)
            if((starting_pos[0]>Boundary_Line[0][0] and starting_pos[0]<Boundary_Line[1][0])):  #Check if it is on the bridge
                if(cur_pos[1] > Boundary_Line[0][1] ): # if object has crossed the boundary line
                    if(tobj.ID not in inbound_counted):
                        inbound_count+=1
                        inbound_counted.add(tobj.ID)
                        if(tobj.ID in outbound_counted):
                            outbound_counted.remove(tobj.ID)
                        class_counts[tobj.label]+=1
                        
                        cur_pos = list(cur_pos)
                        cur_pos[0] = int((Boundary_Line[0][0]+Boundary_Line[1][0])/2)
                        cur_pos[1] = min(cur_pos[1] + 20,video_height)
                        tobj.starting_pos = tuple(cur_pos)
                        #IDs_counted.add(tid)
        else:   #if the object was detected below boundary (OUTBOUND)
            if(cur_pos[1] < Boundary_Line[0][1] and (cur_pos[0]>Boundary_Line[0][0] and cur_pos[0]<Boundary_Line[1][0]) ): #if the object has moved above the boundary line & is on the bridge
                if(tobj.ID not in outbound_counted):
                    outbound_count+=1
                    outbound_counted.add(tobj.ID)
                    if(tobj.ID in inbound_counted):
                        inbound_counted.remove(tobj.ID)
                    class_counts[tobj.label]+=1

                    cur_pos = list(cur_pos)
                    cur_pos[0] = int((Boundary_Line[0][0]+Boundary_Line[1][0])/2)
                    cur_pos[1] = max(0,cur_pos[1] - 20)
                    tobj.starting_pos = tuple(cur_pos)
                    #IDs_counted.add(tid)

        

    analytics_text.append("Inbound: {}".format(inbound_count) )
    analytics_text.append("Outbound: {}".format(outbound_count))

    
    
    for i in range(len(OOI)-1):    # gather counts for each class type
        clabel = OOI[i]
        count = class_counts[clabel]
        if(clabel=="bus"):  #aggregating the heavy vehicles
            count = class_counts["bus"]+class_counts["truck"]
            clabel = "bus/truck"
            
        analytics_text.append("{}:{} ".format(clabel,count) )
        
    return analytics_text


def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median



def video_capture(frame_queue, depth_queue, tracking_img_queue):
    global cap
    fno = int(args.start_frame)
    while 1:
        
        cap = cam.grab(runtime)
        if cap == sl.ERROR_CODE.SUCCESS:
            
            log.info("Captured Frame:{}".format(fno))
            cam.retrieve_image(mat, sl.VIEW.RIGHT)
            image = mat.get_data()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            

            cam.retrieve_measure(point_cloud_mat, sl.MEASURE.XYZRGBA)
            depth = point_cloud_mat.get_data()


            frame_resized = cv2.resize(image, (darknet_width, darknet_height),
                                    interpolation=cv2.INTER_LINEAR)

            tracking_img_queue.put(frame_resized)
            frame_queue.put(image)
            depth_queue.put(depth)

        if(fno > int(args.end_frame)):
           cap = sl.ERROR_CODE.END_OF_SVOFILE_REACHED

        if cap == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            log.info("File Ended")        
            break
        fno += 1

    log.info("Capture Ended")
    cam.close()
    os._exit(1)



def detect_and_track(tracking_img_queue, detection_queue , tracker_queue ,fps_queue):
    
    ct = CentroidTracker()

    while cap!=sl.ERROR_CODE.END_OF_SVOFILE_REACHED:

        tracking_image = tracking_img_queue.get()
        
        prev_time = time.time()
        

        darknet_image = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(darknet_image, tracking_image.tobytes())
        
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)

        darknet.print_detections(detections, args.ext_output)
        
        bboxes = []
        rects = []
        confidences = []
        detections_filtered = []
        labels = []
        # Create a new tracker for each detection
        for label, confidence, bbox in detections:

            if(float(confidence) > args.confidence and (label in OOI) ): #If confidence of detection is high and the object is of interest save that object
                if  bbox[0]+bbox[2] < int(darknet_width - darknet_width/4) and bbox[0]+bbox[2] > int(darknet_width/4) and bbox[1]+bbox[3] > int(darknet_height/5):   #Ignore edges of the image
                    (centerX, centerY, width, height) = bbox
                    bboxes.append(bbox)
                    rect = [int(centerX) - int(width/2), int(centerY) - int(height/2), int(centerX) + int(width/2), int(centerY) + int(height/2)]
                    rects.append( rect )
                    confidences.append(float(confidence))
                    labels.append(label)
                #detections_filtered.append((label,confidence,bbox))
        #detection_queue.put(detections_filtered)

        rects_filtered = []
        labels_filtered = []
        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(bboxes, confidences, args.confidence,args.thresh)
        if len(idxs) > 0:
            for i in idxs.flatten():
                rects_filtered.append(rects[i])
                labels_filtered.append(labels[i])
                detections_filtered.append((labels[i],confidences[i],bboxes[i]))

        detection_queue.put(detections_filtered)

        Objects  = ct.update(labels_filtered,rects_filtered)
        tracker_queue.put(Objects)

        darknet.free_image(darknet_image)

        total_time = (time.time() - prev_time)
        fps = int(1/(total_time))
        fps_queue.put(fps)
    log.info("Terminating Detections")
    cam.close()
    os._exit(1)


def generate_color(meta_path):
    '''
    Generate random colors for the number of classes mentioned in data file.
    Arguments:
    meta_path: Path to .data file.

    Return:
    color_array: RGB color codes for each class.
    '''
    random.seed(42)
    with open(meta_path, 'r') as f:
        content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array



def drawing(frame_queue, depth_queue , detection_queue ,tracker_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    if(args.out_filename is not None):
        video = set_saved_video(cap, args.out_filename,
                                (video_width, video_height))

    while cap != sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        frame = frame_queue.get()
        depth = depth_queue.get()

        fps = fps_queue.get()
        print("FPS: {}".format(fps))

        detections_adjusted = []
        if frame is not None:
            # print("Drawing {} object rectangles".format(len(tracker_output_queues)))
            objects = tracker_queue.get()
            detections = detection_queue.get()
            
            
            for tid, tobj in objects.items():
                 
                text = "ID {}".format(tid)
                centroid = tobj.cur_position
                cx,cy = project_pt(frame,centroid)
                
                cv2.putText(frame, text, (cx - 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            for label,confidence,bbox in detections:
                
                bbox_adjusted = convert2original(frame, bbox)
                x, y, z = get_object_depth(depth, bbox_adjusted)
                distance = math.sqrt(x * x + y * y + z * z)
                distance = "{:.2f}".format(distance)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))

                y_extent = int(bbox_adjusted[3])
                x_extent = int(bbox_adjusted[2])
                # Coordinates are around the center
                x_coord = int(bbox_adjusted[0] - bbox_adjusted[2]/2)
                y_coord = int(bbox_adjusted[1] - bbox_adjusted[3]/2)

                box_color = color_array[class_names.index(label)]
                if(label=="bus" or label=="truck"):           #merging the display
                    label = "bus/truck"
                    box_color = color_array[class_names.index("truck")]
                

                cv2.rectangle(frame, (x_coord - 1, y_coord - 1),
                              (x_coord + x_extent + 1, y_coord + (18 + 4)),
                              box_color, -1)
                cv2.putText(frame, label + " " +  (str(distance) + " m"),
                            (x_coord + (1 * 4), y_coord + (10 + 1 * 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (x_coord - 1, y_coord - 1),
                              (x_coord + x_extent + 1, y_coord + y_extent + 1),
                              box_color, int(1*2))
            #frame = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            
            #drawing line for counter
            frame = cv2.line(frame, (Boundary_Line[0][0],Boundary_Line[0][1]), (Boundary_Line[1][0],Boundary_Line[1][1]), (255,255,0), 5)
            #Adding analytics to the frame
            analytics_text = generate_analytics(frame, objects)
            
            for i in range(len(analytics_text)):
                
                if i > 1:
                    label = OOI[i-2]
                    box_color = color_array[class_names.index(label)]
                    if(label == "bus" or label == "truck"):
                       box_color = color_array[class_names.index("truck")]
                    cv2.rectangle(frame, (0, i*25),(120, i*25+20),box_color, -1)
                else:
                    cv2.rectangle(frame, (0, i*25),(120, i*25+20),(255,255,255), -1)

                cv2.putText(frame, analytics_text[i], (10, 15+i*25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            
            # Displaying the Company Text
            cv2.rectangle(frame, (600, 580),(770, 630),(0,0,255), -1)
            cv2.putText(frame, "  TerrificEye3D  ", (610, 600),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, "AZAD Research Lab", (610, 620),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('TraficAnalyzer3D', frame)
            if args.out_filename is not None:
                video.write(frame)
            if cv2.waitKey(20) == 27:
                os._exit(1)
                # break
    log.info("Stopping Drawings")
    cam.close()
    if(args.out_filename is not None):
        video.release()
    cv2.destroyAllWindows()
    os._exit(1)


if __name__ == '__main__':

    # objects of interest
    OOI = ['person','bicycle','car','bus','truck']
    class_counts = {'person':0,'bicycle':0,'car':0,'bus':0,'truck':0}
    Boundary_Line = ((350,230),(650,210))
    inbound_count = 0
    outbound_count = 0
    IDs_counted = set()
    inbound_counted = set()
    outbound_counted = set()

    frame_queue = Queue(maxsize=1)
    depth_queue = Queue(maxsize=1)
    detection_queue = Queue(maxsize=1)
    tracker_queue = Queue(maxsize=1)
    tracking_img_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )


    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    input_path = str2int(args.input)
    #cap = cv2.VideoCapture(input_path)

    input_type = sl.InputType()
    if input_type is not None:
        log.info("SVO file : " + input_path)
        input_type.set_from_svo_file(input_path)
    else:
        # Launch camera by id
        input_type.set_from_camera_id(0)

    init = sl.InitParameters(input_t=input_type)
    init.coordinate_units = sl.UNIT.METER

    cam = sl.Camera()
    if not cam.is_opened():
        log.info("Opening ZED Camera...")
    status = cam.open(init)
    
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()
    
    runtime = sl.RuntimeParameters()
    # Use STANDARD sensing mode
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    mat = sl.Mat()
    point_cloud_mat = sl.Mat()

    cam.set_svo_position(int(args.start_frame))
    
    cap = sl.ERROR_CODE.SUCCESS
    #cap = cam.grab(runtime)
    
    color_array = generate_color(args.data_file)

    video_width = cam.get_camera_information().camera_resolution.width
    video_height = cam.get_camera_information().camera_resolution.height
    Thread(target=video_capture, args=(
        frame_queue,depth_queue, tracking_img_queue)).start()
    Thread(target=detect_and_track, args=(
        tracking_img_queue, detection_queue, tracker_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, depth_queue , detection_queue,
                                 tracker_queue, fps_queue,)).start()
