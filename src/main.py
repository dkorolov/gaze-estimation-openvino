import sys
import time
import cv2
import yaml
import logging

from argparse import ArgumentParser
from input_feeder import InputFeeder
from mouse_controller import MouseController
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation


def get_arg():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True,
                        help="Path to Preferences YAML file")   
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Input type: 'video' for video file or 'cam' for camera")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify device for inference:"
                             "CPU, GPU, FPGA or MYRIAD")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)") 
    parser.add_argument("-o","--overlay",default=False,
                        help="Overlay models output on video",action="store_true")
    parser.add_argument("-m","--mouse_move",default=True,
                        help="Move mouse based on gaze estimation output",action="store_true")
    return parser.parse_args()


def main():
    """
    Load inference networks, stream video to network,
    and output stats and video.
    :return: None
    """

    # Logger init
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # Get command line args
    args = get_arg()

    #Load Preferencies
    with open(args.config_file, "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    models = cfg['models']
    input_source = args.input
    video_path = cfg['video_path']
    face_model = FaceDetection(models['face_detection'])
    head_pose_model = HeadPoseEstimation(models['head_pose_estimation'])
    facial_landmarks_model = FacialLandmarksDetection(models['facial_landmarks_detection'])
    gaze_estimation_model = GazeEstimation(models['gaze_estimation'])

    # Initialise the MouseController
    mouse_contr = MouseController("low","fast")

    # Load the models and log timing
    start_time = time.time()
    face_model.load_model(args.device)
    logging.info("Load Face Detection model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    facial_landmarks_model.load_model(args.device)
    logging.info("Load Facial Landmarks Detection model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    head_pose_model.load_model(args.device)
    logging.info("Load Head Pose Estimation model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    start_time = time.time()
    gaze_estimation_model.load_model(args.device) 
    logging.info("Load Gaze Estimation model: {:.1f}ms".format(1000 * (time.time() - start_time)) )

    # Get and open video or camera capture
    #input_feed = InputFeeder('video', args.input)
    #input_feed.load_data()

    input_feed = InputFeeder(input_type=input_source, input_file=video_path)
    input_feed.load_data()

    if not input_feed.cap.isOpened():
        log.critical('Error opening input, check --video_path parameter')
        sys.exit(1)
    # FPS = input_feed.get_fps()

    # Grab the shape of the input 
    # width = input_feed.get_width()
    # height = input_feed.get_height()

    # init scene variables
    frame_count = 0

    ### Loop until stream is over ###
    facedetect_infer_time = 0
    landmark_infer_time = 0
    headpose_infer_time = 0
    gaze_infer_time = 0
    while True:
        # Read the next frame
        try:
            frame = next(input_feed.next_batch())
        except StopIteration:
            break

        if frame is None:
            break


        key_pressed = cv2.waitKey(60)
        frame_count += 1
        input_height, input_width, _ = frame.shape
        logging.info("frame {count} size {w}, {h}".format(count= frame_count, w = input_width, h =input_height)) 
        
        # face detection
        p_frame = face_model.preprocess_input(frame)
        start_time = time.time()
        fnoutput = face_model.predict(p_frame)
        facedetect_infer_time += time.time() - start_time
        out_frame,fboxes = face_model.preprocess_output(fnoutput,frame,args.overlay, args.prob_threshold)
        
        #for each face
        for fbox in fboxes:

            face = frame[fbox[1]:fbox[3],fbox[0]:fbox[2]]
            p_frame = facial_landmarks_model.preprocess_input(face)
            
            start_time = time.time()
            lmoutput = facial_landmarks_model.predict(p_frame)
            landmark_infer_time += time.time() - start_time
            out_frame,left_eye_point,right_eye_point = facial_landmarks_model.preprocess_output(lmoutput, fbox, out_frame,args.overlay, args.prob_threshold)

            # get head pose estimation
            p_frame  = head_pose_model.preprocess_input(face)
            start_time = time.time()
            hpoutput = head_pose_model.predict(p_frame)
            headpose_infer_time += time.time() - start_time
            out_frame, headpose_angels = head_pose_model.preprocess_output(hpoutput,out_frame, face,fbox,args.overlay, args.prob_threshold)

            # get gaze  estimation
            out_frame, left_eye, right_eye  = gaze_estimation_model.preprocess_input(out_frame,face,left_eye_point,right_eye_point,args.overlay)
            start_time = time.time()
            geoutput = gaze_estimation_model.predict(left_eye, right_eye, headpose_angels)
            gaze_infer_time += time.time() - start_time
            out_frame, gazevector = gaze_estimation_model.preprocess_output(geoutput,out_frame,fbox, left_eye_point,right_eye_point,args.overlay, args.prob_threshold)

            cv2.imshow('im', out_frame)
            
            if(args.mouse_move):
                logging.info("mouse move vector : x ={}, y={}".format(gazevector[0], gazevector[1])) 
                mouse_contr.move(gazevector[0], gazevector[1])
            
            #use only first detected face in the frame
            break
        
        # Break if escape key pressed
        if key_pressed == 27:
            break

    #logging inference times
    if(frame_count>0):
        logging.info("***** Models Inference time *****") 
        logging.info("Face Detection:{:.1f}ms".format(1000* facedetect_infer_time/frame_count))
        logging.info("Facial Landmarks Detection:{:.1f}ms".format(1000* landmark_infer_time/frame_count))
        logging.info("Headpose Estimation:{:.1f}ms".format(1000* headpose_infer_time/frame_count))
        logging.info("Gaze Estimation:{:.1f}ms".format(1000* gaze_infer_time/frame_count))


    # Release the capture and destroy any OpenCV windows
    input_feed.close()
    cv2.destroyAllWindows()

    

if __name__ == '__main__':
    main()
