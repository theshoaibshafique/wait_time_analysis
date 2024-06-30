import supervision as sv
from inference import get_model
import cv2
from utils.general import load_zones_config,save_video
from utils.timers import FPSBasedTimer
import numpy as np
import time



COLORS = sv.ColorPalette.from_hex(["#E6194B","#3CB44B","#FFE119","#3C76D1"])
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

BBOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

def main(
        source_video_path='data/checkout/input_ss.mp4',
        zone_configuration_path='data/checkout/config.json',
        modelId='coco/24'):

    print("HELLLO")
    model = get_model(model_id=modelId)
    tracker = sv.ByteTrack()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    frame = next(frame_generator)
    resolution_wh = frame.shape[1],frame.shape[0] 
    # length = sum(1 for _ in frame_generator)  # Count elements without storing them
    # print("Total Frames: ",length)

    polygons = load_zones_config(file_path=zone_configuration_path)

    zones = [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=resolution_wh,
            triggering_position=(sv.Position.CENTER)
        )
        for polygon in polygons
    ]

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    timers = [FPSBasedTimer(fps=video_info.fps) for _ in zones]
    output_video_frames = []

    for frame in frame_generator:
        start_time = time.time()
        results = model.infer(frame,confidence=CONFIDENCE_THRESHOLD,iou_threshold=IOU_THRESHOLD)[0]

        detections = sv.Detections.from_inference(results)
        detections = detections[detections.class_id==0]
        detections = tracker.update_with_detections(detections=detections)

        annotated_frame = frame.copy()

        for idx,zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone.polygon,
                color=COLORS.by_idx(idx),
                thickness=2
            )
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full_like(detections_in_zone.class_id,idx)

            annotated_frame = BBOX_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup
            )

            labels = [
                f"#{tracker_id} {time:.1f}s"
                for tracker_id,time in zip(detections_in_zone.tracker_id,time_in_zone)
            ]

            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels
            )
         
        end_time = time.time()
        output_video_frames.append(annotated_frame)
        print(end_time-start_time)
       

    
    save_video(output_video_frames,output_video_path="data/checkout/output.mp4")     
       

    #     cv2.imshow("Results", annotated_frame)
    #     if cv2.waitKey(1) & 0xFF ==ord('q'):
    #         break

    # cv2.destroyAllWindows()






if __name__ == "__main__":
    main()