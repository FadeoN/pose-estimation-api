import requests

from application.command.model.frame_pose_dto import VideoKeypointDTO, KeypointDTO, FrameKeypointDTO
from application.command.predict_video_pose_command import PredictVideoPoseCommand

import tensorflow as tf
import cv2
from application import posenet

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

def download_file(url):
    r = requests.get(url, stream=True)
    with open("video.mp4", 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

async def handle(command: PredictVideoPoseCommand):
    current_frame_count = 0

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(101, sess)
        output_stride = model_cfg['output_stride']

        download_file(command.url)

        cap = cv2.VideoCapture("video.mp4")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(3)
        height = cap.get(4)
        frames = []

        while current_frame_count < frame_count:

            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=0.7125, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0)

            keypoint_scores = keypoint_scores.reshape(-1)
            keypoint_coords *= output_scale
            keypoint_coords = keypoint_coords.reshape(-1, 2)

            keypoints = []

            for idx, (part_name, coords, score) in enumerate(zip(PART_NAMES, keypoint_coords, keypoint_scores)):

                normalized_x = coords[1] / width
                normalized_y = coords[0] / height
                keypoints.append(KeypointDTO(
                    name=part_name,
                    x=normalized_x,
                    y=normalized_y,
                    score=score))

            frames.append(FrameKeypointDTO(keypoints=keypoints))

            current_frame_count += 1

    return VideoKeypointDTO(width=int(width),
                            height=int(height),
                            frames=frames)