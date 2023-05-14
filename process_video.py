"""
Copyright 2023 Noah Mühl

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
from deepface import DeepFace
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Emotion to emoji mapping
emotion_to_emoji = {
    "neutral": "emojis/neutral.png",
    "happy": "emojis/happy.png",
    "surprise": "emojis/surprise.png",
    "disgust": "emojis/disgust.png",
    "fear": "emojis/fear.png",
    "sad": "emojis/sad.png",
    "angry": "emojis/angry.png"
}


def replace_face_with_emoji(frame, face_rect, emotion, emoji_scaler):

    # Load the emoji image
    emoji = cv2.imread(f'emojis/{emotion}.png', -1)

    # Resize the emoji to the size of the face region
    emoji = cv2.resize(emoji, (int(face_rect['w']*emoji_scaler), int(face_rect['h']*emoji_scaler)))

    # Adjust face_rect for the larger emoji
    diff_w = int((face_rect['w']*emoji_scaler - face_rect['w'])/2)
    diff_h = int((face_rect['h']*emoji_scaler - face_rect['h'])/2)
    face_rect['x'] -= diff_w
    face_rect['y'] -= diff_h
    face_rect['w'] += diff_w*2
    face_rect['h'] += diff_h*2

    # Make sure the face_rect does not go outside the frame boundaries
    face_rect['x'] = max(0, face_rect['x'])
    face_rect['y'] = max(0, face_rect['y'])
    face_rect['w'] = min(frame.shape[1] - face_rect['x'], face_rect['w'])
    face_rect['h'] = min(frame.shape[0] - face_rect['y'], face_rect['h'])

    # Make sure the emoji and the roi have the same size
    emoji = cv2.resize(emoji, (face_rect['w'], face_rect['h']))

    # Extract the alpha channel from the emoji
    alpha = emoji[:, :, 3] / 255.0

    # Remove the alpha channel from the emoji
    emoji = emoji[:, :, :3]

    # Extract the region from the frame that will be replaced with the emoji
    roi = frame[face_rect['y']:face_rect['y']+face_rect['h'], face_rect['x']:face_rect['x']+face_rect['w']]

    # Blend the emoji with the ROI
    blended = cv2.convertScaleAbs(roi*(1-alpha)[:, :, None] + emoji*alpha[:, :, None])

    # Replace the ROI on the frame with the blended image
    frame[face_rect['y']:face_rect['y']+face_rect['h'], face_rect['x']:face_rect['x']+face_rect['w']] = blended


def process_frame(frame, emoji_scaler):
    # Write frame to temp file
    cv2.imwrite('temp_frame.jpg', frame)

    # Detect faces
    faces = DeepFace.extract_faces(img_path='temp_frame.jpg', detector_backend='retinaface', enforce_detection=False)
    if faces:  # Only proceed if faces were detected
        i = 0
        for face in faces:
            if face['confidence'] != 0:
                # Analyze the face
                emotion = DeepFace.analyze(img_path='temp_frame.jpg', actions=['emotion'], detector_backend="retinaface", enforce_detection=False)
                dominant_emotion = emotion[i]["dominant_emotion"]

                # Replace the face with an emoji
                replace_face_with_emoji(frame, face['facial_area'], dominant_emotion, emoji_scaler)
                i += 1

    return frame


def main(input_video, output_video, emoji_scaler):
    # Video reading and writing setup
    cap = cv2.VideoCapture(input_video)

    # Get the dimensions of the video frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, emoji_scaler)

        out.write(frame)

        # Preview the video
        cv2.imshow('Video Preview', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press q to stop and save to output file
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Setup argument parser
parser = argparse.ArgumentParser(description='Process video.')
parser.add_argument('input_video', type=str, help='Input video file')
parser.add_argument('--output_video', type=str, help='Output video file')
parser.add_argument('--emoji_scaler', type=float, default=1.5, help='Emoji scaler')

args = parser.parse_args()

# Check if input video file exists
if not os.path.isfile(args.input_video):
    raise FileNotFoundError(f"Input video file '{args.input_video}' not found.")

# If output video file is not provided, generate it from input video file
if args.output_video is None:
    filename, file_extension = os.path.splitext(args.input_video)
    args.output_video = f"{filename}_output{file_extension}"

main(args.input_video, args.output_video, args.emoji_scaler)
