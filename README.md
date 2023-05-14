# EmojiPrivacy
A simple python script to overlay faces in a video with emojis for privacy. Additionally the facial expression gets analysed so that the expression gets projected onto one of seven emojis (angry, disgust, fear, happy, neutral, sad and surprise)

## Requirements
`pip install tensorflow deepface`

## Usage
```python process_video.py <input_video> [--output_video] [--emoji_scaler]```

- `input_video`: Path to the input video file. This is a required argument.
- `output_video`: Path to the output video file. This is an optional argument. If not provided, the output filename will be the same as the input filename, appended with _output.
- `emoji_scaler`: A floating-point value to scale the size of the emojis. This is an optional argument. If not provided, it defaults to 1.5.

### Example Usages
```
python process_video.py input_video.mp4
python process_video.py input_video.mp4 --output_video output_video.mp4 --emoji_scaler 2.0
```

## Output Test
[![Watch the video](https://img.youtube.com/vi/_OjslipZQ0s/maxresdefault.jpg)](https://youtu.be/_OjslipZQ0s)
Original Video: https://www.youtube.com/watch?v=ydAyvvDQrgY

## Limitations
Currently no audio of the input video gets copied and there is no possibility to exclude specific faces. I am currently working on the "exclude face feature".

# License

This project is licensed under the terms of the Apache License 2.0.

For details, see the [LICENSE](./LICENSE) file.

