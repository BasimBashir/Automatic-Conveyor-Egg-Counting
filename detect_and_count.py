"""Standalone egg detection + counting CLI.

Counting is delegated 100% to ultralytics ``solutions.ObjectCounter`` (detection
+ tracking + line-crossing), the same engine the API uses, with the same
egg-conveyor tuning (ByteTrack + NMS IoU). The counting line is the frame's
center line, oriented by direction:
  * "tb" (top->bottom): horizontal line at y = height // 2; downward = counted.
  * "lr" (left->right): vertical line at x = width // 2;   rightward = counted.
"""
import platform
import cv2
import pathlib
import argparse
import warnings
import time

from ultralytics import solutions

from app.core.detector import detect_frame
from app.core.model_cache import get_model
from app.core.annotator import annotate_boxes, annotate_image_detections
from app.core.counting import build_region, extract_solution_boxes
from app.core.video_processor import TRACKER, NMS_IOU, CONF

warnings.filterwarnings("ignore", category=FutureWarning)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath

MODEL_PATH = r"best.pt"


def detect_and_annotate_image(model_path, image_path, save_path=None):
    """Detect eggs in a single image, annotate it, and return the count."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None, 0

    model = get_model(model_path)
    det_info = detect_frame(model, image, conf=CONF, iou=NMS_IOU)
    annotated, egg_count = annotate_image_detections(image, det_info)

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Annotated image saved to '{save_path}'")

    print(f"Detected {egg_count} egg(s) in '{image_path}'")
    return annotated, egg_count


def detect_and_annotate_video(model_path, video_path, save_path=None, direction="tb"):
    """Detect and count eggs crossing the center line in a video using ObjectCounter."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_stream = total_frames <= 0

    region_pts = build_region(direction, width, height)
    counter = solutions.ObjectCounter(model=model_path, region=region_pts,
                                      tracker=TRACKER, iou=NMS_IOU, conf=CONF,
                                      show=False, verbose=False)
    line_thickness = max(2, int(getattr(counter, "line_width", 2) or 2) * 2)

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    if is_stream:
        print(f"Processing live stream: {video_path}")
    else:
        print(f"Processing video: {video_path} ({total_frames} frames at {fps:.1f} FPS)")
    axis = f"y={height // 2}" if direction == "tb" else f"x={width // 2}"
    print(f"Center counting line ({direction}) at {axis}")

    total_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if is_stream:
                print("Stream ended or connection lost.")
            break

        # ObjectCounter.process() draws on `frame` in place; snapshot first so
        # only our bbox-only overlay shows.
        annotate_src = frame.copy()
        results = counter.process(frame)
        total_count = int(getattr(results, "in_count", 0) or 0)
        boxes = extract_solution_boxes(counter)

        annotated = annotate_boxes(annotate_src, boxes, region_pts=region_pts,
                                   region_thickness=line_thickness)
        if writer:
            writer.write(annotated)

        cv2.imshow("Egg Counter", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Playback stopped by user.")
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Annotated video saved to '{save_path}'")
    cv2.destroyAllWindows()

    print(f"Total eggs crossed the counting line: {total_count}")
    return total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Egg Detection and Counting")
    parser.add_argument("input", help="Path to an image, video file, or RTSP stream URL")
    parser.add_argument("--save", default=None, help="Path to save the annotated output")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the YOLO model weights")
    parser.add_argument("--direction", choices=["tb", "lr"], default="tb",
                        help="Conveyor direction: 'tb' (top->bottom) or 'lr' (left->right)")
    args = parser.parse_args()

    is_stream = args.input.startswith("rtsp://") or args.input.startswith("http")

    if is_stream:
        detect_and_annotate_video(
            args.model, args.input, save_path=args.save, direction=args.direction,
        )
    else:
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        ext = pathlib.Path(args.input).suffix.lower()

        if ext in image_exts:
            annotated, count = detect_and_annotate_image(
                args.model, args.input, save_path=args.save
            )
            if annotated is not None:
                cv2.imshow("Egg Detection - Image", annotated)
                print("Press any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        elif ext in video_exts:
            detect_and_annotate_video(
                args.model, args.input, save_path=args.save, direction=args.direction,
            )
        else:
            print(f"Error: Unsupported file extension '{ext}'")
            print(f"Supported images: {image_exts}")
            print(f"Supported videos: {video_exts}")
