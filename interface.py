import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("trainedmodel.pt")


# ---------------------- Detection Functions ---------------------- #
def run_detection_on_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 33

    frame_count = 0
    skip_every = 10 # Detect every Nth frame
    last_annotated = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1

        # Only run detection every N frames
        if frame_count % skip_every == 0:
            results = model(frame, verbose=False)
            last_annotated = results[0].plot()
        else:
            # Reuse last detection result if available
            last_annotated = last_annotated if last_annotated is not None else frame

        cv2.imshow("YOLOv8 - Video Detection (Frame Skipping)", last_annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_detection_on_webcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("YOLOv8 - Webcam Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_detection_on_image(path):
    frame = cv2.imread(path)
    frame = cv2.resize(frame, (640, 360))
    results = model(frame, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLOv8 - Image Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------- Tkinter GUI ---------------------- #
def browse_video():
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if path:
        root.destroy()
        run_detection_on_video(path)


def browse_image():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if path:
        root.destroy()
        run_detection_on_image(path)


def start_webcam():
    root.destroy()
    run_detection_on_webcam()


# ---------------------- Main Window ---------------------- #
root = tk.Tk()
root.title("YOLOv8 Detection - Select Source")
root.geometry("400x200")

tk.Label(root, text="Choose input for YOLOv8 Detection", font=("Arial", 14)).pack(pady=20)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Choose Image", command=browse_image, width=20).grid(row=0, column=0, padx=10, pady=10)
tk.Button(btn_frame, text="Choose Video", command=browse_video, width=20).grid(row=0, column=1, padx=10, pady=10)
tk.Button(btn_frame, text="Start Webcam", command=start_webcam, width=20).grid(row=1, column=0, columnspan=2, pady=10)

def on_closing():
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
