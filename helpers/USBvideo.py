import cv2
from flask import Flask, Response, request

app = Flask(__name__)

def open_camera(index: int, width: int | None, height: int | None, fps: int | None):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows; harmless elsewhere
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap

@app.route("/")
def index():
    # Minimal page to show the stream
    return """
    <html>
      <head><title>USB Camera Stream</title></head>
      <body>
        <h1>USB Camera Stream</h1>
        <img src="/video.mjpg" />
      </body>
    </html>
    """

@app.route("/video.mjpg")
def video():
    # Query params allow quick tweaks: /video.mjpg?cam=0&width=1280&height=720&fps=30&jpeg=80
    cam_index = int(request.args.get("cam", 8))
    width = request.args.get("width")
    height = request.args.get("height")
    fps = request.args.get("fps")
    jpeg_quality = int(request.args.get("jpeg", 80))

    width = int(width) if width else None
    height = int(height) if height else None
    fps = int(fps) if fps else None

    cap = open_camera(cam_index, width, height, fps)

    def gen():
        try:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                # Encode as JPEG
                ok, jpg = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    continue

                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
        finally:
            cap.release()

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    # Listen on all interfaces so other devices on your LAN can view it.
    # For local-only, use host="127.0.0.1"
    app.run(host="0.0.0.0", port=5000, threaded=True)
