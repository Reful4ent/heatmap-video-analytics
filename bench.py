import time, cv2
from ultralytics import YOLO

path = r"C:\Users\dima2\Downloads\13108899_3840_2160_30fps (1).mp4"
cap = cv2.VideoCapture(path)
print("разрешение:", int(cap.get(3)), "x", int(cap.get(4)), "fps видео:", cap.get(5))

t = time.time()
frames = []
for _ in range(60):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
print(f"чтение кадров: {len(frames)/(time.time()-t):.1f} кадр/с")

for name in ("yolov8l.pt", "yolov8m.pt"):
    m = YOLO(name)
    m.predict(frames[0], device=0, verbose=False)  # прогрев
    for half in (False, True):
        t = time.time()
        for f in frames:
            m.predict(f, device=0, half=half, imgsz=640, verbose=False)
        print(f"{name} half={half}: {len(frames)/(time.time()-t):.1f} кадр/с")