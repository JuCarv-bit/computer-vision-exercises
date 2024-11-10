import cv2
import numpy as np
from IPython.display import display, Image, clear_output
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('gatinho.mp4')
ret, first_frame = cap.read()
if not ret:
    print("Failed to load video.")
    cap.release()
    exit()

gray_first = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Define the point to be tracked
point = []

def select_point(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [(x, y)]
        cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("First Frame", first_frame)

cv2.namedWindow("First Frame")
cv2.setMouseCallback("First Frame", select_point)

while True:
    cv2.imshow("First Frame", first_frame)
    if len(point) > 0:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("First Frame")

lk_params = dict(winSize  = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

mask = np.zeros_like(first_frame)

p0 = np.array([point], dtype=np.float32)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (first_frame.shape[1], first_frame.shape[0]))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_first, gray_frame, p0, None, **lk_params)
    
    if st[0][0] == 1:
        a, b = p1.ravel()
        a, b = int(a), int(b)  # Convert to integers
        cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        mask = cv2.line(mask, tuple(p0.ravel().astype(int)), (a, b), (0, 255, 0), 2)
        frame = cv2.add(frame, mask)
        p0 = p1
        gray_first = gray_frame.copy()

    out.write(frame)
    if frame_count % 20 == 0:  # 
        _, img = cv2.imencode('.jpg', frame)
        final_image = 'final_frame.jpg'
        cv2.imwrite(final_image, frame)
        display(Image(data=img))
        clear_output(wait=True)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

