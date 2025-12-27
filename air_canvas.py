import cv2
import numpy as np
import mediapipe as mp
import time

# ===============================
# MediaPipe setup
# ===============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ===============================
# Webcam
# ===============================
WIDTH, HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

# ===============================
# Canvas
# ===============================
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# ===============================
# COLORS (NAME : BGR)
# ===============================
colors = [
    ("BLUE",   (255, 0, 0)),
    ("GREEN",  (0, 255, 0)),
    ("RED",    (0, 0, 255)),
    ("YELLOW", (0, 255, 255)),
    ("PURPLE", (255, 0, 255)),
    ("ORANGE", (0, 165, 255)),
    ("CYAN",   (255, 255, 0)),
    ("WHITE",  (255, 255, 255))
]

draw_color = colors[0][1]

# ===============================
# Drawing settings
# ===============================
brush_thickness = 6
eraser_thickness = 60
prev_x, prev_y = 0, 0

# ===============================
# FPS
# ===============================
p_time = 0

# ===============================
# Draw SIDE color palette
# ===============================
def draw_side_palette(img):
    block_height = HEIGHT // len(colors)
    for i, (name, col) in enumerate(colors):
        y1 = i * block_height
        y2 = y1 + block_height
        cv2.rectangle(img, (0, y1), (130, y2), col, -1)

        text_color = (0, 0, 0) if col == (255,255,255) else (255,255,255)
        cv2.putText(
            img, name,
            (10, y1 + block_height // 2 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )

# ===============================
# Finger detection
# ===============================
def fingers_up(lm):
    tips = [8, 12, 16, 20]
    fingers = []
    for tip in tips:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers

# ===============================
# Main loop
# ===============================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_side_palette(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            fingers = fingers_up(lm)

            x = int(lm[8].x * WIDTH)
            y = int(lm[8].y * HEIGHT)

            # 🎨 Color selection
            if x < 130:
                index = y // (HEIGHT // len(colors))
                if index < len(colors):
                    draw_color = colors[index][1]
                prev_x, prev_y = 0, 0

            # ✊ FIST → ERASER
            elif fingers == [False, False, False, False]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         (0, 0, 0), eraser_thickness)
                prev_x, prev_y = x, y

            # ☝️ DRAW
            elif fingers == [True, False, False, False]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y),
                         draw_color, brush_thickness)
                prev_x, prev_y = x, y

            # ✌️ MOVE
            elif fingers == [True, True, False, False]:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (x, y), 8, (255, 255, 255), cv2.FILLED)

            else:
                prev_x, prev_y = 0, 0

            mp_draw.draw_landmarks(frame, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

    # ===============================
    # Merge canvas with frame
    # ===============================
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    # ===============================
    # FPS display
    # ===============================
    c_time = time.time()
    fps = int(1 / (c_time - p_time)) if c_time != p_time else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {fps}", (500, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("AIR CANVAS – SIDE PALETTE", frame)

    key = cv2.waitKey(1) & 0xFF

    # 💾 SAVE OPTION
    if key == ord('s'):
        cv2.imwrite("air_canvas_output.png", canvas)
        print("Saved: air_canvas_output.png")

    # ❌ EXIT
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
