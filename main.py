import cv2
import numpy as np
import time
import math

def professional_motion_tracker():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # --- Feature detection parameters (stable + reliable) ---
    feature_params = dict(
        maxCorners=500,
        qualityLevel=0.20,
        minDistance=6,
        blockSize=7
    )

    # --- Lucas–Kanade parameters (high accuracy) ---
    lk_params = dict(
        winSize=(25, 25),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.03)
    )

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read initial frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Professional subtle trail overlay
    trail = np.zeros_like(old_frame)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            old_gray = gray.copy()
            trail = np.zeros_like(frame)
            continue

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # -------- Professional visualization --------
        for new, old in zip(good_new, good_old):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()

            # calculate motion
            dx = x_new - x_old
            dy = y_new - y_old
            dist = math.sqrt(dx * dx + dy * dy)

            # reject unstable points
            if dist > 40:
                continue

            # professional minimal color: soft blue
            color = (220, 220, 255)

            # draw motion trail
            cv2.line(trail, (int(x_old), int(y_old)),
                     (int(x_new), int(y_new)), color, 2)

            # draw direction arrow (clean + minimal)
            cv2.arrowedLine(
                frame,
                (int(x_old), int(y_old)),
                (int(x_new), int(y_new)),
                (255, 255, 255),
                1,
                tipLength=0.3
            )

        # fade older trails for professional look
        trail = cv2.addWeighted(trail, 0.92, np.zeros_like(trail), 0.08, 0)

        output = cv2.add(frame, trail)

        # -------- Professional HUD (FPS + vector count) --------
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        cv2.putText(output, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(output, f"Vectors: {len(good_new)}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

        # Show final
        cv2.imshow("Movment Capture", output)

        # Prepare for next loop
        old_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # refresh features if too few remain
        if len(p0) < 120:
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

