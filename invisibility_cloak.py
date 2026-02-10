import cv2
import numpy as np
from ultralytics import YOLO


def build_person_mask(result, frame_shape):
    """
    Build a single binary mask (uint8, 0/255) for all detected persons.
    Returns None if no valid person masks are available.
    """
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None:
        return None

    if masks.data is None or boxes.cls is None:
        return None

    mask_data = masks.data  # shape: (n, h, w)
    if mask_data.numel() == 0:
        return None

    # Filter for class 0 (person)
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    person_idxs = np.where(cls == 0)[0]
    if len(person_idxs) == 0:
        return None

    # Combine all person masks into a single mask
    # masks.data is float tensor [0..1]; threshold to binary
    combined = None
    for idx in person_idxs:
        m = mask_data[idx].detach().cpu().numpy()
        m = (m > 0.5).astype(np.uint8) * 255
        combined = m if combined is None else cv2.bitwise_or(combined, m)

    if combined is None:
        return None

    # Resize mask to match frame resolution if needed
    h, w = frame_shape[:2]
    if combined.shape[0] != h or combined.shape[1] != w:
        combined = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST)

    return combined


def main():
    print("Loading Invisibility Cloak (YOLOv8-Seg)...")
    model = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    background = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Calibration state: show instructions until background is captured
        if background is None:
            cv2.putText(
                display,
                "Press 'b' to capture background",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # Invisibility state: run segmentation
            try:
                results = model(frame, verbose=False)
                if results and len(results) > 0:
                    person_mask = build_person_mask(results[0], frame.shape)
                else:
                    person_mask = None
            except Exception:
                person_mask = None

            if person_mask is not None:
                # Bitwise trick:
                # 1) inv_mask keeps everything except the person.
                # 2) person_mask keeps only the background where the person is.
                inv_mask = cv2.bitwise_not(person_mask)
                foreground = cv2.bitwise_and(frame, frame, mask=inv_mask)
                background_only = cv2.bitwise_and(background, background, mask=person_mask)
                display = cv2.add(foreground, background_only)

        cv2.imshow("Invisibility Cloak", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            background = frame.copy()
            print("Background captured.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
