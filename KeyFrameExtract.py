import cv2
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import os
from pathlib import Path
import argparse


class KeyFramesExtractUtils:

    def __init__(self, video_path=None, save_path=None, csv_path=None, window_size=7):
        self.video_path = video_path
        self.save_path = save_path
        self.csv_path = csv_path
        self.window_size = window_size
        self.half_window = window_size // 2
        self.key_frame_interval = 10

    def extract_keyframe(self, method="use_local_maxima"):
        print("method===>", method)
        print(f"🎯 window_size={self.window_size}")

        len_window = 50

        cap = cv2.VideoCapture(str(self.video_path))
        curr_frame, prev_frame = None, None

        frame_diffs = []
        frames = []
        video_frames = []

        j = 0

        # ================== Reading Video ================== #
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
            success, frame = cap.read()

            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_frame = gray

            if prev_frame is not None:
                diff = cv2.absdiff(curr_frame, prev_frame)
                diff_mean = np.mean(diff)

                real_frame_id = j

                frame_diffs.append(diff_mean)
                frames.append(Frame(real_frame_id, diff_mean))
                video_frames.append((real_frame_id, frame))

            prev_frame = curr_frame
            j += self.key_frame_interval

        cap.release()

        # ================== Reading AU Data ================== #
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        au_path = os.path.join(self.csv_path, f"{video_name}.csv")

        if not os.path.exists(au_path):
            print(f"⚠️ Missing AU file: {au_path}")
            return

        au_data = pd.read_csv(au_path)
        au_data = au_data.sort_values("frame")
        au_data = au_data.set_index("frame")

        if len(frame_diffs) == 0:
            return

        # ================== Keyframe detection ================== #
        diff_array = np.array(frame_diffs)
        sm_diff = smooth(diff_array, len_window)

        idxs = np.asarray(argrelextrema(sm_diff, np.greater))[0]

        keyframe_ids = set()
        for i in idxs:
            if i - 1 >= 0:
                keyframe_ids.add(frames[i - 1].id) 

        # ================== Extracting from small window ================== #
        for center in sorted(keyframe_ids):

            folder = os.path.join(self.save_path, f"small_window_{center}")
            os.makedirs(folder, exist_ok=True)

            au_window = []

            half = self.window_size // 2

            start = center - half
            end = center + half

            for frame_id in range(start, end + 1):

                if frame_id in au_data.index:
                    row = au_data.loc[frame_id].copy()
                    row["frame"] = frame_id   
                    au_window.append(row)

            if au_window:
                df = pd.DataFrame(au_window)

                cols = ["frame"] + [c for c in df.columns if c != "frame"]
                df = df[cols]

                df.to_csv(
                    os.path.join(folder, f"frame_window_{center}.csv"),
                    index=False
                )


class Frame:
    def __init__(self, id, diff):
        self.id = id  
        self.diff = diff

    def __lt__(self, other):
        return self.id < other.id


def smooth(x, window_len=13, window='hanning'):
    if len(x) < window_len:
        return x
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x,
              2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=str, default="7")
    parser.add_argument("--base-root", type=str, default="/raw_data")
    return parser.parse_args()

def main():
    args = parse_args()
    base_root = Path(args.base_root)
    window_sizes = [int(x) for x in args.window_size.split(",")]
      
    q_list = ["q3", "q4", "q5", "q6"]
    splits = ["train", "valid", "test"]

    for window_size in window_sizes:
        print(f"\n================ Window Size: {window_size} ================\n")

        for q in q_list:
            for split in splits:

                print(f"\n🚀 Processing {q} | {split}")

                video_dir = base_root / q / split
                csv_dir = base_root / q / f"{split}_csv"
                output_root = base_root / q / f"{split}_results"

                video_list = list(video_dir.glob("*.mp4"))

                print(f"📦 Found {len(video_list)} videos")

                for video_path in video_list:
                    video_name = video_path.stem
                    save_dir = output_root / video_name
                    save_dir.mkdir(parents=True, exist_ok=True)

                    if any(save_dir.glob("small_window_*")):
                        print(f"✅ Found existing results for: {video_name}")
                        continue

                    keyFrame = KeyFramesExtractUtils(
                        video_path=str(video_path),
                        save_path=str(save_dir),
                        csv_path=str(csv_dir),
                        window_size=window_size
                    )

                    keyFrame.extract_keyframe(method="use_local_maxima")

if __name__ == "__main__":
    main()