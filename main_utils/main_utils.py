import cv2, os


def launch_recording(W,H, folder='Videos'):
    os.makedirs(folder,exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    num_files = len(os.listdir(folder))
    vid_loc = f'Videos/smca{num_files}.mkv'
    video_out = cv2.VideoWriter(vid_loc, fourcc, 30.0, (W,H))

    return video_out