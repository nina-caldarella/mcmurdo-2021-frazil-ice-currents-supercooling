import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cv2
import os
import time
import glob
import gsw


def get_frame_at_time(path_v, video_file, time_in_s):
    """
    this function pulls the frames out of the video at given times in seconds since the start of the video
    """
    # Open the video file
    video = cv2.VideoCapture(path_v + video_file)

    # Get the frames per second (fps) and total frames
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the frame index corresponding to the given time (ms)
    frame_index = int((time_in_s) * fps)

    # Set the video to the calculated frame index
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame
    ret, frame = video.read()

    # Release the video capture object
    video.release()

    #    # Generate a directory to save the individual frames
    #    os.makedirs(path_v + "box4/frames/" + video_file[:-4], exist_ok=True)

    # Check if the image is empty before saving
    if frame.size == 0:
        print("Error: Image is empty, cannot save!")
    #     else:
    #         cv2.imwrite(
    #             path_v + "box4/frames/" + video_file[:-4] + "/" + str(time_in_s) + "_s.jpg",
    #             frame,
    #         )
    #
    return frame


def extract_frames_period(path_v, video_file, start_time):
    """
    Extract every other frame from 4k videos over selected 48 seconds period
    path_v: general path for video files
    video_number: the number of the video in numeric order of name
    start_time: start time for extraction in minutes
    """
    output_dir = (os.path.join(
        path_v, video_file[:-10] + "_frames_" + str(start_time))
    )
    #if not os.path.exists(output_dir):
    print("Extracing frames...")
    # Directory to save the frames
    os.makedirs(output_dir, exist_ok=True)

    start_time = start_time * 60  # seconds
    dt = 0.8 * 60  # Duration in seconds

    # Open the video file
    video = cv2.VideoCapture(os.path.join(path_v, video_file))
    if not video.isOpened():
        print("Error: Could not open video. It's a 100 GB file, request access.")
        print("Processing will continue with provided frames.")

    # Get the frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate frame range based on start time and duration
    start_frame = int(start_time * fps)
    end_frame = int((start_time + dt) * fps)

    # Loop through the frames, saving every other frame
    video.set(
        cv2.CAP_PROP_POS_FRAMES, start_frame
    )  # Start at the first frame of the range
    frame_idx = start_frame
    i = 1
    while frame_idx < end_frame:
        # Read the frame
        ret, frame = video.read()
        if not ret:
            print("Video file path: ", os.path.join(path_v, video_file))
            print(f"Total frames {video.get(cv2.CAP_PROP_FRAME_COUNT)}")
            print(f"Warning: Skipping unreadable frame at position {video.get(cv2.CAP_PROP_POS_FRAMES)}")
            break  # Stop if frame could not be read (e.g., end of video)

        # Save the frame
        frame_filename = os.path.join(output_dir, f"frame_{i}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Skip to the next frame (every other frame)
        frame_idx += 2
        print("saving frame ", i)
        i += 1
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Release the video capture object
        video.release()


def disp_pressure_video(ds2, video_timing, save_fig=None):
    """
    display combined pressure and 4k video timing
    """
    # display combined pressure and 4k video timing
    # display pressure
    fig2, ax = plt.subplots()
    ax.plot(ds2.datetime, ds2.depth, label = "depth")

    for i in range(len(video_timing["box4 start (UTC)"])):
        y1 = pd.to_datetime(video_timing["box4 start (UTC)"][i])
        y2 = pd.to_datetime(video_timing["box4 start (UTC)"][i])\
                        + pd.Timedelta(video_timing["box4 duration (minutes)"][i], "minutes")
        if (i>0) & (pd.isna(y1)==False):
            ax.axvspan(y1, y2, color='orange', alpha=0.5)
        elif (pd.isna(y1)==False):
            print(y1, y2)
            ax.axvspan(y1, y2, color='orange', alpha=0.5, label = "4k video")
    # set interval for x-ticks
    xtick_interval_minutes = int(
        np.round(np.sum(video_timing["box4 duration (minutes)"] / 4), -1)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # Display hours:minutes
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=xtick_interval_minutes))
    day = pd.Timestamp(ds2.datetime[0].item()).day
    month = pd.Timestamp(ds2.datetime[0].item()).month
    ax.set_xlabel("Time (hours:minutes on " + str(day) + "-" + str(month) + "-2021)")
    # Rotate x-axis tick labels
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")  # Rotate the tick labels
    ax.set_ylabel("Depth (m)")
    plt.tight_layout()
    plt.rcParams.update({"font.size": 18})
    ax.set_title("Icefin " + str(day) + "-" + str(month) + " dive data timeline")
    if save_fig != None:
        fig2.savefig(
            save_fig + str(day) + "-" + str(month) + "_timeline.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        ax.legend()
    return fig2, ax


def four_epochs_analysis(
    ds2, video_timing, video_number, start_time_minutes, end_time_minutes
):
    """
    This function plots the depth with four points equally spaced in time with error bars corresponding with a certain amount of frames. This is used to select periods for video analysis of frazil ice.
    """
    start_time = pd.to_datetime(video_timing["box4 start (UTC)"][video_number])
    end_time = pd.to_datetime(
        video_timing["box4 start (UTC)"][video_number]
    ) + pd.to_timedelta(video_timing["box4 duration (minutes)"][video_number], unit="m")
    ds2_sel = ds2.sel(datetime=slice(start_time, end_time))

    # Calculate relative time in minutes
    rel_time = pd.Series(ds2_sel.datetime) - start_time
    rel_time_minutes = rel_time.dt.total_seconds() / 60

    # Plot the pressure data
    fig2, ax = plt.subplots()
    ax.plot(rel_time_minutes, ds2_sel.pressure_dbar, label="Depth")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Depth (m)")

    # set title with hour:minute
    hour = pd.Timestamp(ds2_sel.datetime[0].item()).hour
    minute = pd.Timestamp(ds2_sel.datetime[0].item()).minute
    ax.set_title("Video at " + str(hour) + ":" + str(minute) + " UTC")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")  # Rotate the tick labels

    # Select timings with equal spacing to match the ramps
    time_sel = np.linspace(start_time_minutes, end_time_minutes, 4)  # time in minutes

    # Interpolate or find the corresponding pressure values for selected times
    pressure_sel = np.interp(time_sel, rel_time_minutes, ds2_sel.pressure_dbar)

    # Define the horizontal error bar size (adjust as needed)
    xerr = 0.8  # Example error bar length (in minutes)
    print("bar length ", xerr * 60 * 60, " frames")
    # Plot larger red dots at the selected times on the pressure curve
    ax.scatter(
        time_sel, pressure_sel, color="red", s=40, zorder=5, label="Selected Points"
    )  # 's' controls the size of the points
    depth_sel = -1*gsw.z_from_p(pressure_sel, -77.8667)
    print(depth_sel)
    # Plot horizontal error bars (bars only along x-axis, no vertical error)
    ax.errorbar(
        time_sel,
        depth_sel,
        xerr=xerr,
        fmt="o",
        color="red",
        ecolor="black",
        capsize=5,
        linestyle="None",
    )
    ax.invert_yaxis()
    # Number the points by adding text labels next to the points
    for i, (x, y) in enumerate(zip(time_sel, pressure_sel), start=1):
        ax.text(x, y, f"{i}", fontsize=18, color="black", ha="center", va="bottom")
    # Add a legend to clarify the plotted data
    ax.legend(loc="best")
    plt.tight_layout()
    plt.rcParams.update({"font.size": 21})
    return time_sel, depth_sel

def subtract_background(path_v, video_file, start_time):
    """
    Subtract background (average gray-scale image) from a selection of frames. Also apply CLAHE filter for enhancement.

    path_v: folder of video file
    video_file: name of video file
    start_time: start time of frame selection
    """
    # Path to the folder with images
    image_folder = os.path.join(path_v, video_file[:-10]+ "_frames_"+ str(start_time))
    
    # Create an output directory to save the background-subtracted images
    output_dir = os.path.join(image_folder, "background_subtracted")
    if not os.path.exists(os.path.join(image_folder, "background_subtracted")):
        os.makedirs(output_dir, exist_ok=True)
        # Get a sorted list of image paths 
        image_files = sorted(glob.glob(os.path.join(image_folder, '*.jpg'))) 

        # Read the first image to get the dimensions (height, width) 
        first_image = cv2.imread(image_files[0]) 
        height, width, channels = first_image.shape 

        # Initialize a numpy array to hold the sum of images 
        image_sum = np.zeros((height, width, channels), dtype=np.float32) 

        # Loop through all images and add them to the sum 
        for image_file in image_files: 
            image = cv2.imread(image_file) 
            image_sum += image.astype(np.float32) 

        # Calculate the average image by dividing the sum by the number of images 
        average_image = image_sum / len(image_files) 

        # Convert the average image to uint8 
        average_image = np.uint8(average_image) 
        cv2.imwrite(os.path.join(output_dir, "background.jpg"), average_image)
        
        # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) 

        # Create a mask with the same size as the image (all black) 
        mask = np.ones(image.shape[:2], dtype=np.uint8) 

        # Define the rectangular region (840, 0) to (1080, 1000) and set it to 0 (black) 
        mask[470:1080, 540:1100] = 0 
         
        # Subtract the average image from each frame, apply CLAHE, and crop 
        for i, image_file in enumerate(image_files): 
            image = cv2.imread(image_file) 

            # Step 1: Background subtraction 
            foreground = cv2.absdiff(image, average_image)  # Subtract the average image 

            # Step 2: Convert to grayscale for CLAHE 
            gray_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY) 

            # Step 3: Apply CLAHE 
            clahe_foreground = clahe.apply(gray_foreground) 

            # Step 4: Convert back to BGR for saving as a colored image 
            clahe_foreground_bgr = cv2.cvtColor(clahe_foreground, cv2.COLOR_GRAY2BGR) 

            # Apply the mask to the image using bitwise_and 
            masked_image = cv2.bitwise_and(clahe_foreground_bgr, clahe_foreground_bgr, mask=mask)  
            
            # Step 6: Save the cropped and enhanced image
            output_file = os.path.join(output_dir, f"frame_{i+1}_foreground_clahe_masked.jpg")
            cv2.imwrite(output_file, masked_image)
    
        print(f"Background-subtracted, CLAHE-enhanced, and cropped images are saved in {output_dir}")
    return output_dir

def mp4_from_processed_frames(path_processed):
    import re
    if not os.path.exists(os.path.join(path_processed, "timelapse_video.mp4")):
        print("generating mp4 file...")

        image_folder = path_processed
        output_video = os.path.join(image_folder, "timelapse_video.mp4")
        frame_rate = 30

        # Glob and natural sort by numeric chunk in filename
        images = glob.glob(os.path.join(image_folder, "*.jpg"))

        def num_key(p):
            m = re.search(r'(\d+)', os.path.basename(p))
            return int(m.group(1)) if m else float('inf')  # non-matching names go last

        images = sorted(images, key=num_key)

        if not images:
            print("No images found in the specified folder.")
            raise KeyboardInterrupt

        first_frame = cv2.imread(images[0])
        if first_frame is None:
            raise IOError(f"Could not read first image: {images[0]}")
        height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

        for image_path in images:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            # Optional: ensure size matches, resize if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video.write(frame)

        video.release()
        print(f"Video saved as {output_video}")


#if __name__ == "main":
