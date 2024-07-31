import os  # For interacting with the operating system
import cv2  # OpenCV library for computer vision tasks
import sys  # For accessing command-line arguments
from zipfile import ZipFile  # For handling ZIP files
from urllib.request import urlretrieve  # For downloading files from the internet

# ========================-Downloading Assets-========================
def download_and_unzip(url, save_path):
    """
    Downloads a zip file from the given URL and extracts its contents to the specified save path.

    Parameters:
    url (str): The URL of the zip file to download.
    save_path (str): The path where the zip file should be saved and extracted.
    """
    print("Downloading and extracting assets...", end="")

    # Downloading the zip file using the urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting the zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory as the zip file.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

# URL of the assets zip file
URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

# Path to save the downloaded zip file
asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")

# Download if the asset ZIP does not exist.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
# ====================================================================

# Default source for video capture (0 for default camera)
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]  # Use the first command-line argument as the video source if provided

# Initialize video capture from the source
source = cv2.VideoCapture(s)

# Name of the window to display video
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained model for face detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Model parameters
in_width = 300  # Width of the input image for the model
in_height = 300  # Height of the input image for the model
mean = [104, 117, 123]  # Mean values for image normalization
conf_threshold = 0.5  # Confidence threshold for detections

# Main loop to capture and process video frames
while cv2.waitKey(1) != 27:  # Exit loop when 'ESC' key is pressed
    has_frame, frame = source.read()  # Read a frame from the video source
    if not has_frame:
        break  # Exit the loop if no frame is captured
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]  # Get the height of the frame
    frame_width = frame.shape[1]  # Get the width of the frame

    # Create a 4D blob from the frame for input to the model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Set the blob as input to the network
    net.setInput(blob)
    detections = net.forward()  # Perform forward pass to get detections

    # Loop over detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Get the confidence score of the detection
        if confidence > conf_threshold:  # Check if confidence is above the threshold
            # Get the bounding box coordinates
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            # Draw the bounding box around the detected face
            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence  # Create a label with the confidence score
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw the label background
            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            # Put the label text on the frame
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Get the inference time and display it on the frame
    t, _ = net.getPerfProfile()  # Get performance profile
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())  # Calculate inference time
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))  # Display inference time
    
    # Show the frame in the window
    cv2.imshow(win_name, frame)

# Release the video source and destroy the window
source.release()  # Release the video capture object
cv2.destroyWindow(win_name)  # Close the window
