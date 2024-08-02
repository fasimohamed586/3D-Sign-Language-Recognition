import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.preprocessing
import tempfile
import time

def detect_sign_language(video_file_path, SEQUENCE_LENGTH, model, CLASSES_LIST):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    video_file_path:  The path of the video stored in the disk on which the action recognition is to be performed.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    '''
    # # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
    # CLASSES_LIST = ["above", "afraid","afternoon","angry","backpack","balloon","banana","bicycle","call","car","church","clock","dinner","eat","drink","help", "sorry","wait", "what","write"]
    # Specify the height and width to which each video frame will be resized in our dataset.
    IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64
    # Initialize the VideoCapture object to read from the video file.
    video_reader = cv2.VideoCapture(video_file_path)
    # Get the width and height of the video.
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Declare a list to store video frames we will extract.
    frames_list = []
    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''
    # Get the number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),1)
    # Iterating the number of times equal to the fixed length of sequence.
    for frame_counter in range(SEQUENCE_LENGTH):
        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        # Read a frame.
        success, frame = video_reader.read() 
        # Check if frame is not read properly then break the loop.
        if not success:
            break
        # Resize the Frame to fixed Dimensions.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
        normalized_frame = resized_frame / 255
        # Appending the pre-processed frame into the frames list
        frames_list.append(normalized_frame)
    # Passing the  pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities =  model.predict(np.expand_dims(frames_list, axis = 0))[0]   
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    # Display the predicted action along with the prediction confidence.
    st.write(f'Action Predicted : {predicted_class_name}')
    st.write(f'Confidence : {predicted_labels_probabilities[predicted_label]}')
    # Release the VideoCapture object. 
    video_reader.release()

def predict_live_action(frames_list, SEQUENCE_LENGTH, model, CLASSES_LIST):
    '''
    This function will perform single action recognition prediction on a video using the LRCN model.
    Args:
    frames_list:  List of frames to predict the action on.
    SEQUENCE_LENGTH:  The fixed number of frames of a video that can be passed to the model as one sequence.
    model: Loaded LRCN model.
    '''
    # Convert frames to numpy array and normalize
    frames_array = np.array(frames_list) / 255.0
    # Passing the pre-processed frames to the model and get the predicted probabilities.
    predicted_labels_probabilities = model.predict(np.expand_dims(frames_array, axis=0))[0]
    # Get the index of class with highest probability.
    predicted_label = np.argmax(predicted_labels_probabilities)
    # Get the class name using the retrieved index.
    predicted_class_name = CLASSES_LIST[predicted_label]
    # Return the predicted action along with the prediction confidence.
    return predicted_class_name, predicted_labels_probabilities[predicted_label]

def display_intro():
    st.title("3D Sign Language Recognition System")
    # st.image(r"C:\Users\user2\Desktop\logo_image.jpeg", use_column_width=False)
    st.write("""
             
    DEVELOPED BY MOHAMED FASI ALTAF
             
    Welcome to the 3D Sign Language Recognition app. This advanced application uses 3D recognition technology
    to identify and interpret sign language gestures. You can use this app to either upload a video or
    use your webcam to see real-time sign language recognition.
    """)

def display_about():
    st.sidebar.title("About")
    st.sidebar.write("""
    This application demonstrates 3D sign language recognition. Developed by **MOHAMED FASI ALTAF**, 
    it leverages cutting-edge technology to recognize gestures. For more details, visit
    [Github Repositary].
    """)
    
def display_statistics():
    st.sidebar.title("Statistics")
    st.sidebar.write("""
    - **Total Gestures Recognized: 20**
    - **Accuracy Rate: 92%**
    """)
    fig, ax = plt.subplots()
    categories = ['Category 1', 'Category 2', 'Category 3']
    values = [25, 50, 75]
    ax.bar(categories, values)
    ax.set_ylabel('Recognition Count')
    ax.set_title('Recognition Statistics')
    st.sidebar.pyplot(fig)


def main():
    st.set_page_config(page_title="3D Sign Language Recognition", layout="wide")
    
    display_intro()
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose Input Method", ["Upload Video", "Use Webcam"])
    
    # Specify the height and width to which each video frame will be resized in our dataset.
    SEQUENCE_LENGTH = 20
    # Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
    CLASSES_LIST = ["above", "afraid","afternoon","angry","backpack","balloon","banana","bicycle","call","car","church","clock","dinner","eat","drink","help", "sorry","wait", "what","write"]
    # Load Model
    model = tf.keras.models.load_model('LRCN_model___Date_Time_2024_08_02__18_34_16___Loss_0.603868842124939___Accuracy_0.8849999904632568.h5')

    if option == "Upload Video":
        st.write("""**NOTE:** This app work best only when you uplode Video of ASL.""")
        f = st.file_uploader("Please upload a Video of ASL Sign which You want to Translate")

        if f is None:
            st.write("""Please upload an Video file""")
        else:
            if st.button("Predict"):
                try:
                    #read video & frames from upload
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(f.read())
                    video_file = open(tfile.name, 'rb')            
                    video_bytes = video_file.read()
                    # Perform Single detection on the Uploaded Video.
                    detect_sign_language(tfile.name, SEQUENCE_LENGTH, model, CLASSES_LIST)
                    st.success("Successfully Predicted")
                    st.video(video_bytes)
                except:
                    st.error("Invalid Video Type For This Model Or The Uploaded video does not belong to ASL Category")
                    st.video(video_bytes)

    elif option == "Use Webcam":
        # Initialize variables
        frames_list = []
        predicted_class_name = ""
        confidence = 0

        # Initialize webcam
        cap = cv2.VideoCapture(1)

        if 'start' not in st.session_state:
            st.session_state.start = False

        if not st.session_state.start:
            if st.button("Start Webcam", key="start_button"):
                st.session_state.start = True
        else:
            if st.button("Stop Webcam", key="stop_button"):
                st.session_state.start = False
                cap.release()

            frame_placeholder = st.empty()
            while st.session_state.start:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image")
                    break

                # Resize the Frame to fixed Dimensions.
                resized_frame = cv2.resize(frame, (64, 64))
                # Append the pre-processed frame into the frames list
                frames_list.append(resized_frame)

                # If we have collected enough frames, make a prediction
                if len(frames_list) == SEQUENCE_LENGTH:
                    predicted_class_name, confidence = predict_live_action(frames_list, SEQUENCE_LENGTH, model, CLASSES_LIST)
                    frames_list = []

                # Display the frame and the prediction on the screen
                display_frame = frame.copy()
                cv2.putText(display_frame, f'Action: {predicted_class_name} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                frame_placeholder.image(display_frame, channels="BGR")

                # Sleep to allow Streamlit to update the UI
                time.sleep(0.1)

        # Release the webcam
        cap.release()

    display_about()
    display_statistics()

if __name__ == "__main__":
    main()
