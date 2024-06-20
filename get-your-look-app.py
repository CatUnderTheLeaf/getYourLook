import streamlit as st
import os
import numpy as np
import json
import keras
from mtcnn import MTCNN
import tensorflow as tf
import cv2
import requests
import imutils


# make a wide layout, not with a fixed width in the center 
st.set_page_config(
    page_title="Get haircut recommendations",
    layout="wide",
)

# Page layout

st.title('Get haircut recommendations')

top = st.container()
left_column, right_column = st.columns([1, 3])
bottom = st.container()

############################################################

@st.cache_resource
def load_detector():
    """load MTCNN detector

    Returns:
        MTCNN(object): face detector
    """
    return MTCNN()

def preprocess_image(image, img_size = (150, 150)):
    """detect face, crop and resize it, pack it to a batch

    Args:
        image (ndarray): image astype('float32')
        img_size (tuple, optional): target image size for nn model. Defaults to (150, 150).

    Returns:
        tf.Tensor: batched image tensor
    """
    
    detector = load_detector()
    min_conf = 0.9
    offset = 20
    new_batch = []

    h,w,ch = image.shape
    area = 0
    final_face = None
    detections = detector.detect_faces(image)

    # transform only face with the biggest area 
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            object = image[max(y-offset,0):min(y+height+offset,h), max(0,x-offset):min(w,x+width+offset), :]
            object_area = object.shape[0]*object.shape[1]
            if (object_area > area):
                area = object_area
                final_face = object
    final_face = cv2.resize(final_face, img_size)
    
    new_batch.append(final_face.astype(int))
    results_tensor = tf.stack(new_batch)
    return results_tensor

def download_model():
    """download keras model from google drive storage
        and display download progress
    """
    def save_response_content(response, destination):
        try:
            progress_bar = st.progress(0)
            length = st.secrets["MODEL_SIZE"]
            CHUNK_SIZE = 32768
            
            with open(destination, "wb") as f:
                counter = 0.0
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        counter += CHUNK_SIZE
                        
                        progress_bar.progress(min(counter / length, 1.0), text="downloading model weights file")
        finally:
            
            if progress_bar is not None:
                progress_bar.empty()

    destination = 'face_shape_model.keras'
    if os.path.exists(destination) and os.path.getsize(destination) == st.secrets["MODEL_SIZE"]:
        return
    
    id=st.secrets["MODEL_ID"]    
    URL = st.secrets["MODEL_URL"]
    
    session = requests.Session()

    params = {'id': id, 
              'confirm': 't',
              'export': 'download',
              'uuid': st.secrets["UUID"] }

    response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

@st.cache_resource(show_spinner="Loading model weights...")
def load_nn_model():
    """load nn model and cache it, so it is loaded only once

    Returns:
        Keras model: faceShape classification model
    """
    # download weights file if it is not uploaded
    return keras.saving.load_model("face_shape_model.keras", compile=False)

def get_face_shape(model, batched_img):
    """get model classification on the batched image

    Args:
        model (Keras model): faceShape classification model
        batched_img (tf.Tensor): batched image tensor

    Returns:
        str: one of the 5 face shapes
    """    
    class_names = ['heart', 'oblong', 'oval', 'round', 'square']

    predicted_batch = model.predict(batched_img)
    predicted_id = np.argmax(predicted_batch, axis=1)

    return class_names[predicted_id[0]]

@st.cache_data
def load_recommendations():
    """load recommendation file with 
    hair cut text prompts and comments

    Returns:
        json object: prompts and comments as json object
    """    
    try:
        with open("hair_cut/recommendationPrompts.json") as stream:
            try:
                return json.load(stream)
            except ValueError:  # includes simplejson.decoder.JSONDecodeError
                st.text('Decoding JSON has failed')
    except FileNotFoundError:
        st.text('This file does not exist, try again!')

def recommend(model, face_img):
    """recommend a hair cut based on user image

    Args:
        model (Keras model): faceShape classification model
        face_img (ndarray): user image astype('float32')

    Returns:
        json object: hair cut recommendations
    """ 
    
    processed_face = preprocess_image(face_img)
    face_shape = get_face_shape(model, processed_face)
    recommendations = load_recommendations()
    if recommendations is not None:
        return recommendations[face_shape]
    else:
        return None

@st.cache_data
def load_resized_image(length, cut, width):
    """load image with a white border as padding

    Args:
        length (str): haircut length
        cut (str): haircut name
        width (int): width of the image

    Returns:
        ndarray: image with a border
    """
    image_path = "/".join(['hair_cut','gen_images', length, cut +'.png'])
    img = cv2.imread(image_path)   # reads an image in the BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize image with the same aspect ratio
    resized_image = imutils.resize(img, width=width)
    # add border/padding
    top, bottom, left, right = [10]*4
    img_with_border = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255,255,255])
    return img_with_border

def main():
    
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    if 'display_result' not in st.session_state or st.session_state.display_result==False:
        st.session_state.display_result = False
    else:
        st.session_state.display_result = True

    def btn_b_callback():
        st.session_state.display_result=False
        st.session_state.uploaded_file = None
        
    def btn_upload_callback():
        st.session_state.display_result = True
        st.session_state.uploaded_file = st.session_state.upload

    def btn_photo_callback():
        st.session_state.display_result = True
        st.session_state.uploaded_file = st.session_state.photo

    def get_photo(img_file, key):
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        face_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        st.session_state[key] = face_img

    # wait before nn model is loaded
    # only after load everything else
    download_model()
    model = load_nn_model()
    # show the possibility to upload image file
    # and after successful upload - show button
    if not st.session_state.display_result:
        with top:
            tab1, tab2 = st.tabs(['Use a camera', 'Upload an image'])

            tab1_column1, tab1_column2 = tab1.columns(2)
            photo = tab1_column1.camera_input(label_visibility = "collapsed", label='camera input')
            if photo:
                get_photo(photo, 'photo')
                button_a = tab1_column2.button('Get recommendations', on_click=btn_photo_callback, type='primary', key='but_photo')

            tab2_column1, tab2_column2 = tab2.columns(2)
            image = tab2_column1.file_uploader(label_visibility = "collapsed", type=['png', 'jpg'], label='upload picture')
            if image:
                get_photo(image, 'upload')
                tab2_column1.image(st.session_state.upload)
                button_a = tab2_column2.button('Get recommendations', on_click=btn_upload_callback, type='primary', key='but_upload')            

    # when button 'Get recommendations' is pressed
    # hide upload content and show only recommendations content
    # show button to reset recoomendations
    if st.session_state.display_result:
        face_img = st.session_state.uploaded_file
        with top:
            if face_img is not None:

                recommendations = None

                with st.spinner('Your faceshape is analysed...'):                    
                    recommendations = recommend(model, face_img)
                    button_b = top.button('Reset', on_click=btn_b_callback, type='primary')
    
                if recommendations is not None:
                    # format recommendations in the botom section
                    top.subheader(f"Congratulations! Your faceshape is {recommendations['faceShape'].upper()}!", divider='rainbow')
                    left_column.image('/'.join(['hair_cut', 'images', recommendations['faceShape']+'.jpg']), use_column_width=True)
                    with right_column:
                        does, donts = st.tabs(['#### :green-background[Do\'s:]', '#### :red-background[Don\'ts:]'])
                        does.success(('\n\n').join(recommendations['does']))
                        donts.error(('\n\n').join(recommendations['donts']))

                    bottom.subheader('Your recommended haircuts:')
                    # compose images in rows for each hair-cut
                    for length, cuts in recommendations['haircut'].items():
                        expander = bottom.expander('##### ' + length.title() + ' length')                        
                        im_width = 285
                        images = [load_resized_image(length, cut, im_width) for cut in cuts]
                        captions = [cut for cut in cuts]
                        expander.image(images, width=im_width, caption=captions)                          
            
            with bottom.popover("__Next features:__ "):
                st.write('''
                         - sun-glasses shape
                         - men haircuts
                         - hair type is considered
                         - apply style to the photo 
                         - color palette recommendation''')

            bottom.markdown('Images were created with [stability-ai](https://replicate.com/stability-ai/sdxl)')                

           

if __name__ == "__main__":
    main()