# module_name, package_name, ClassName, method_name, 
# ExceptionName, function_name, GLOBAL_CONSTANT_NAME, 
# global_var_name, instance_var_name, function_parameter_name, local_var_name.

import streamlit as st
import numpy as np
import json
import keras
from glob import glob
from random import choice
from mtcnn import MTCNN
import tensorflow as tf
import cv2
from google_images_search import GoogleImagesSearch
from google_images_search.fetch_resize_save import GSImage

# make a wide layout, not with a fixed width in the center 
st.set_page_config(
    page_title="Get haircut recommendations",
    layout="wide",
)

# Page layout

st.title('Get haircut recommendations')

top = st.container()
border_left, left_column, right_column, border_right = st.columns([1, 2, 3, 1])
bottom = st.container()


def preprocess_image(image, img_size = (150, 150)):
    """detect face, crop and resize it, pack it to a batch

    Args:
        image (ndarray): image astype('float32')
        img_size (tuple, optional): target image size for nn model. Defaults to (150, 150).

    Returns:
        tf.Tensor: batched image tensor
    """
    detector = MTCNN()
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

@st.cache_resource
def load_nn_model():
    """load nn model and cache it, so it is loaded only once

    Returns:
        Keras model: faceShape classification model
    """
    return keras.saving.load_model("face_shape/models_weights/fine_tune_block6_aug.keras")


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
def gis(query, num=2):
    """search for images with custom image google search API

    Args:
        query (str): hair cut recommendation text prompt
        num (int, optional): number of images. Defaults to 2.

    Returns:
        list: list of images with urls and ref_urls
    """
    @st.cache_resource
    def load_gis():
        """load custom google search API only once

        Returns:
            GoogleImagesSearch: google search object
        """        
        API_KEY = st.secrets["GoogleAPI_key"]
        SE_KEY = st.secrets["SE_key"]
        return GoogleImagesSearch(API_KEY, SE_KEY)
    
    gis = load_gis()
    search_params = {
        'q': query,
        'num': num,
        # 'fileType': 'jpg|gif|png',
        # 'rights': 'cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived', #cc_publicdomain|cc_attribute|cc_sharealike|cc_noncommercial|cc_nonderived
        # 'safe': 'active', ##
        # 'imgSize': 'small', ##
        'imgColorType': 'color' ##
    }
    # gis.search(search_params=search_params)
    # return gis.results()

    # for test purposes, so not to exceed API limits
    test_images = []
    for _ in range(num):
        test_image = GSImage(gis)
        test_image.url = choice(glob(f'face_shape/shapeofyou-2/test/**/*.jpg'))
        test_image.referrer_url = 'nfnfbfb'
        test_images.append(test_image)
    return test_images


def main():
    # wait before nn model is loaded
    # only after load everything else
    model = load_nn_model()
    with top:
        if top.button("Get Recommendations", type="primary"):

            recommendations = None
            random_file = None

            with st.spinner('Your faceshape is analysed...'):
                random_file = choice(glob(f'face_shape/shapeofyou-2/test/**/*.jpg'))
                face_img = cv2.cvtColor(cv2.imread(random_file), cv2.COLOR_BGR2RGB)
                left_column.image(face_img, caption='For test purposes:  '+random_file.split('/')[-2])
                recommendations = recommend(model, face_img)

            if recommendations is not None:

                # format recommendations in the botom section
                top.subheader(f"Congratulations! You have a {recommendations['faceShape']} shape!", divider='rainbow')
                does = '#### Do\'s\n\n'+('\n\n').join(recommendations['does'])
                right_column.success(does)
                donts = '#### Don\'ts\n\n'+('\n\n').join(recommendations['donts'])
                right_column.error(donts)
                right_column.info('#### Your recommended haircuts :arrow_down:')

                # compose google images in rows for each hair-cut
                for length, cuts in recommendations['haircut'].items():
                    bottom.divider()
                    bottom.subheader(length.title() + ' length')
                    for cut in cuts:
                        num_of_images = 5
                        hair_cut_images = gis(cut, num_of_images)
                        image_columns = bottom.columns(num_of_images+1)
                        bottom.write(
                            """<style>
                            [data-testid="stHorizontalBlock"] {
                                align-items: center;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )
                        image_columns[0].markdown('##### '+cut)
                        for hair_cut,column in zip(hair_cut_images, image_columns[1:]):
                            column.image(hair_cut.url, use_column_width="always")
                            column.caption('[source]('+ hair_cut.referrer_url +')')
           

if __name__ == "__main__":
    main()