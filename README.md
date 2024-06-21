# [getyourlook.streamlit.app](https://getyourlook.streamlit.app/)

Here is a Minimal Viable Product - a simple WebApp where user can take/upload photo and receive haircut recommendations.
![app](/hair_cut/images/haircutrec.png)

## Architecture

![app architecture](/hair_cut/images/architecture.png)
*diagram was made with [excalidraw.com](https://excalidraw.com/)*

All models and files are cached to save resources and for the app to have quick response. I used a Multi-task CNN[^1] as a face detector and a [face shape classificator](/face_shape/README.md) to get the shape of the face. 

## Future improvements

__Apply style to the photo__

As it is a MVP, the recommeded haircuts are in the form of images manually generated with [stability-ai](https://replicate.com/stability-ai/sdxl). To make WebApp more user-oriented images should be generated from the input image and recommendation prompts, so the user can see not some random images, but all recommended haircuts applied to his/her image.

__Men haircuts__

Right now only female haircuts are shown and the model was trained only on female celebrities. The model should be retrained also with male faces, and haircut recommendations should include more variety of haircuts.

__Hair type__

It would be cool to take into account hair type, because if the user has curly hair, why recommend a haircut with straight hair, also some haircuts look differently depending on the thickness of hair.

__Sun-glasses shape__

Face shape can help not only with a haircut, but also with the shape of sun-glasses or how to apply a make-up.

__Color palette__

Ideally the face also is analysed for skin/eyes/hair color. The corresponding color palette is recommended based on 'spring', 'summer', 'autumn' or 'winter' types. For this task additional model should be trained or current model should be changed to the MTCNN.

__Camera calibration and distortion__

It seems that shape classification also depends on camera distortion. As all users have different cameras it is impossible to apply some random hardcoded undistortion. Maybe camera information can be retrived from the image and matrices for undistortion can be generated.

[^1]: [(Implementation)](https://github.com/ipazc/mtcnn) Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Letters, 23(10):1499–1503.