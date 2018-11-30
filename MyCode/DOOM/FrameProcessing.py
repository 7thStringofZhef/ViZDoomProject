import cv2

# Takes in image of form 3*height*width
#Outputs image of form 3*newHeight*newWidth
def processImage(image, newHeight=60, newWidth=108):
    if image.shape != (3, newHeight, newWidth):
        image = cv2.resize(
            image.transpose(1,2,0),
            (newWidth, newHeight),
            interpolation=cv2.INTER_AREA
        ).transpose(2,0,1)
    return image

# Processes an image from the game buffer
def processGameBuffer(game):
    buffer = game.screen_buffer
    return processImage(buffer)
