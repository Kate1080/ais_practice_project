import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def to_rgb( img ):
    return cv.cvtColor( img, cv.COLOR_BGR2RGB )


#test
# img = cv.imread( "src/pic4.jpg" )
# plt.imshow( img )
# plt.show()
#
# img_new = cv.imread( "src/pic4.jpg" )
# img_rgb = to_rgb( img_new )
# plt.imshow( img_rgb )
# plt.show()

