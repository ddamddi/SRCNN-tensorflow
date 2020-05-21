import tensorflow as tf
import numpy as np
import cv2


img = cv2.imread('./img/cropped-dog.jpg')
print(img.shape)
downscaled = cv2.resize(img,(256,256))
cv2.imshow("downscaled",downscaled)

for x in range(5):
    img_resized = cv2.resize(downscaled, (512,512), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("bicubic",img_resized)
    # cv2.imshow("ground-truth",img)

    mse = np.mean(np.square(img-img_resized))
    print("<%d>" % x)
    print(mse)
    print(10*np.log10(np.square(255)/mse))
# cv2.imshow("difference",)

cv2.waitKey(0)
cv2.destroyAllWindows()