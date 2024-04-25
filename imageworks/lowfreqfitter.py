import cv2

img = cv2.imread('../result/val_images/steg/steg_1_001.png')
img_blur=cv2.boxFilter(img,-1,(2,2))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imwrite('../result/images/steg_.png',img_blur)
cv2.destroyAllWindows()