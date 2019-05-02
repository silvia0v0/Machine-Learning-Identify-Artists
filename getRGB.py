import cv2

img = cv2.imread('./resized/resized/cropped/Alfred_Sisley_23_4.jpg')
b,g,r = cv2.split(img)
print(b)
print(g)
print(r)
red_test = cv2.imwrite('./resized/resized/AS234red.jpg',r)
blue_test = cv2.imwrite('./resized/resized/AS234blue.jpg',b)
green_test = cv2.imwrite('./resized/resized/AS234green.jpg',g)

