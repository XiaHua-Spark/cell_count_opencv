import cv2
img_rgb = cv2.imread('123.jpg')
img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)

'''
cv2.imshow('image',v)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

_, v_bin = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
v_bin = cv2.dilate(v_bin,None,iterations=2) #膨胀操作

'''
cv2.imshow('image',v_bin)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

_, cnts, h = cv2.findContours(v_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(cnts[1])

cs = sorted(cnts, key=cv2.contourArea, reverse=True)[:6]


result = []

for ind,c in enumerate(cs):
    rect = cv2.minAreaRect(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    length = len(approx)
    x, y, w, h = cv2.boundingRect(c)
    if length==4 and w > h: 
        src_img_rot = img_rgb[y:y+h,x:x+w]
        result.append(src_img_rot)


final = result[0]

cv2.imshow('image',final)
cv2.waitKey(0)
cv2.destroyAllWindows()