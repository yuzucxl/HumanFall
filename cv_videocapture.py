import cv2

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架 640x369 1280×720 1920×1080
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new

source = 'rtsp://admin:q123456789@192.168.1.200/Streaming/Channels/101'

cap = cv2.VideoCapture(source)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    img_new = img_resize(frame)
    cv2.imshow("frame", img_new)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

