import urllib.request
url = "http://192.168.0.101:8080/shot.jpg"
while True:
    imgurl = urllib.request.urlopen(url)
    img = np.array(bytearray(imgurl.read()),dtype = np.uint8)
    img = cv2.imdecode(img,-1)
    cv2.imshow("face detection",facedetector(img))
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()
