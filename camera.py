import cv2
import webbrowser
import subprocess
import time


def start_server():
    global server_process
    if server_process is None or server_process.poll() is not None: 
        server_process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2) 

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1 = cv2.GaussianBlur(frame1, (21, 29), 0)

text_x, text_y, text_w, text_h = 10, 5, 100, 30
motion_detected = False  
server_process = None  

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    cv2.putText(frame2, "Chat Bot", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 29), 0)

    roi_frame1 = frame1[text_y:text_y + text_h, text_x:text_x + text_w]
    roi_gray = gray[text_y:text_y + text_h, text_x:text_x + text_w]

    diff = cv2.absdiff(roi_frame1, roi_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x + text_x, y + text_y), (x + text_x + w, y + text_y + h), (120, 255, 0), 2)

            if not motion_detected:
                start_server()  
                webbrowser.open("http://127.0.0.1:5000/")  
                motion_detected = True  

    cv2.imshow("Chuyen Dong", frame2)
    cv2.imshow("Nguong", thresh)

    frame1[text_y:text_y + text_h, text_x:text_x + text_w] = gray[text_y:text_y + text_h, text_x:text_x + text_w]

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if server_process:
    server_process.terminate()  
