import face_recognition
import cv2

from PIL import Image, ImageDraw

nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')




image_of_kasumi= face_recognition.load_image_file('./known/kasumidetect.jpg')
kasumi_face_encoding= face_recognition.face_encodings(image_of_kasumi)[0]


image_of_kenshi= face_recognition.load_image_file('./known/kenshitest.jpg')
kenshi_face_encoding= face_recognition.face_encodings(image_of_kenshi)[0]





#create array of encoding and names

known_face_encoding=[kasumi_face_encoding, kenshi_face_encoding]

known_face_names=["kasumi", "kenshi"]

 # load test image to find faces in

test_image= face_recognition.load_image_file('./unknown/group2.jpg')

gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


face_locations= face_recognition.face_locations(test_image)

face_encodings= face_recognition.face_encodings(test_image, face_locations)

# convert to pil format
pil_image= Image.fromarray(test_image)

#create a image draw instance
draw = ImageDraw.Draw(pil_image)

#loop through faces in test images

for(top, right, bottom, left), face_encoding in zip (face_locations, face_encodings):
    matches= face_recognition.compare_faces(known_face_encoding, face_encoding)

    name = "unknown person"

#if match
    if True in matches:
     first_match_index= matches.index(True)
     name= known_face_names[first_match_index]

     #draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
    nose = nose_cascade.detectMultiScale(gray, 1.3, 5)
    for (e, f, g, i) in nose:
        cv2.rectangle(test_image, (e, f), (e + g, f + i), (0, 0, 255), 3)

    # draw lable
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))

    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    del draw

    #display image
    pil_image.show()









