import cv2

# Inicializar la cámara USB (asegúrate de tenerla conectada al puerto USB)
cap = cv2.VideoCapture(0)

# Inicializar el clasificador HOG para la detección de personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # Capturar el cuadro de la cámara USB
    ret, frame = cap.read()

    # Realizar la detección de personas
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Dibujar un rectángulo alrededor de las personas detectadas
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar la imagen resultante
    cv2.imshow('Person Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
