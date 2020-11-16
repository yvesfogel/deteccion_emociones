# deteccion_emociones

En la carpeta video/python hay 3 archivos de Python

# Reconocimiento_emociones_video.py

Este es el archivo principal. Al correrlo te pedirá que selecciones un archivo de video (por ahora debe ser .mp4) y un nombre y path para guardar el nuevo video. 
El programa se correrá y mostrará el resultado de cada frame. El resultado es la detección de información facial de la persona y la emoción detectado.
Cuando el programa acabe se guardará el video en el lugar especificado al inicio. Se recomienda abrir el nuevo video en el programa VLC.

El programa también especificará el brillo promedio de la imagen en cada cuadro.


# Reconocimiento_emociones_webcam.py

Este programa funciona con webcam y la velocidad dependerá del procesamiento de cada ordenador. Va a detectar en tiempo real la imagen de la webcam y sobreponer la emoción detectada en cada cuadro

# duracion_video.py 

Un programa simple que imprime en consola la duración del video especificado. Para sustituirlo simplemente cambiar en la linea 3 "video.mp4" por el archivo del video que se quiere saber. El resultado será una lista con la información [horas, minutos, segundos, cuadros]. El proximo paso es hacer que despliegue de manera más linda esta información.



Para correr este programa se necesita varios requerimientos. Instalar con "pip install <requerimiento>":

Python : 3.6.5
- Tensorflow : 1.10.1
- Keras : 2.2.2
- Numpy : 1.15.4
- Opencv-python : 4.0.0
- cmake
- scipy
- dlib
- imutils
- scikit-learn
- re
- os
