# proj-machineLearningRoboticArm
This is a 3 degrees of freedom(DOF) robotic arm developed using Machine Learning. The robotic arm used consists of 6 servo motors. Arduino was used to interface the servos of the arm with the Object Detection and other Machine Learning modules.
The folder named 'Sample_TFLite_mode_dust_new' contains the tflite trained model and label map for the Machine Learning model. 

Also, find the Arduino file for controlling the servos of the robotic arm. This code will come into use when you set up the hardware which is the servo motors of the robotic arm with Arduino and then connect the Arduino controller board with your machine on which this machine learning learning model is running.

The file with the name 'tflite_object_detection_picking' is the model for Webcam Object Detection Using a Tensorflow-trained Classifier. This program uses a TensorFlow Lite object detection model to perform object detection on an image or a folder full of images. It draws boxes and scores around the objects of interest in each image.

Run the above-mentioned 'tflite_object_detection_picking' .py file to start the project and make sure that you have a webcam attached to your machine. The model should detect objects in real-time using the webcam and it should also recognize a 'Coca Cola' can because this model is trained specifically for that. You can train your model or modify this one for other objects etc as you may like.
