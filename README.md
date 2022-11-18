# proj-machineLearningRoboticArm
This is a 3 degrees of freedom(DOF) robotic arm developed using Machine Learning. Robotic arm used is servo motor based. Arduino was used to interface the servos of the arm with the Object Detection and various Machine Learning modules.

The file with name 'tflite_object_detection_picking' is the model for Webcam Object Detection Using Tensorflow-trained Classifier. This program uses a TensorFlow Lite object detection model to perform object detection on an image or a folder full of images. It draws boxes and scores around the objects of interest in each image.

Download the Folder as .zip and simply run the above mentioned 'tflite_object_detection_picking' .py file to start the project and make sure that you have a webcam attached to your machine.

The model should detect objects on real-time using the webcam and it should also recognize a 'Coca Cola' can because this model is trained specifically for that. You can train your own model or can modify this one for other objects etc as you may like.
