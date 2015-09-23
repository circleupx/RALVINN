
                          Rover Autonomous Land Vehicle in a Neural Network
-------------------------------------------------------

                        Machine Perception and Cognitive Robotics Laboratory
                            Center for Complex Systems and Brain Sciences
                                    Florida Atlantic University
-------------------------------------------------------

  Distributed ALVINN, See:
  Pomerleau, Dean A. Alvinn:
  An autonomous land vehicle in a neural network.
  No. AIP-77. Carnegie-Mellon Univ Pittsburgh Pa
  Artificial Intelligence And Psychology Project, 1989.

-------------------------------------------------------
                                          Brookstone Rover 2.0
-------------------------------------------------------
    1) To use this code you will need a Brookstone Rover 2.0
    2) Download the Source code into a local directory
    3) Open the code in your favorite python IDE i.e Pycharm or Spyder
    4) Turn on the Rover 2.0 and connect to the signal created by the rover. 
       It will have a name like ROVER_ID
    5) Run MPCR_RALVINN.py
    6) If done correctly you will see a live feed from the Rover 2.0 camera.
    7) Control the Rover 2.0 using:
        W - Forward
        S - Back
        D - Right
        A - LEFT
        U - Toggle Infrared Night Vision.
        J - Camera Up
        K - Camera Down
        I - Toggle Lights
        R - Neural Network
        Space bar - Take a pic
    8) Enjoy, have fun and help us by improving the code.


-------------------------------------------------------
                                            Notes
-------------------------------------------------------
    1) The Neural Network is set to 50 neurons
    2) The current frame rate is 10 FPS, max is 48
    3) Each tread is set to full speed [1,1]
    4) All picutes taken are saved under folder /RALVINN.
    5) Rover 2.0 has a mic and speakers.
