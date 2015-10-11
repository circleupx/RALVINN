########################################################
# ------------------------------------------------------#
#
# Machine Perception and Cognitive Robotics Laboratory
#   Center for Complex Systems and Brain Sciences
#           Florida Atlantic University
#
# ------------------------------------------------------#
########################################################
# ------------------------------------------------------#
#
# Distributed ALVINN, See:
# Pomerleau, Dean A. Alvinn:
# An autonomous land vehicle in a neural network.
# No. AIP-77. Carnegie-Mellon Univ Pittsburgh Pa
# Artificial Intelligence And Psychology Project, 1989.
#
# ------------------------------------------------------#
########################################################

from Shell import *
import numpy as np
from pygame.locals import *
from time import sleep
from datetime import date
from random import choice
from string import ascii_lowercase, ascii_uppercase


class Brain:
    def __init__(self):
        pygame.init()
        pygame.display.init()

        self.quit = False
        self.rover = Shell()
        self.fps = 48  # Camera Frame Rate
        self.windowSize = [740, 280]
        self.imageRect = (0, 0, 320, 280)
        self.displayCaption = "Machine Perception and Cognitive Robotics"

        pygame.display.set_caption(self.displayCaption)

        self.screen = pygame.display.set_mode(self.windowSize, HWSURFACE | DOUBLEBUF | RESIZABLE)
        self.clock = pygame.time.Clock()
        self.run()

    def run(self):
        sleep(1.5)
        while not self.quit:
            self.update_rover_state()
            self.update_image_from_video_feed()
        self.rover.quit = True
        pygame.quit()

    def blit_scale(self, x):
        np.seterr(divide = 'ignore', invalid = 'ignore')
        x -= np.min(x)
        x = x / np.linalg.norm(x)
        x *= 255.0 / x.max()

        return x

    def update_image_from_video_feed(self):

        self.rover.lock.acquire()
        image = self.rover.currentImage
        self.rover.lock.release()

        image = pygame.image.load(cStringIO.StringIO(image), 'tmp.jpg').convert()

        imagearray = pygame.surfarray.array3d(image)
        imagearray = imresize(imagearray, (32, 24))
        image10 = pygame.surfarray.make_surface(imagearray)

        self.screen.blit(image10, (400, 0))
        pygame.display.update((400, 0, 32, 24))

        # in min(7) 7 is the number of neurons you can see.

        for k in range(min(7, self.rover.number_of_neurons)):
            imagew11 = pygame.surfarray.make_surface(
                np.reshape(self.blit_scale(self.rover.network_weight_one[:-1, k]), (32, 24, 3)))
            self.screen.blit(imagew11, (500 + 40 * k, 0))
            pygame.display.update((500 + 40 * k, 0, 32, 24))

        for k in range(min(7, self.rover.number_of_neurons)):
            imagedw11 = pygame.surfarray.make_surface(
                np.reshape(self.blit_scale(self.rover.dw1[:-1, k]), (32, 24, 3)))
            self.screen.blit(imagedw11, (500 + 40 * k, 50))
            pygame.display.update((500 + 40 * k, 50, 32, 24))

        self.screen.blit(image, (0, 0))
        pygame.display.update(self.imageRect)

        self.clock.tick(self.fps)

    def update_rover_state(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.quit = True
            elif event.type == KEYDOWN:
                if event.key in (K_j, K_k, K_SPACE, K_u, K_i, K_o):
                    self.updatePeripherals(event.key)
                elif event.key in (K_w, K_a, K_s, K_d, K_q, K_e, K_z, K_c, K_r, K_l):
                    self.update_wheel_movement(event.key)
                else:
                    pass
            elif event.type == KEYUP:
                if event.key in (K_w, K_a, K_s, K_d, K_q, K_e, K_z, K_c, K_r, K_l):
                    self.update_wheel_movement()
                elif event.key in (K_j, K_k):
                    self.updatePeripherals()
                else:
                    pass
            else:
                pass

    def take_picture(self):
        with open(self.new_picture_name, 'w') as pic:
            self.rover.lock.acquire()
            pic.write(self.rover.currentImage)
            self.rover.lock.release()

    @property
    def new_picture_name(self):
        current_date = str(date.today())
        unique_key = ''.join(choice(ascii_lowercase + ascii_uppercase))
        for _ in range(4):
            return current_date + '_' + unique_key + '.jpq'

    def update_wheel_movement(self, key=None):

        # tread speed ranges from 0 (none) to one (full speed) so [.5 ,.5] would be half full speed
        if key is None:
            self.rover.treads = [0, 0]
        elif key is K_w:
            self.rover.treads = [1, 1]
        elif key is K_s:
            self.rover.treads = [-1, -1]
        elif key is K_a:
            self.rover.treads = [-1, 1]
        elif key is K_d:
            self.rover.treads = [1, -1]
        elif key is K_q:
            self.rover.treads = [.1, 1]
        elif key is K_e:
            self.rover.treads = [1, .1]
        elif key is K_z:
            self.rover.treads = [-.1, -1]
        elif key is K_c:
            self.rover.treads = [-1, -.1]
        elif key is K_l:
            #sio.savemat('rover_brain.mat', {'number_neurons':})
            pass
        elif key is K_r:
            self.rover.treads = self.rover.nn_treads
        else:
            pass

    def updatePeripherals(self, key=None):
        if key is None:
            self.rover.peripherals['camera'] = 0
        elif key is K_j:
            self.rover.peripherals['camera'] = 1
        elif key is K_k:
            self.rover.peripherals['camera'] = -1
        elif key is K_u:
            self.rover.peripherals['stealth'] = not \
                self.rover.peripherals['stealth']
        elif key is K_i:
            self.rover.peripherals['lights'] = not \
                self.rover.peripherals['lights']
        elif key is K_o:
            self.rover.peripherals['detect'] = not \
                self.rover.peripherals['detect']
        elif key is K_SPACE:
            self.take_picture()
        else:
            pass
