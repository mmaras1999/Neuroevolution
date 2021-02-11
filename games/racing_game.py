import pygame
import math
import numpy as np
from pygame.locals import *

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, val):
        return Vector(self.x * val, self.y * val)

    def __truediv__(self, val):
        return Vector(self.x / val, self.y / val)

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def length(self):
        return math.sqrt(self.dot(self))

    def rotate(self, angle):
        return Vector(self.x * math.cos(angle) + self.y * math.sin(angle),
                      self.x * -math.sin(angle) + self.y * math.cos(angle))

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.a = Vector(x1, y1)
        self.b = Vector(x2, y2)
    
    def distanceFromPoint(self, p):
        return abs((self.a - p).cross(self.a - self.b)) / (self.a - self.b).length()

    def getIntersectionScalar(self, other): #return how far from (x1, y1) is intersection in scalar of vector <x2 - x1, y2 - y1>
        myDir = self.a - self.b
        otherDir = other.a - other.b
        if (self.a - other.a).cross(myDir) * (self.a - other.b).cross(myDir) > 0: #check if signs are equal
            return None
        if (other.a - self.a).cross(otherDir) * (other.a - self.b).cross(otherDir) > 0:
            return None
        
        cr = myDir.cross(otherDir)
        if cr == 0: #paraller lines
            return None
        return (self.a - other.a).cross(otherDir) / cr

def VectorFromAngle(angle):
    return Vector(math.cos(angle), -math.sin(angle))

class RacingGame:
    class Car:
        def __init__(self, x, y, game, angle=math.pi / 2):
            self.pos = Vector(x, y)
            self.angle = angle
            self.game = game
            self.velocity = Vector(0, 0)

            self.RAY_LENGTH = 150
            self.RAY_ANGLES = [i * math.pi / 4 for i in range(-2, 3)]
            self.TURN_ANGLE = 12 / 180 * math.pi

            self.distances = np.zeros(len(self.RAY_ANGLES))

            self.calcDistances()

        def calcDistances(self):
            for i in range(len(self.RAY_ANGLES)):
                angle = self.angle + self.RAY_ANGLES[i]
                ex = self.pos.x + self.RAY_LENGTH * math.cos(angle)
                ey = self.pos.y - self.RAY_LENGTH * math.sin(angle)
                ray = Line(self.pos.x, self.pos.y, ex, ey)

                minDistance = 1
                for obstacle in self.game.obstacles:
                    dist = ray.getIntersectionScalar(obstacle)
                    if dist:
                        minDistance = min(dist, minDistance)
                
                self.distances[i]= minDistance

        def update(self, speed, turn):
            if speed == 1 and self.velocity.length() < 15:
                if self.velocity.length() > 8:
                    self.velocity += VectorFromAngle(self.angle) * 0.1
                else:
                    self.velocity += VectorFromAngle(self.angle) * 0.2
            
            if self.velocity.length() > 0.05:
                if speed == -1:
                    self.velocity -= VectorFromAngle(self.angle) * 0.4
            else:
                self.velocity = Vector(0, 0)
            
            # print(self.velocity.length())
            if self.velocity.length() > 0:
                back = self.pos - VectorFromAngle(self.angle) * 8
                front = self.pos + VectorFromAngle(self.angle) * 8

                newBack = back + self.velocity
                newFront = front + self.velocity.rotate(turn * self.TURN_ANGLE)

                newVelocity = (newFront - newBack)
                newAngle = self.angle
                newVelocity = newVelocity / newVelocity.length() * self.velocity.length()
                newAngle = math.atan2(-newVelocity.y, newVelocity.x)

                newVelocity = self.velocity + (newVelocity - self.velocity) * (1 - self.velocity.length() / 17) #drifting

                newVelocity = newVelocity / newVelocity.length() * self.velocity.length()

                self.angle = newAngle
                self.pos = self.pos + newVelocity
                self.velocity = newVelocity

            self.calcDistances()

        def draw(self, surface):
            for i in range(len(self.RAY_ANGLES)):
                ex = self.pos.x + self.RAY_LENGTH * self.distances[i] * math.cos(self.angle + self.RAY_ANGLES[i])
                ey = self.pos.y - self.RAY_LENGTH * self.distances[i] * math.sin(self.angle + self.RAY_ANGLES[i])
                self.game.drawLine(Line(self.pos.x, self.pos.y, ex, ey), (0,0,255), surface)

            pygame.draw.circle(surface, (255,0,0), (self.pos.x, self.pos.y), 10)

            wx = self.pos.x + 7 * math.cos(self.angle)
            wy = self.pos.y - 7 * math.sin(self.angle)
            
            pygame.draw.circle(surface, (0,255,0), (wx, wy), 2)

        def getState(self):
            return list(self.distances) + [self.velocity.length() / 10]
    

    def __init__(self):
        pygame.init()
        pygame.display.set_mode((1500, 1000))
        pygame.display.set_caption('Race!')

    def addObs(self, dx, dy, a=None, checkpoint=None):
        if not a:
            a = self.obstacles[-1].b

        b = a + Vector(dx, dy)
        self.obstacles.append(Line(a.x, a.y, b.x, b.y))
        
        if checkpoint:
            c = b + checkpoint
            self.checkpoints.append(Line(b.x, b.y, c.x, c.y))

    def loadFirstMap(self):
        self.addObs(0, -550, a=Vector(100, 750), checkpoint=Vector(100, 0))
        self.addObs(100, -150)
        self.addObs(100, 0, checkpoint=Vector(0,150))
        self.addObs(100, 150)
        self.addObs(0, 400, checkpoint=Vector(-100, 0))
        self.addObs(50, 50, checkpoint=Vector(0, 100))
        self.addObs(50, -50,checkpoint=Vector(100, 0))
        self.addObs(0, -400, checkpoint=Vector(100, 0))
        self.addObs(100, -150)
        self.addObs(100, 0, checkpoint=Vector(0,150))
        self.addObs(100, 150)
        self.addObs(0, 550, checkpoint=Vector(-100, 0))
        self.addObs(-25, 75)
        self.addObs(-100, 100)
        self.addObs(-75, 25)
        self.addObs(-300, 0, checkpoint=Vector(0, -100))
        self.addObs(-75, -25)
        self.addObs(-100, -100)
        self.addObs(-25, -75)

        self.addObs(0, -550, a=Vector(200, 750))
        self.addObs(50, -50)
        self.addObs(50, 50)
        self.addObs(0, 400)
        self.addObs(100, 150)
        self.addObs(100, 0)
        self.addObs(100, -150)
        self.addObs(0, -400)
        self.addObs(50, -50)
        self.addObs(50, 50)
        self.addObs(0, 550)
        self.addObs(-100, 100)
        self.addObs(-300, 0)
        self.addObs(-100, -100)

        self.car = RacingGame.Car(150, 600, self)

    def loadSecondMap(self):
        self.addObs(0, -400, a=Vector(100, 700), checkpoint=Vector(100, 0))
        self.addObs(25, -75)
        self.addObs(100, -100)
        self.addObs(75, -25)
        self.addObs(800, 25, checkpoint=Vector(0, 50))
        self.addObs(75, 25)
        self.addObs(50, 50)
        self.addObs(25, 75)
        self.addObs(25, 400, checkpoint=Vector(-100, 0))
        self.addObs(-25, 75)
        self.addObs(-100, 100)
        self.addObs(-75, 25)
        self.addObs(-50, 0)

        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)

        self.addObs(-100, 0)
        
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)

        self.addObs(-50, 0, checkpoint=Vector(0, -100))
        self.addObs(-75, -25)
        self.addObs(-75, -75)
        self.addObs(-25, -75)

        self.addObs(0, -400, a=Vector(200, 700))
        self.addObs(100, -100)
        self.addObs(800, -25)        
        self.addObs(40, 20)
        self.addObs(40, 40)
        self.addObs(20, 40)
        self.addObs(-25, 400)
        self.addObs(-100, 100)
        self.addObs(-50, 0)

        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)

        self.addObs(-50, 0)
        
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)
        self.addObs(-50, -50)
        self.addObs(-50, 50)

        self.addObs(-100, 0)
        self.addObs(-75, -75)

        self.car = RacingGame.Car(150, 700, self)

    def loadThirdMap(self):
        self.addObs(-1050, 0, a=Vector(1200, 950), checkpoint=Vector(0, -100))
        self.addObs(-40, -15)
        self.addObs(-40, -40)
        self.addObs(-15, -40)
        self.addObs(0, -40)
        self.addObs(15, -40)
        self.addObs(40, -40)
        self.addObs(40, -15)
        self.addObs(40, 0)
        self.addObs(40, -15)
        self.addObs(40, -40)
        self.addObs(15, -40)
        self.addObs(0, -40)
        self.addObs(15, -40)
        self.addObs(40, -40)
        self.addObs(40, -15)
        self.addObs(40, 0)
        self.addObs(40, -15)
        self.addObs(40, -40, checkpoint=Vector(65, 65))
        self.addObs(15, -40)
        self.addObs(0, -40)
        self.addObs(-15, -40)
        self.addObs(-40, -40)
        self.addObs(-40, -15)
        self.addObs(-40, 0)
        self.addObs(-40, -15)
        self.addObs(-40, -40)
        self.addObs(-15, -40)
        self.addObs(0, -40, checkpoint=Vector(115, 0))
        self.addObs(15, -40)
        self.addObs(40, -40)
        self.addObs(40, -15)
        self.addObs(670, 0, checkpoint=Vector(-75, 75))
        self.addObs(50, 50)
        self.addObs(0, 200)
        self.addObs(-50, 50)
        self.addObs(0, 340)
        self.addObs(50, 50)
        self.addObs(100, 0)
        self.addObs(40, 15)
        self.addObs(40, 40, checkpoint=Vector(-120, 65))
        self.addObs(15, 40)
        self.addObs(0, 40)
        self.addObs(-15, 40)
        self.addObs(-40, 40)
        self.addObs(-40, 15)

        self.addObs(-1000, 0, a=Vector(1150, 850))
        self.addObs(-10, -10)
        self.addObs(0, -20)
        self.addObs(10,-10)
        self.addObs(40, 0)
        self.addObs(80, -25)
        self.addObs(60, -60)
        self.addObs(25, -60)
        self.addObs(0, -20)
        self.addObs(15, -40)
        self.addObs(40, -40)
        self.addObs(40, -15)        
        self.addObs(40, 0)
        self.addObs(40, -15)
        self.addObs(50, -50)
        self.addObs(25, -60)
        self.addObs(0, -100)
        self.addObs(-25, -60)
        self.addObs(-60, -60)
        self.addObs(-60, -25)
        self.addObs(-40, 0)
        self.addObs(-20, -20)
        self.addObs(0, -40)
        self.addObs(20, -20)
        self.addObs(550, 0)
        self.addObs(50, 50)
        self.addObs(0, 50)
        self.addObs(-80, 80)
        self.addObs(0, 450)
        self.addObs(60, 60)
        self.addObs(150, 20)
        self.addObs(10, 10)
        self.addObs(0, 20)
        self.addObs(-10, 10)
        # self.addObs(0, 920)
        

        self.car = RacingGame.Car(1000, 900, self, angle = math.pi)

    def init_game(self, map_id=1):
        self.obstacles = []
        self.checkpoints = []
        if map_id == 1:
            self.loadFirstMap()
        if map_id == 2:
            self.loadSecondMap()
        if map_id == 3:
            self.loadThirdMap()

        self.nextCheckpoint = 0

        self.updateCheckpoint()

        self.checkpointPassed = 0
        self.gameState = 1

    def drawLine(self, line, color, surface):
        pygame.draw.line(surface, color, (line.a.x, line.a.y), (line.b.x, line.b.y), width=2)

    def draw(self, surface):
        surface.fill((0,0,0))
        for i in range(len(self.checkpoints)):
            color = (127, 127, 127)
            if i == self.nextCheckpoint:
                color = (0, 0, 127)
            self.drawLine(self.checkpoints[i], color, surface)

        for obsticle in self.obstacles:
            self.drawLine(obsticle, (255, 255, 255), surface)

        self.car.draw(surface)

    def updateCheckpoint(self):
        current = self.checkpoints[self.nextCheckpoint]
        self.distanceToCheckpoint = current.distanceFromPoint(self.car.pos)
        if self.distanceToCheckpoint < 15:
            # print("new checkpoint")
            self.checkpointPassed += 1
            self.nextCheckpoint += 1
            if self.nextCheckpoint == len(self.checkpoints):
                self.nextCheckpoint = 0
                # print("new lap!")
            
            self.distanceToCheckpoint = current.distanceFromPoint(self.car.pos)

    def checkLost(self):
        for dist in self.car.distances:
            if dist < 0.10:
                self.gameState = 0
                # print("lost :(")
                # self.init_game() #restart
    def make_move(self, vec):
        speed = 0
        if vec[0] < 0.33:
            speed = -1
        if vec[0] > 0.66:
            speed = 1

        turn = 0
        if vec[1] < 0.33:
            turn = -1
        if vec[1] > 0.66:
            turn = 1

        return speed, turn

    def play(self, NN, render=False, wait=None, map_id=1):
        self.init_game(map_id)

        clock = pygame.time.Clock()
        tiks = 0
        while self.gameState == 1 and tiks < 2000:
            tiks += 1
            # for event in pygame.event.get():
            #     if event.type == QUIT:
            #         return

            # speed = 0
            # turn = 0
            # keys=pygame.key.get_pressed()
            # if keys[pygame.K_a]:
            #     turn = 1
            # if keys[pygame.K_d]:
            #     turn = -1
            # if keys[pygame.K_w]:
            #     speed = 1
            # if keys[pygame.K_s]:
            #     speed = -1
            # self.car.update(speed, turn)

            out = NN.eval(self.car.getState())
            # print(out)
            speed, turn = self.make_move(out)

            self.car.update(speed, turn)
            self.checkLost()
            self.updateCheckpoint()

            if render:
                clock.tick(100)
                self.draw(pygame.display.get_surface())
                pygame.display.flip()
        
        return self.checkpointPassed * 1100 +  1100 - self.distanceToCheckpoint

    
    def test(self):
        self.init_game(map_id=3)

        clock = pygame.time.Clock()
        tiks = 0
        while self.gameState == 1 and tiks < 2000:
            tiks += 1
            for event in pygame.event.get():
                if event.type == QUIT:
                    return

            speed = 0
            turn = 0
            keys=pygame.key.get_pressed()
            if keys[pygame.K_a]:
                turn = 1
            if keys[pygame.K_d]:
                turn = -1
            if keys[pygame.K_w]:
                speed = 1
            if keys[pygame.K_s]:
                speed = -1

            self.car.update(speed, turn)
            self.checkLost()
            self.updateCheckpoint()

            clock.tick(30)
            self.draw(pygame.display.get_surface())
            pygame.display.flip()
        
        return self.checkpointPassed * 1100 +  1100 - self.distanceToCheckpoint
                    
                    

if __name__ == "__main__":
    game = RacingGame()
    print(game.test())