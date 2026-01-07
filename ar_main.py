import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import sys


# ============================================================================
# КЛАСС ДЛЯ РАБОТЫ С ФОНОМ (ВИДЕО С КАМЕРЫ)
# ============================================================================
class WebcamBackground:
    def __init__(self):
        self.texture_id = glGenTextures(1)
        
    def update_texture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0)
        
        h, w = frame_rgb.shape[:2]
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.tobytes())

    def draw(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, width, 0, height)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(width, 0)
        glTexCoord2f(1, 1); glVertex2f(width, height)
        glTexCoord2f(0, 1); glVertex2f(0, height)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


class HandLandmarksRenderer:
    def draw_landmarks_2d(self, frame, hand_landmarks, width, height, shape_color=(0, 255, 0)):
        h, w = frame.shape[:2]
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            x, y = int(landmark.x * w), int(landmark.y * h)
            
            if idx == 0:
                color = (255, 0, 0)
                radius = 8
            elif idx in [4, 8, 12, 16, 20]:
                color = (0, 0, 255)
                radius = 6
            else:
                color = (0, 255, 0)
                radius = 4
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius + 2, (255, 255, 255), 1)


class ParticleSystem:
    def __init__(self, num_particles=500):
        self.num_particles = num_particles
        self.current_pos = np.random.uniform(-0.6, 0.6, (num_particles, 3))
        self.target_pos = np.copy(self.current_pos)
        self.colors = np.ones((num_particles, 3))
        self.colors[:, 0] = 0.2
        self.colors[:, 1] = 1.0
        self.colors[:, 2] = 0.5
    
    def set_color(self, r, g, b):
        self.colors[:, 0] = r
        self.colors[:, 1] = g
        self.colors[:, 2] = b

    def set_shape_cube(self):
        self.target_pos = np.random.uniform(-0.3, 0.3, (self.num_particles, 3))

    def set_shape_sphere(self):
        phi = np.random.uniform(0, 2 * np.pi, self.num_particles)
        costheta = np.random.uniform(-1, 1, self.num_particles)
        u = np.random.uniform(0, 1, self.num_particles)
        theta = np.arccos(costheta)
        r = 0.35 * np.cbrt(u)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_torus(self):
        R, r = 0.3, 0.12
        theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
        phi = np.random.uniform(0, 2 * np.pi, self.num_particles)
        
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_pyramid(self):
        base = np.random.uniform(0, 1, self.num_particles)
        angle = np.random.uniform(0, 2 * np.pi, self.num_particles)
        height = np.random.uniform(0, 0.45, self.num_particles)
        
        radius = base * (0.45 - height)
        x = radius * np.cos(angle)
        y = height - 0.225
        z = radius * np.sin(angle)
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_spiral(self):
        turns = 4
        tube_radius = 0.08
        
        base_t = np.linspace(0, turns * 2 * np.pi, self.num_particles)
        theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
        r_tube = np.random.uniform(0, tube_radius, self.num_particles)
        
        spiral_radius = 0.05 + (base_t / (turns * 2 * np.pi)) * 0.35
        
        x = (spiral_radius + r_tube * np.cos(theta)) * np.cos(base_t)
        y = np.linspace(-0.4, 0.4, self.num_particles)
        z = (spiral_radius + r_tube * np.sin(theta)) * np.sin(base_t)
        
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_cylinder(self):
        theta = np.random.uniform(0, 2 * np.pi, self.num_particles)
        y = np.random.uniform(-0.35, 0.35, self.num_particles)
        r = np.sqrt(np.random.uniform(0, 1, self.num_particles)) * 0.25
        
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        
        self.target_pos = np.column_stack((x, y, z))

    def update(self):
        self.current_pos += (self.target_pos - self.current_pos) * 0.16

    def draw_axes(self):
        axis_length = 0.4
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor4f(1.0, 1.0, 1.0, 0.8)
        glVertex3f(-axis_length, 0, 0)
        glVertex3f(axis_length, 0, 0)
        glVertex3f(0, -axis_length, 0)
        glVertex3f(0, axis_length, 0)
        glVertex3f(0, 0, -axis_length)
        glVertex3f(0, 0, axis_length)
        glEnd()

    def draw(self):
        self.draw_axes()
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPointSize(2.0)
        
        z_depths = self.current_pos[:, 2]
        brightness = 0.6 + (z_depths + 0.6) / 1.2 * 0.7
        brightness = np.clip(brightness, 0.3, 1.3)
        
        colors_with_brightness = self.colors * brightness[:, np.newaxis]
        colors_rgba = np.column_stack((colors_with_brightness, np.full(self.num_particles, 0.98)))
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.current_pos.astype(np.float32))
        glColorPointer(4, GL_FLOAT, 0, colors_rgba.astype(np.float32))
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        glDisable(GL_BLEND)


class GestureRecognizer:
    @staticmethod
    def is_fist(hand_landmarks):
        """Проверка на кулак - все пальцы согнуты"""
        lm = hand_landmarks.landmark
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        folded_count = sum(1 for i in range(4) if lm[tips[i]].y > lm[pips[i]].y)
        return folded_count == 4
    
    @staticmethod
    def is_open_palm(hand_landmarks):
        """Проверка на открытую ладонь - все 5 пальцев выпрямлены"""
        lm = hand_landmarks.landmark
        
        # Большой палец выпрямлен
        thumb_up = lm[4].x < lm[3].x
        
        # Остальные 4 пальца выпрямлены
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        fingers_up = sum(1 for i in range(4) if lm[tips[i]].y < lm[pips[i]].y)
        
        return thumb_up and fingers_up == 4
    
    @staticmethod
    def is_rotation_gesture(hand_landmarks):
        """
        Жест для вращения: большой и указательный пальцы выпрямлены,
        остальные согнуты (жест "пистолет")
        """
        lm = hand_landmarks.landmark
        
        # Указательный палец выпрямлен
        index_up = lm[8].y < lm[6].y
        
        # Средний, безымянный и мизинец согнуты
        middle_down = lm[12].y > lm[10].y
        ring_down = lm[16].y > lm[14].y
        pinky_down = lm[20].y > lm[18].y
        
        return index_up and middle_down and ring_down and pinky_down
    
    @staticmethod
    def get_rotation_from_two_fingers(hand_landmarks):
        """
        Вычисляет углы вращения на основе положения большого и указательного пальцев.
        Использует угол между пальцами для rot_y и расстояние для rot_x
        """
        lm = hand_landmarks.landmark
        
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        
        finger_vector = index_tip - thumb_tip
        
        angle = np.arctan2(finger_vector[1], finger_vector[0])
        rot_y = np.degrees(angle) * 2
        
        distance = np.linalg.norm(finger_vector)
        rot_x = (distance - 0.1) * 400
        rot_x = np.clip(rot_x, -90, 90)
        
        return rot_x, rot_y


def main():
    pygame.init()
    info = pygame.display.Info()
    WIDTH, HEIGHT = info.current_w, info.current_h
    FPS = 60
    
    CAM_WIDTH, CAM_HEIGHT = 1280, 720
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Camera not found")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    clock = pygame.time.Clock()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | FULLSCREEN)
    pygame.display.set_caption("AR Gesture Control")
    
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    bg = WebcamBackground()
    particles = ParticleSystem(num_particles=5000)
    hand_renderer = HandLandmarksRenderer()
    gesture_recognizer = GestureRecognizer()
    
    shapes = [
        ("Cube", particles.set_shape_cube, (0.2, 1.0, 0.5)),
        ("Sphere", particles.set_shape_sphere, (0.2, 0.5, 1.0)),
        ("Torus", particles.set_shape_torus, (1.0, 0.2, 0.2)),
        ("Pyramid", particles.set_shape_pyramid, (1.0, 1.0, 0.2)),
        ("Cylinder", particles.set_shape_cylinder, (0.2, 1.0, 1.0)),
        ("Spiral", particles.set_shape_spiral, (0.8, 0.2, 1.0))
    ]
    current_shape_index = 0
    shapes[0][1]()
    particles.set_color(*shapes[0][2])
    
    rot_x, rot_y = 20.0, 0.0
    target_rot_x, target_rot_y = 20.0, 0.0
    auto_rotate = True
    auto_rot_speed = 0.5
    
    fist_detected = False
    fist_time = 0
    cooldown_duration = 3.0
    
    frame_count = 0
    process_every_n_frames = 2
    last_hand_landmarks = None
    
    print("AR Gesture Control")
    print("Fist + Palm = Change shape | Point = Rotate | ESC = Exit")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        
        frame_count += 1
        if frame_count % process_every_n_frames == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                last_hand_landmarks = results.multi_hand_landmarks[0]
        
        if last_hand_landmarks:
            current_shape_color = shapes[current_shape_index][2]
            hand_renderer.draw_landmarks_2d(frame, last_hand_landmarks, CAM_WIDTH, CAM_HEIGHT, current_shape_color)
            
            current_time = time.time()
            
            if gesture_recognizer.is_fist(last_hand_landmarks):
                if not fist_detected:
                    fist_detected = True
                    fist_time = current_time
                    print("Fist detected - open palm to change")
            
            elif fist_detected:
                time_since_fist = current_time - fist_time
                
                if gesture_recognizer.is_open_palm(last_hand_landmarks) and time_since_fist <= cooldown_duration:
                    current_shape_index = (current_shape_index + 1) % len(shapes)
                    shape_name, shape_func, color = shapes[current_shape_index]
                    shape_func()
                    particles.set_color(*color)
                    print(f"Changed to: {shape_name}")
                    fist_detected = False
                
                elif time_since_fist > cooldown_duration:
                    fist_detected = False
            
            if gesture_recognizer.is_rotation_gesture(last_hand_landmarks):
                target_rot_x, target_rot_y = gesture_recognizer.get_rotation_from_two_fingers(last_hand_landmarks)
                auto_rotate = False
            else:
                auto_rotate = True
        else:
            auto_rotate = True
            fist_detected = False
        
        if auto_rotate:
            target_rot_y = rot_y + auto_rot_speed
            target_rot_x = 20.0
        
        rot_x += (target_rot_x - rot_x) * 0.15
        rot_y += (target_rot_y - rot_y) * 0.15
        
        particles.update()
        
        shape_name = shapes[current_shape_index][0]
        cv2.putText(frame, f"Shape: {shape_name}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Fist + Open Palm: Change", (20, CAM_HEIGHT - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Index Finger: Rotate", (20, CAM_HEIGHT - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        bg.update_texture(frame)
        bg.draw(WIDTH, HEIGHT)
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45, WIDTH / HEIGHT, 0.1, 50.0)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glTranslatef(-1.5, 0.5, -3.0)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        particles.draw()
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    hands.close()
    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()