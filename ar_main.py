import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
import sys
import threading
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not installed.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# --- ПОТОК КАМЕРЫ ---
class HandTrackingThread:
    def __init__(self, cap, hands):
        self.cap = cap
        self.hands = hands
        self.frame = None
        self.hand_landmarks = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
    
    def _process_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret: continue
            
            frame = cv2.flip(frame, 1) 
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            with self.lock:
                self.frame = frame
                if results.multi_hand_landmarks:
                    self.hand_landmarks = results.multi_hand_landmarks[0]
                else:
                    self.hand_landmarks = None
    
    def get_data(self):
        with self.lock:
            return self.frame, self.hand_landmarks
    
    def stop(self):
        self.running = False
        self.thread.join()

# --- ФОН ---
class WebcamBackground:
    def __init__(self):
        self.texture_id = glGenTextures(1)
        self.width = 0
        self.height = 0
        
    def update_texture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.flip(frame_rgb, 0) 
        
        h, w = frame_rgb.shape[:2]
        self.width, self.height = w, h
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.tobytes())

    def draw(self, screen_w, screen_h):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, screen_w, 0, screen_h)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(screen_w, 0)
        glTexCoord2f(1, 1); glVertex2f(screen_w, screen_h)
        glTexCoord2f(0, 1); glVertex2f(0, screen_h)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

# --- СКЕЛЕТ РУКИ ---
class HandLandmarksRenderer:
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    def __init__(self):
        self.interpolated_landmarks = None
        self.interpolation_speed = 0.5
    
    def update_interpolation(self, hand_landmarks):
        if hand_landmarks is None: return
        current = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        if self.interpolated_landmarks is None:
            self.interpolated_landmarks = current
        else:
            self.interpolated_landmarks += (current - self.interpolated_landmarks) * self.interpolation_speed
    
    def draw_landmarks_for_bloom(self, width, height, color=(0.2, 1.0, 0.5)):
        """Рендеринг ландмарков для Bloom эффекта (яркие)"""
        if self.interpolated_landmarks is None: return
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 1, 0, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Яркие линии для Bloom
        glLineWidth(4.0)
        glBegin(GL_LINES)
        glColor4f(color[0] * 1.5, color[1] * 1.5, color[2] * 1.5, 1.0)
        for start, end in self.HAND_CONNECTIONS:
            x1, y1 = self.interpolated_landmarks[start][0], 1 - self.interpolated_landmarks[start][1]
            x2, y2 = self.interpolated_landmarks[end][0], 1 - self.interpolated_landmarks[end][1]
            glVertex2f(x1, y1); glVertex2f(x2, y2)
        glEnd()
        
        # Яркие точки для Bloom
        glPointSize(12.0)
        glBegin(GL_POINTS)
        for idx, lm in enumerate(self.interpolated_landmarks):
            x, y = lm[0], 1 - lm[1]
            if idx == 0: glColor4f(1.5, 0, 0, 1)
            elif idx in [4, 8, 12, 16, 20]: glColor4f(0, 0, 1.5, 1)
            else: glColor4f(color[0] * 1.5, color[1] * 1.5, color[2] * 1.5, 1)
            glVertex2f(x, y)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def draw_landmarks_opengl(self, width, height, color=(0.2, 1.0, 0.5)):
        """Рендеринг обычных ландмарков (без Bloom)"""
        if self.interpolated_landmarks is None: return
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, 1, 0, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor4f(color[0], color[1], color[2], 0.8)
        for start, end in self.HAND_CONNECTIONS:
            x1, y1 = self.interpolated_landmarks[start][0], 1 - self.interpolated_landmarks[start][1]
            x2, y2 = self.interpolated_landmarks[end][0], 1 - self.interpolated_landmarks[end][1]
            glVertex2f(x1, y1); glVertex2f(x2, y2)
        glEnd()
        
        glPointSize(10.0)
        glBegin(GL_POINTS)
        for idx, lm in enumerate(self.interpolated_landmarks):
            x, y = lm[0], 1 - lm[1]
            if idx == 0: glColor4f(1, 0, 0, 1)
            elif idx in [4, 8, 12, 16, 20]: glColor4f(0, 0, 1, 1)
            else: glColor4f(color[0], color[1], color[2], 1)
            glVertex2f(x, y)
        glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

# --- ЧАСТИЦЫ ---
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def update_particles_numba(current_pos, target_pos, speed=0.16):
        return current_pos + (target_pos - current_pos) * speed
else:
    def update_particles_numba(current_pos, target_pos, speed=0.16):
        return current_pos + (target_pos - current_pos) * speed

class ParticleSystem:
    def __init__(self, num_particles=5000):
        self.num_particles = num_particles
        self.current_pos = np.random.uniform(-0.6, 0.6, (num_particles, 3)).astype(np.float32)
        self.target_pos = np.copy(self.current_pos)
        self.colors = np.ones((num_particles, 3), dtype=np.float32)
        self.colors[:, 0], self.colors[:, 1], self.colors[:, 2] = 0.2, 1.0, 0.5
    
    def set_color(self, r, g, b):
        self.colors[:, 0], self.colors[:, 1], self.colors[:, 2] = r, g, b

    # --- ФИГУРЫ ---
    def set_shape_cube(self):
        self.target_pos = np.random.uniform(-0.39, 0.39, (self.num_particles, 3))
    
    def set_shape_icosahedron(self):
        """Икосаэдр (20-гранник, платоново тело)"""
        # Золотое сечение для вершин икосаэдра
        phi = (1 + np.sqrt(5)) / 2
        
        # 12 вершин икосаэдра
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]) * 0.25
        
        # Генерируем точки на поверхности икосаэдра
        points = []
        for _ in range(self.num_particles):
            # Выбираем случайную вершину как центр грани
            v_idx = np.random.randint(0, len(vertices))
            v = vertices[v_idx]
            
            # Добавляем небольшое случайное смещение для заполнения граней
            offset = np.random.uniform(-0.15, 0.15, 3)
            point = v + offset
            
            # Нормализуем для сферической формы
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm * 0.45
            
            points.append(point)
        
        self.target_pos = np.array(points)

    def set_shape_torus(self):
        R, r = 0.39, 0.156
        theta = np.random.uniform(0, 2*np.pi, self.num_particles)
        phi = np.random.uniform(0, 2*np.pi, self.num_particles)
        x = (R + r * np.cos(theta)) * np.cos(phi)
        y = (R + r * np.cos(theta)) * np.sin(phi)
        z = r * np.sin(theta)
        self.target_pos = np.column_stack((x, y, z))

    def set_shape_pyramid(self):
        base = np.random.uniform(0, 1, self.num_particles)
        angle = np.random.uniform(0, 2*np.pi, self.num_particles)
        height = np.random.uniform(0, 0.585, self.num_particles)
        radius = base * (0.585 - height)
        x = radius * np.cos(angle)
        y = height - 0.2925
        z = radius * np.sin(angle)
        self.target_pos = np.column_stack((x, y, z))

    def set_shape_spiral(self):
        turns = 4
        base_t = np.linspace(0, turns*2*np.pi, self.num_particles)
        theta = np.random.uniform(0, 2*np.pi, self.num_particles)
        r_tube = np.random.uniform(0, 0.104, self.num_particles)
        spiral_radius = 0.065 + (base_t / (turns*2*np.pi)) * 0.455
        x = (spiral_radius + r_tube * np.cos(theta)) * np.cos(base_t)
        y = np.linspace(-0.52, 0.52, self.num_particles)
        z = (spiral_radius + r_tube * np.sin(theta)) * np.sin(base_t)
        self.target_pos = np.column_stack((x, y, z))

    def set_shape_heart(self):
        t = np.random.uniform(0, 2*np.pi, self.num_particles)
        u = np.random.uniform(0, np.pi, self.num_particles)
        scale = 0.0195
        x = 16 * np.sin(t)**3 * np.sin(u)
        y = (13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)) * np.sin(u)
        z = 6 * np.cos(u)
        self.target_pos = np.column_stack((x*scale, y*scale + 0.1, z*scale))

    # --- ИСПРАВЛЕННЫЕ ФИГУРЫ (ЧТОБЫ НЕ ВЫЛЕТАЛО) ---
    def set_shape_dna(self):
        """ДНК с объёмными спиралями (несколько частиц в толщине)"""
        num_turns = 80
        tube_radius = 0.065  # Радиус трубки спирали
        helix_radius = 0.325  # Радиус двойной спирали
        height_scale = 0.104
        
        particles_per_helix = self.num_particles // 2
        
        # Первая спираль
        t1 = np.random.uniform(-2*np.pi, 2*np.pi, particles_per_helix)
        # Центральная линия первой спирали
        cx1 = helix_radius * np.cos(t1)
        cz1 = helix_radius * np.sin(t1)
        cy1 = t1 * height_scale
        
        # Добавляем толщину (случайные точки вокруг центральной линии)
        theta1 = np.random.uniform(0, 2*np.pi, particles_per_helix)
        r1 = np.sqrt(np.random.uniform(0, 1, particles_per_helix)) * tube_radius
        
        # Локальная система координат для трубки (перпендикулярно спирали)
        dx1 = r1 * np.cos(theta1) * (-np.sin(t1))
        dz1 = r1 * np.cos(theta1) * np.cos(t1)
        dy1 = r1 * np.sin(theta1)
        
        x1 = cx1 + dx1
        z1 = cz1 + dz1
        y1 = cy1 + dy1
        
        # Вторая спираль (сдвинута на π)
        t2 = np.random.uniform(-2*np.pi, 2*np.pi, self.num_particles - particles_per_helix)
        cx2 = helix_radius * np.cos(t2 + np.pi)
        cz2 = helix_radius * np.sin(t2 + np.pi)
        cy2 = t2 * height_scale
        
        theta2 = np.random.uniform(0, 2*np.pi, self.num_particles - particles_per_helix)
        r2 = np.sqrt(np.random.uniform(0, 1, self.num_particles - particles_per_helix)) * tube_radius
        
        dx2 = r2 * np.cos(theta2) * (-np.sin(t2 + np.pi))
        dz2 = r2 * np.cos(theta2) * np.cos(t2 + np.pi)
        dy2 = r2 * np.sin(theta2)
        
        x2 = cx2 + dx2
        z2 = cz2 + dz2
        y2 = cy2 + dy2
        
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        z = np.concatenate((z1, z2))
        
        self.target_pos = np.column_stack((x, y, z))

    def set_shape_atom(self):
        nucleus_count = int(self.num_particles * 0.2)
        orbit_count = (self.num_particles - nucleus_count) // 3 + 10 
        
        phi = np.random.uniform(0, 2*np.pi, nucleus_count)
        costheta = np.random.uniform(-1, 1, nucleus_count)
        theta = np.arccos(costheta)
        r = 0.104 * np.cbrt(np.random.uniform(0, 1, nucleus_count))
        nx = r * np.sin(theta) * np.cos(phi)
        ny = r * np.sin(theta) * np.sin(phi)
        nz = r * np.cos(theta)
        
        def get_ring(ax, ay, count):
            t = np.random.uniform(0, 2*np.pi, count)
            rx, ry = 0.52 * np.cos(t), 0.52 * np.sin(t)
            rz = np.zeros(count)
            # Rot X
            ry_r = ry * np.cos(ax) - rz * np.sin(ax)
            rz_r = ry * np.sin(ax) + rz * np.cos(ax)
            ry, rz = ry_r, rz_r
            # Rot Y
            rx_r = rx * np.cos(ay) + rz * np.sin(ay)
            rz_r = -rx * np.sin(ay) + rz * np.cos(ay)
            rx, rz = rx_r, rz_r
            return rx, ry, rz

        o1x, o1y, o1z = get_ring(0, 0, orbit_count)
        o2x, o2y, o2z = get_ring(np.pi/3, 0, orbit_count)
        o3x, o3y, o3z = get_ring(-np.pi/3, 0, orbit_count)
        
        x = np.concatenate((nx, o1x, o2x, o3x))
        y = np.concatenate((ny, o1y, o2y, o3y))
        z = np.concatenate((nz, o1z, o2z, o3z))
        self.target_pos = np.column_stack((x, y, z))[:self.num_particles]
    
    def set_shape_dodecahedron(self):
        """Додекаэдр (12-гранник, платоново тело)"""
        phi = (1 + np.sqrt(5)) / 2
        
        # 20 вершин додекаэдра
        vertices = []
        # Куб
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    vertices.append([i, j, k])
        # Прямоугольники с золотым сечением
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([0, i/phi, j*phi])
                vertices.append([i/phi, j*phi, 0])
                vertices.append([j*phi, 0, i/phi])
        
        vertices = np.array(vertices) * 0.22
        
        points = []
        for _ in range(self.num_particles):
            v_idx = np.random.randint(0, len(vertices))
            v = vertices[v_idx]
            offset = np.random.uniform(-0.12, 0.12, 3)
            point = v + offset
            norm = np.linalg.norm(point)
            if norm > 0:
                point = point / norm * 0.42
            points.append(point)
        
        self.target_pos = np.array(points)
    
    def set_shape_mobius(self):
        """Лента Мёбиуса"""
        u = np.random.uniform(0, 2*np.pi, self.num_particles)
        v = np.random.uniform(-0.195, 0.195, self.num_particles)
        radius = 0.455
        x = (radius + v * np.cos(u / 2)) * np.cos(u)
        y = (radius + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_lorenz(self):
        """Аттрактор Лоренца (хаотическая система)"""
        # Параметры системы Лоренца
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        dt = 0.01
        
        # Начальные условия
        x, y, z = 0.1, 0.0, 0.0
        points = []
        
        # Генерируем траекторию
        for _ in range(self.num_particles):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            
            x += dx
            y += dy
            z += dz
            
            points.append([x, y, z])
        
        points = np.array(points)
        # Нормализация и центрирование
        points = (points - np.mean(points, axis=0)) / (np.std(points) * 3.5)
        points *= 0.5
        
        self.target_pos = points
    
    def set_shape_star(self):
        """Пятиконечная звезда 3D"""
        angle = np.random.uniform(0, 2*np.pi, self.num_particles)
        r = np.random.uniform(0, 0.52, self.num_particles)
        r *= (0.65 + 0.65 * np.cos(5 * angle))  # 5 лучей
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = np.random.uniform(-0.13, 0.13, self.num_particles)
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_octahedron(self):
        """Октаэдр (8-гранник)"""
        # Случайные точки на поверхности октаэдра
        u = np.random.uniform(-1, 1, self.num_particles)
        v = np.random.uniform(-1, 1, self.num_particles)
        w = np.random.uniform(-1, 1, self.num_particles)
        
        # Нормализация для октаэдра: |x| + |y| + |z| = 1
        norm = np.abs(u) + np.abs(v) + np.abs(w)
        x = 0.52 * u / norm
        y = 0.52 * v / norm
        z = 0.52 * w / norm
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_trefoil_surface(self):
        """Поверхность трилистника (Trefoil Surface)"""
        u = np.random.uniform(0, 2*np.pi, self.num_particles)
        v = np.random.uniform(0, 2*np.pi, self.num_particles)
        
        # Параметрическая поверхность трилистника
        r = 0.5 + 0.3 * np.cos(1.5 * u)
        
        x = r * np.cos(u) * (1 + 0.2 * np.cos(v))
        y = r * np.sin(u) * (1 + 0.2 * np.cos(v))
        z = 0.3 * np.sin(1.5 * u) + 0.2 * np.sin(v)
        
        # Масштабирование
        scale = 0.45
        x *= scale
        y *= scale
        z *= scale
        
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_hyperboloid(self):
        """Однополостный гиперболоид (башня)"""
        u = np.random.uniform(0, 2*np.pi, self.num_particles)
        v = np.random.uniform(-1, 1, self.num_particles)
        
        # Параметрическое уравнение однополостного гиперболоида
        a, b, c = 0.3, 0.3, 0.5
        
        x = a * np.sqrt(1 + v**2) * np.cos(u)
        y = c * v
        z = b * np.sqrt(1 + v**2) * np.sin(u)
        
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_seashell(self):
        """Раковина (спираль)"""
        u = np.random.uniform(0, 6*np.pi, self.num_particles)
        v = np.random.uniform(0, 2*np.pi, self.num_particles)
        
        # Параметрическая раковина
        a = 0.2
        b = 0.15
        c = 0.1
        
        x = a * (1 - u/(2*np.pi)) * np.cos(u) * (1 + np.cos(v)) + c * np.cos(u)
        y = a * (1 - u/(2*np.pi)) * np.sin(u) * (1 + np.cos(v)) + c * np.sin(u)
        z = b * u/(2*np.pi) + a * (1 - u/(2*np.pi)) * np.sin(v)
        
        # Центрирование и масштабирование
        x = (x - np.mean(x)) * 0.8
        y = (y - np.mean(y)) * 0.8
        z = (z - np.mean(z)) * 0.8
        
        self.target_pos = np.column_stack((x, y, z))
    
    def set_shape_wave(self):
        """Волна (синусоидальная поверхность)"""
        # Создаем сетку
        sqrt_n = int(np.sqrt(self.num_particles))
        x_lin = np.linspace(-0.52, 0.52, sqrt_n)
        z_lin = np.linspace(-0.52, 0.52, sqrt_n)
        
        x_grid, z_grid = np.meshgrid(x_lin, z_lin)
        
        # Волновая функция
        y_grid = 0.2 * np.sin(3 * x_grid) * np.cos(3 * z_grid)
        
        # Преобразуем в одномерные массивы
        x = x_grid.flatten()[:self.num_particles]
        y = y_grid.flatten()[:self.num_particles]
        z = z_grid.flatten()[:self.num_particles]
        
        # Дополняем до нужного размера если нужно
        if len(x) < self.num_particles:
            shortage = self.num_particles - len(x)
            x = np.concatenate([x, np.random.uniform(-0.52, 0.52, shortage)])
            y = np.concatenate([y, np.random.uniform(-0.2, 0.2, shortage)])
            z = np.concatenate([z, np.random.uniform(-0.52, 0.52, shortage)])
        
        self.target_pos = np.column_stack((x, y, z))
    
    def update(self):
        self.current_pos = update_particles_numba(self.current_pos, self.target_pos, 0.16)

    def draw_frame(self):
        frame_size = 0.65
        glLineWidth(3.0)
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_LINE_LOOP)
        glVertex3f(-frame_size, -frame_size, 0)
        glVertex3f(frame_size, -frame_size, 0)
        glVertex3f(frame_size, frame_size, 0)
        glVertex3f(-frame_size, frame_size, 0)
        glEnd()

    def draw(self, bloom_enabled=True, shape_color=(0.2, 1.0, 0.5)):
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glPointSize(3.0)
        
        z_depths = self.current_pos[:, 2]
        brightness = 0.6 + (z_depths + 0.6) / 1.2 * 0.7
        brightness = np.clip(brightness, 0.3, 1.3)
        
        bloom_multiplier = 1.2 if bloom_enabled else 1.0
        shape_color_array = np.tile(shape_color, (self.num_particles, 1)).astype(np.float32)
        colors = shape_color_array * brightness[:, np.newaxis] * bloom_multiplier
        colors_rgba = np.column_stack((colors, np.full(self.num_particles, 0.98)))
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.current_pos.astype(np.float32))
        glColorPointer(4, GL_FLOAT, 0, colors_rgba.astype(np.float32))
        glDrawArrays(GL_POINTS, 0, self.num_particles)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisable(GL_BLEND)

# --- BLOOM ---
class BloomEffect:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.enabled = True
        
        # --- ШЕЙДЕРЫ ---
        vertex_shader = """#version 120
        void main() { gl_TexCoord[0] = gl_MultiTexCoord0; gl_Position = ftransform(); }"""
        
        # Исправили hardcoded делители 800.0/600.0 на uniform resolution, 
        # чтобы размытие было одинаковым на всех экранах
        fragment_shader_template = """#version 120
        uniform sampler2D tex; 
        uniform float blur_size; 
        uniform vec2 dir;
        uniform vec2 resolution; // Добавили разрешение
        
        void main() {
            vec4 sum = vec4(0.0);
            vec2 tc = gl_TexCoord[0].xy;
            
            // Вычисляем размер пикселя для корректного смещения
            vec2 radius = blur_size / resolution; 
            
            sum += texture2D(tex, tc - 4.0*radius*dir) * 0.05;
            sum += texture2D(tex, tc - 3.0*radius*dir) * 0.09;
            sum += texture2D(tex, tc - 2.0*radius*dir) * 0.12;
            sum += texture2D(tex, tc - 1.0*radius*dir) * 0.15;
            sum += texture2D(tex, tc) * 0.16;
            sum += texture2D(tex, tc + 1.0*radius*dir) * 0.15;
            sum += texture2D(tex, tc + 2.0*radius*dir) * 0.12;
            sum += texture2D(tex, tc + 3.0*radius*dir) * 0.09;
            sum += texture2D(tex, tc + 4.0*radius*dir) * 0.05;
            gl_FragColor = sum;
        }"""
        
        try:
            from OpenGL.GL import shaders
            self.program = shaders.compileProgram(
                shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                shaders.compileShader(fragment_shader_template, GL_FRAGMENT_SHADER)
            )
            self.shaders_available = True
        except:
            self.shaders_available = False
            self.enabled = False
            return
            
        self.fbo = glGenFramebuffers(1)
        self.scene_tex = glGenTextures(1)
        self.blur_tex = glGenTextures(1)
        
        for tex in [self.scene_tex, self.blur_tex]:
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def render_bloom(self, draw_scene_func):
        if not self.enabled or not self.shaders_available:
            return

        # 1. Сохраняем текущий Viewport экрана
        viewport_orig = glGetIntegerv(GL_VIEWPORT)
        
        # 2. Рисуем сцену в FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.scene_tex, 0)
        
        # !!! ВАЖНО: Принудительно ставим Viewport под размер текстуры !!!
        glViewport(0, 0, self.width, self.height)
        
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        draw_scene_func() 
        
        # 3. Размываем по горизонтали
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.blur_tex, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, "tex"), 0)
        glUniform1f(glGetUniformLocation(self.program, "blur_size"), 3.0) # Размер размытия
        glUniform2f(glGetUniformLocation(self.program, "resolution"), float(self.width), float(self.height)) # Передаем разрешение
        glUniform2f(glGetUniformLocation(self.program, "dir"), 1.0, 0.0)
        
        self._render_quad(self.scene_tex)
        
        # 4. Размываем по вертикали и рисуем на ЭКРАН
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # !!! ВАЖНО: Восстанавливаем Viewport экрана !!!
        glViewport(viewport_orig[0], viewport_orig[1], viewport_orig[2], viewport_orig[3])
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE) 
        
        glUniform2f(glGetUniformLocation(self.program, "dir"), 0.0, 1.0)
        self._render_quad(self.blur_tex)
        
        glUseProgram(0)
        glDisable(GL_BLEND)

    def _render_quad(self, tex_id):
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, 1, 0, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glColor4f(1,1,1,1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(1, 0)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(0, 1)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        
# --- ТЕКСТ (ИСПРАВЛЕННЫЙ) ---
class TextRenderer:
    def __init__(self):
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 48)

    def render_2d(self, text, x, y, screen_w, screen_h):
        surf = self.font.render(text, True, (200, 200, 200))
        data = pygame.image.tostring(surf, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        
        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, screen_w, 0, screen_h)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1, 1, 1, 1)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x, screen_h - y)
        glTexCoord2f(1, 1); glVertex2f(x + surf.get_width(), screen_h - y)
        glTexCoord2f(1, 0); glVertex2f(x + surf.get_width(), screen_h - y - surf.get_height())
        glTexCoord2f(0, 0); glVertex2f(x, screen_h - y - surf.get_height())
        glEnd()
        
        glDisable(GL_TEXTURE_2D); glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
        glDeleteTextures([tex])

def _render_3d_text_fixed(text, x, y, z, font):
    """Рендеринг 3D текста без переворота и зеркалирования"""
    surf = font.render(text, True, (255, 255, 255))
    # КРИТИЧНО: НЕ переворачиваем изображение (False вместо True)
    data = pygame.image.tostring(surf, "RGBA", False)
    
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surf.get_width(), surf.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    
    glPushMatrix()
    glTranslatef(x, y, z)
    scale = 0.0015
    w, h = surf.get_width() * scale, surf.get_height() * scale
    
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    
    glColor4f(1, 1, 1, 1)
    # Правильные координаты для неперевёрнутой текстуры
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex3f(0, h, 0)  # Левый верх
    glTexCoord2f(1, 0); glVertex3f(w, h, 0)  # Правый верх
    glTexCoord2f(1, 1); glVertex3f(w, 0, 0)  # Правый низ
    glTexCoord2f(0, 1); glVertex3f(0, 0, 0)  # Левый низ
    glEnd()
    
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glDeleteTextures([tex])

# --- ЖЕСТЫ ---
class GestureRecognizer:
    @staticmethod
    def is_fist(lm):
        tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
        return sum(1 for i in range(4) if lm.landmark[tips[i]].y > lm.landmark[pips[i]].y) == 4
    
    @staticmethod
    def is_open_palm(lm):
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        thumb_up = lm.landmark[4].x < lm.landmark[3].x
        fingers_up = sum(1 for i in range(4) if lm.landmark[tips[i]].y < lm.landmark[pips[i]].y) == 4
        return thumb_up and fingers_up

    @staticmethod
    def is_pinching(lm):
        return np.linalg.norm(np.array([lm.landmark[4].x, lm.landmark[4].y]) - np.array([lm.landmark[8].x, lm.landmark[8].y])) < 0.05

    @staticmethod
    def get_hand_center(lm):
        return lm.landmark[9].x, lm.landmark[9].y

# --- MAIN ---
def main():
    try:
        pygame.init()
        WIDTH, HEIGHT = pygame.display.Info().current_w, pygame.display.Info().current_h
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280); cap.set(4, 720)
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        tracker = HandTrackingThread(cap, hands)
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL | FULLSCREEN)
        glEnable(GL_DEPTH_TEST)

        bg = WebcamBackground()
        particles = ParticleSystem(5000)
        hand_renderer = HandLandmarksRenderer()
        gesture = GestureRecognizer()
        text_r = TextRenderer()
        bloom = BloomEffect(WIDTH, HEIGHT)

        shapes = [
            ("Cube", particles.set_shape_cube, (0.2, 1.0, 0.5)),
            ("Icosahedron", particles.set_shape_icosahedron, (0.2, 0.8, 1.0)),
            ("Torus", particles.set_shape_torus, (1.0, 0.2, 0.2)),
            ("Pyramid", particles.set_shape_pyramid, (1.0, 1.0, 0.2)),
            ("Spiral", particles.set_shape_spiral, (0.8, 0.2, 1.0)),
            ("Atom", particles.set_shape_atom, (0.4, 0.4, 1.0)),
            ("DNA", particles.set_shape_dna, (0.2, 1.0, 0.8)),
            ("Heart", particles.set_shape_heart, (1.0, 0.2, 0.5)),
            ("Dodecahedron", particles.set_shape_dodecahedron, (1.0, 0.8, 0.3)),
            ("Mobius", particles.set_shape_mobius, (1.0, 0.6, 0.2)),
            ("Lorenz", particles.set_shape_lorenz, (0.5, 0.3, 1.0)),
            ("Star", particles.set_shape_star, (1.0, 1.0, 0.3)),
            ("Octahedron", particles.set_shape_octahedron, (0.5, 1.0, 0.5)),
            ("Trefoil", particles.set_shape_trefoil_surface, (0.3, 1.0, 0.7)),
            ("Hyperboloid", particles.set_shape_hyperboloid, (1.0, 0.5, 0.2)),
            ("Seashell", particles.set_shape_seashell, (0.9, 0.7, 0.5)),
            ("Wave", particles.set_shape_wave, (0.2, 0.6, 1.0))
        ]
        
        current_shape = 0
        shapes[0][1](); particles.set_color(*shapes[0][2])
        
        rot_x, rot_y = 20.0, 0.0
        target_rot_x, target_rot_y = 20.0, 0.0
        is_dragging, prev_x, prev_y = False, 0, 0
        fist_detected, fist_time = False, 0
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE): running = False
            
            frame, landmarks = tracker.get_data()
            if frame is None: continue
            
            # --- ЛОГИКА ---
            hand_renderer.update_interpolation(landmarks)
            
            if landmarks:
                cx, cy = gesture.get_hand_center(landmarks)
                # Смена фигуры
                if gesture.is_fist(landmarks):
                    if not fist_detected: fist_detected, fist_time = True, time.time()
                elif fist_detected:
                    if gesture.is_open_palm(landmarks) and (time.time() - fist_time < 1.0):
                        current_shape = (current_shape + 1) % len(shapes)
                        shapes[current_shape][1]()
                        particles.set_color(*shapes[current_shape][2])
                        fist_detected = False
                    elif time.time() - fist_time > 1.0: fist_detected = False
                
                # Вращение
                if gesture.is_pinching(landmarks):
                    if not is_dragging: is_dragging, prev_x, prev_y = True, cx, cy
                    else:
                        target_rot_y += (cx - prev_x) * 300
                        target_rot_x += (cy - prev_y) * 300
                        prev_x, prev_y = cx, cy
                else: is_dragging = False
            else: is_dragging = False

            if not is_dragging:
                target_rot_y += 0.5
                target_rot_x = 20.0
            
            rot_x += (target_rot_x - rot_x) * 0.1
            rot_y += (target_rot_y - rot_y) * 0.1
            particles.update()

            # --- ОТРИСОВКА ---
            # 1. Фон
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            bg.update_texture(frame)
            bg.draw(WIDTH, HEIGHT)
            
            # 2. Bloom эффект для ландмарков руки и частиц
            def draw_bloom_elements():
                # Bloom для ландмарков руки
                hand_renderer.draw_landmarks_for_bloom(WIDTH, HEIGHT)
                # Bloom для частиц
                glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluPerspective(45, WIDTH/HEIGHT, 0.1, 50.0)
                glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity(); glTranslatef(-1.0, 0.2, -3.0)
                glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)
                particles.draw(bloom_enabled=True, shape_color=shapes[current_shape][2])
                glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

            bloom.render_bloom(draw_bloom_elements)

            # 3. Четкие ландмарки руки (поверх свечения)
            hand_renderer.draw_landmarks_opengl(WIDTH, HEIGHT)

            # 4. Четкие частицы (поверх свечения)
            glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluPerspective(45, WIDTH/HEIGHT, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity(); glTranslatef(-1.0, 0.2, -3.0)
            glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)
            particles.draw(bloom_enabled=False, shape_color=shapes[current_shape][2])
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

            # 5. UI (Текст) - РИСУЕМ В САМОМ КОНЦЕ, ЧТОБЫ БЫЛ ЧЕТКИМ
            glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluPerspective(45, WIDTH/HEIGHT, 0.1, 50.0)
            glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity(); glTranslatef(-1.0, 0.2, -3.0)
            
            shape_name = shapes[current_shape][0]
            _render_3d_text_fixed(f"> shape :: {shape_name}", -0.5, 0.65, 0, text_r.font_large)
            
            glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)

            # 2D HUD
            text_r.render_2d(f"FPS: {int(clock.get_fps())}", WIDTH-120, 30, WIDTH, HEIGHT)
            
            pygame.display.flip()
            clock.tick(60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Error detected. Press Enter to exit...")
    finally:
        tracker.stop()
        cap.release()
        pygame.quit()

if __name__ == "__main__":
    main()