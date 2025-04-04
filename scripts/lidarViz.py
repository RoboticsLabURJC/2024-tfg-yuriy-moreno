import carla
import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import signal
import pygame
import os
import cv2


VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

# Variables globales
actor_list = []
manual_mode = False

# Crear un contenedor para los puntos combinados
combined_point_cloud = o3d.geometry.PointCloud()

def lidar_callback(lidar_data, point_cloud, frame):
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))
    
    #labels = data[:, -1].astype(int)  # Etiquetas semánticas
    #colors = np.array([LABELS[label][1] if label in LABELS else (255, 255, 255) for label in labels]) / 255.0  # Normalizar RGB a [0, 1]

    # Reflejar los datos en el eje X
    data[:, 0] = -data[:, 0]

    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])]

    point_cloud.points = o3d.utility.Vector3dVector(data[:, :-1])
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)
    #point_cloud.colors = o3d.utility.Vector3dVector(colors)
    output_dir = 'dataset/lidar'

    # Crear la carpeta de salida si no existeww
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Guardar el point cloud cada ciertos frames (por ejemplo, cada 10 frames)
    if frame % 20 == 0:
        # Crear el nombre de archivo con un timestamp o el número de frame
        filename = os.path.join(output_dir, f"lidar_points_{frame:04d}.ply")
        print(f"Guardando archivo {filename}...")
        o3d.io.write_point_cloud(filename, point_cloud)

def spawn_lidar_camera_segmentation(world, bp, delta):
    # Configuración base
    #lidar_bp.set_attribute('channels', '32')  # Canales típicos para sensores básicos (32 o 64 son comunes).
    #lidar_bp.set_attribute('range', '100')  # Rango en metros, útil para capturar suficiente entorno.
    #lidar_bp.set_attribute('points_per_second', '500000')  # Moderado, más alto mejora detalle pero requiere más recursos.
    #lidar_bp.set_attribute('rotation_frequency', 'str(1 / delta)')  # Frecuencia de rotación en Hz, valores típicos entre 10-20. con este uso 33
    #lidar_bp.set_attribute('upper_fov', '10')  # Campo de visión superior, enfocado pero suficiente.
    #lidar_bp.set_attribute('lower_fov', '-30')  # Campo de visión inferior, amplio para objetos bajos.
    #lidar_bp.set_attribute('noise_stddev', '0.05')  # Nivel de ruido moderado para simular datos reales.

    # Configuración de sensor LiDAR
    lidar_bp = bp.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('points_per_second', '500000')
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta)) #str(1 / delta)
    lidar_bp.set_attribute('upper_fov', '10')  # Incrementar el límite superior
    lidar_bp.set_attribute('lower_fov', '-30')  # Reducir el límite inferior
    lidar_bp.set_attribute('noise_stddev', '0.05')
    ##lidar_bp.set_attribute('semantic_segmentation', 'True')


    lidar_position = carla.Transform(
        carla.Location(x=0,y=-20, z=2.5) #cambiar orientacion
        #carla.Rotation(pitch=0, yaw=90, roll=0)  # Mirar hacia la dirección deseada (90° en yaw)
        )
    lidar = world.spawn_actor(lidar_bp, lidar_position)

    # Configuración de cámara RGB
    camera_bp = bp.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(
        carla.Location(x=0,y=-15, z=5), 
        carla.Rotation(pitch=0, yaw=90, roll=0)   #cambiar orientacion
        )
    camera = world.spawn_actor(camera_bp, camera_transform)

    # Configuración de cámara de segmentación semántica
    segmentation_bp = bp.find('sensor.camera.semantic_segmentation')
    segmentation_bp.set_attribute('image_size_x', '800')
    segmentation_bp.set_attribute('image_size_y', '600')
    segmentation_bp.set_attribute('fov', '90')
    segmentation_transform = carla.Transform(
        carla.Location(x=0,y=-15, z=5), 
        carla.Rotation(pitch=0, yaw=90, roll=0) #cambiar orientacion
        )
    segmentation_camera = world.spawn_actor(segmentation_bp, segmentation_transform)

    return lidar, camera, segmentation_camera


def set_camera_view(viz, third_person):
    ctr = viz.get_view_control()
    if third_person:
        # Configuración de cámara en tercera persona
        ctr.set_zoom(0.06)
        ctr.set_front([1.0, 0.0, 0.3])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([0, 0, 1])
    else:
        # Configuración de cámara en primera persona (ajusta según necesidad)
        ctr.set_zoom(0.3)
        ctr.set_front([0, 0, 1])
        ctr.set_lookat([0, 0, 0])  
        ctr.set_up([-1, 0, 0])

def camera_callback(image, display_surface, frame):
    output_dir = 'dataset/rgb'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convertir la imagen al formato numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa

    # Guardar la imagen como PNG
    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"rgb_{frame:04d}.png")
        print(f"Guardando imagen RGB en {filename}")
        image_to_save = array[:, :, ::-1]  # Convertir de BGRA a RGB
        cv2.imwrite(filename, image_to_save)



    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))
    # Actualizar solo esta superficie en vez de toda la pantalla
    #pygame.display.update(display_surface.get_rect())

###################################Problema
def segmentation_callback(image, display_surface, frame):
    output_dir = 'dataset/segmentation'
    # Crear el directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convertir la imagen de segmentación directamente usando CityScapesPalette
    image.convert(carla.ColorConverter.CityScapesPalette)
    
    # Convertir los datos de la imagen a un array numpy
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))

    # Extraer los canales RGB y convertir de BGRA a RGB
    seg_array = array[:, :, :3]
    seg_array = seg_array[:, :, ::-1]

    # Guardar la imagen como PNG
    if frame % 20 == 0:
        filename = os.path.join(output_dir, f"segmentation_{frame:04d}.png")
        print(f"Guardando imagen de segmentación en {filename}")
        cv2.imwrite(filename, seg_array)

    # Crear la superficie de Pygame y mostrarla en la sección inferior de la pantalla
    surface = pygame.surfarray.make_surface(seg_array.swapaxes(0, 1))
    display_surface.blit(surface, (0, 0))

    # Llamar a la función para mostrar las etiquetas
    display_labels(display_surface)
    
    # Actualizar solo el área de la pantalla donde se muestra la segmentación
    #pygame.display.update(pygame.Rect(0, 600, image.width, image.height))

# Diccionario de etiquetas y colores de segmentación
LABELS = {
    0: ("Desconocido", (0, 0, 0)),
    1: ("Edificio", (70, 70, 70)),
    2: ("Valla", (100, 40, 40)),
    3: ("Otro", (55, 90, 80)),
    4: ("Peatón", (220, 20, 60)),
    5: ("Señalización", (153, 153, 153)),
    6: ("Semáforo", (250, 170, 30)),
    7: ("Vegetación", (107, 142, 35)),
    8: ("Terreno", (152, 251, 152)),
    9: ("Cielo", (70, 130, 180)),
    10: ("Acera", (244, 35, 232)),
    11: ("Carretera", (128, 64, 128)),
    12: ("Barandilla", (190, 153, 153)),
    13: ("Carril-bici", (0, 0, 230)),
    14: ("Coche", (0, 0, 142)),
    15: ("Motocicleta", (0, 0, 70)),
    16: ("Bicicleta", (119, 11, 32)),
    17: ("Tierra", (81, 0, 81))
}

def display_labels(display_surface):
    # Inicializar la fuente para dibujar texto
    font = pygame.font.SysFont("Arial", 18)
    y_offset = 10  # Posición vertical inicial para el texto

    for label_id, (label_name, color) in LABELS.items():
        # Crear una superficie para cada etiqueta
        label_surface = font.render(f"{label_name}", True, color)

        # Dibujar un pequeño rectángulo de color al lado de cada etiqueta
        color_rect = pygame.Rect(10, y_offset, 20, 20)
        pygame.draw.rect(display_surface, color, color_rect)

        # Mostrar la etiqueta junto al rectángulo
        display_surface.blit(label_surface, (40, y_offset))
        y_offset += 30  # Mover hacia abajo para la siguiente etiqueta


def main():
    pygame.init()
    # Configurar Pygame sin usar OpenGL explícitamente
    pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
    width, height = 800, 600 # Aumentamos el tamaño vertical para RGB y segmentación

    # Crear la ventana principal de Pygame y dos subventanas como superficies
    screen = pygame.display.set_mode((width * 2, height))  # Doble ancho para mostrar ambas vistas en paralelo
    pygame.display.set_caption("CARLA - RGB y Segmentación")

    #screen = pygame.display.set_mode((width, height), pygame.SRCALPHA)
    #pygame.display.set_caption("CARLA Vehículo Control")

    rgb_surface = pygame.Surface((width, height))          # Subventana para la cámara RGB
    segmentation_surface = pygame.Surface((width, height)) # Subventana para la segmentación

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    settings = world.get_settings()
    delta = 0.03
    settings.fixed_delta_seconds = delta
    settings.synchronous_mode = True
    world.apply_settings(settings)

    #Clima
    weather = world.get_weather()

    # Ajustar el ángulo azimutal y la altura del sol
    weather.sun_azimuth_angle = 90  # Sol en dirección este (por ejemplo)
    weather.sun_altitude_angle = 60  # Sol alto en el cielo (ángulo positivo)

    # Aplicar la configuración
    world.set_weather(weather)


    global actor_list, third_person_view
    lidar, camera, segmentation_camera = spawn_lidar_camera_segmentation(world, blueprint_library, delta)
    actor_list.append(lidar)
    actor_list.append(camera)
    actor_list.append(segmentation_camera)

    # Llamada al callback de cámara RGB
    camera.listen(lambda image: camera_callback(image, rgb_surface, frame))

    # Llamada al callback de segmentación
    segmentation_camera.listen(lambda image: segmentation_callback(image, segmentation_surface, frame))

    point_cloud = o3d.geometry.PointCloud()

    frame = 0 # Contador de frames
    lidar.listen(lambda data: lidar_callback(data, point_cloud, frame))

    # Utilizar VisualizerWithKeyCallback
    viz = o3d.visualization.VisualizerWithKeyCallback()
    viz.create_window(window_name='Lidar simulado en Carla', width=960, height=540, left=480, top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    third_person_view = True

    # Añadir el point cloud al visualizador en la primera iteración
    lidar_data_received = False

    # Configurar el callback de tecla 'v'
    def toggle_camera_view(_):
        global third_person_view
        third_person_view = not third_person_view
        set_camera_view(viz, third_person_view)
        print("Cambiando a tercera persona" if third_person_view else "Cambiando a primera persona")
        return True  # Devolver True para continuar el evento de renderizado

    # Asignar el callback al visualizador
    viz.register_key_callback(ord("V"), toggle_camera_view)

    dt0 = datetime.now()


    while True:

        # Dibujar ambas subventanas en la ventana principal
        screen.blit(rgb_surface, (0, 0))                    # Poner RGB a la izquierda
        screen.blit(segmentation_surface, (width, 0))       # Poner Segmentación a la derecha
        pygame.display.flip()

        if frame == 5 and not lidar_data_received:
            viz.add_geometry(point_cloud)
            lidar_data_received = True
            print("Geometry added to the visualizer")
            set_camera_view(viz, third_person_view)

        viz.update_geometry(point_cloud)
        viz.poll_events()
        viz.update_renderer()
        time.sleep(0.03)
        world.tick()

        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            fps = 1.0 / process_time.total_seconds()
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        dt0 = datetime.now()
        frame += 1

        if not viz.poll_events():
            print("Exiting visualization")
            break

    cleanup()

def cleanup():
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
