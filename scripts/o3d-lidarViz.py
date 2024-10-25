import carla
import numpy as np
import open3d as o3d
import time
import random
from datetime import datetime
from matplotlib import colormaps as cm
import sys
import signal
import pygame  # Importar pygame para detectar teclas

VIRIDIS = np.array(cm.get_cmap('inferno').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def lidar_callback(lidar_data, point_cloud):
    '''
    Procesa los datos brutos de carla 
    cada vez que se toma una muestra (callback) 
    y actualiza la nube de puntos en el objeto PointCloud 
    '''
    data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    intensity = data[:, -1]
    int_color = np.c_[
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2])]

    # Convertir la nube de puntos al formato de Open3D
    point_cloud.points = o3d.utility.Vector3dVector(data[:, :-1])
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)

def spawn_vehicle_lidar_camera(world, bp, traffic_manager, delta):
    vehicle_bp = bp.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()

    # Elegir un punto de spawn aleatorio
    spawn_point = random.choice(spawn_points)

    # Crear el vehículo en el punto aleatorio
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = bp.find('sensor.lidar.ray_cast')

    lidar_bp.set_attribute('range', '100')
    lidar_bp.set_attribute('rotation_frequency', str(1 / delta))
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '500000')

    lidar_position = carla.Transform(carla.Location(x=-0.5, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_position, attach_to=vehicle)

    # Configuración de la cámara
    camera_bp = bp.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    # Posicionar la cámara en el frente del vehículo
    camera_transform = carla.Transform(carla.Location(x=-4.0,z=2.5))  # Aprox en el techo del coche
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)


    # Conectar el vehículo al Traffic Manager y activar el autopiloto
    vehicle.set_autopilot(True, traffic_manager.get_port())

    return vehicle, lidar, camera

def add_open3d_axis(vis):
    """
    Añade un pequeño 3D axis en el Open3D Visualizer
    """
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

# Variables globales para almacenar actores
actor_list = []

manual_mode = False  # Variable para alternar entre automático y manual

def cleanup():
    """
    Elimina todos los actores de Carla (vehículos, sensores, etc.)
    """
    global actor_list
    print("\nLimpiando actores...")
    for actor in actor_list:
        if actor is not None:
            actor.destroy()
    actor_list = []
    print("Actores eliminados.")

def signal_handler(sig, frame):
    """
    Captura la señal de interrupción (Ctrl+C) y limpia los actores antes de salir.
    """
    print("\nInterrupción recibida. Finalizando...")
    cleanup()
    sys.exit(0)

def vehicle_control(vehicle):
    
    # Función para gestionar los controles del vehículo en modo manual.
    # Detecta las teclas WASD para controlar el coche y cambia entre manual y automático con la tecla R.

    global manual_mode  # Acceder a la variable global para cambiar el modo

    # Procesar eventos de Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cleanup()  # Limpiar los actores si se cierra la ventana
            pygame.quit()
            sys.exit()


        # Alternar entre modo automático y manual con la tecla R
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            manual_mode = not manual_mode
            vehicle.set_autopilot(not manual_mode)
            if manual_mode:
              print("\nCoche en modo manual.")
            else:
               print("\nCoche en modo automático.")
            time.sleep(0.3)  # Evitar múltiples activaciones por pulsación rápida

        # Si estamos en modo manual, controlar el coche con WASD
        if manual_mode:
            control = carla.VehicleControl()
            # Capturar eventos de teclado
            keys = pygame.key.get_pressed()

            # Controlar el vehículo con WASD
            if keys[pygame.K_w]:
                control.throttle = 1  # Acelerar
            if keys[pygame.K_s]:
                control.brake = 1  # Frenar
            if keys[pygame.K_a]:
                control.steer = -0.3  # Girar a la izquierda
            if keys[pygame.K_d]:
                control.steer = 0.3  # Girar a la derecha

            vehicle.apply_control(control)  # Aplicar los controles al vehículo

def camera_callback(image, display_surface):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB

    # Convertir la imagen a formato pygame
    surface = pygame.surfarray.make_surface(array[:, :, :3].swapaxes(0, 1))

    # Mostrar la imagen en la ventana de pygame
    display_surface.blit(surface, (0, 0))
    

def main():
    '''
    Funcion Main del programa
    '''
    # Inicializar Pygame para capturar las entradas del teclado
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA Vehículo Control")

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Tiempo límite para conectarse

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Configuración del Traffic Manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)  # Asegurar que el Traffic Manager esté sincronizado
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # Distancia de seguridad entre vehículos

    settings = world.get_settings()
    delta = 0.05

    settings.fixed_delta_seconds = delta
    settings.synchronous_mode = True  # Activar modo sincronizado para modo sin renderizar
    world.apply_settings(settings)

    # Global para evitar que los actores se eliminen automáticamente
    global actor_list
    vehicle, lidar, camera = spawn_vehicle_lidar_camera(world, blueprint_library, traffic_manager, delta)
    actor_list.append(vehicle)
    actor_list.append(lidar)
    actor_list.append(camera)

    # Llamar al callback de la cámara para actualizar Pygame con las imágenes
    camera.listen(lambda image: camera_callback(image, screen))
    

    #point_cloud = o3d.geometry.PointCloud()

    #lidar.listen(lambda data: lidar_callback(data, point_cloud))
    '''
    viz = o3d.visualization.Visualizer()
    viz.create_window(
            window_name='Lidar simulado en Carla',
            width=960,
            height=540,
            left=480,
            top=270)
    viz.get_render_option().background_color = [0.05, 0.05, 0.05]
    viz.get_render_option().point_size = 1.35
    viz.get_render_option().show_coordinate_frame = True

    add_open3d_axis(viz)
    '''
    frame = 0
    
    dt0 = datetime.now()
     
    #lidar_data_received = False  # Verificar si se recibe data de LiDAR
    

    while True:
        pygame.display.flip()
        '''
        if frame == 5 and not lidar_data_received: # Pequeño buffer para que no colapse el visualizador
            # Añadir la nube de puntos solo después de recibir los datos
            viz.add_geometry(point_cloud)
            lidar_data_received = True  # Marca que hemos recibido datos
            print("Geometry added to the visualizer")
         '''
        '''
        # Actualizamos la geometría y nos aseguramos de que los puntos sigan siendo negros
        viz.update_geometry(point_cloud)

        viz.poll_events() # Sondear eventos de usuario para mantener la interactividad fluida en el bucle (movimientos de camara, etc)
        viz.update_renderer() # Actualizar el renderizado con datos nuevos
         '''
        
        time.sleep(0.03) # Tiempo de espera para sincronismo
        world.tick() # Avanzar un frame en el simulador

        # Calcular el tiempo de procesamiento para determinar los FPS
        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:  # Evitar divisiones por cero
            fps = 1.0 / process_time.total_seconds()
            # Actualizar los FPS en la misma línea
            sys.stdout.write(f'\rFPS: {fps:.2f} ')
            sys.stdout.flush()
        dt0 = datetime.now()
        frame += 1

         # Llamar a la función que controla el vehículo
        vehicle_control(vehicle)

        # Condición de salida para terminar de forma segura
        '''
        if not viz.poll_events():
            print("Exiting visualization")
            break
        '''
    #cleanup()  # Asegurarse de limpiar los actores al salir del ciclo principal
    
if __name__ == "__main__":
    # Capturar señales de interrupción (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    
    main()