import carla
import pygame
import random
import time
import numpy as np
import open3d as o3d  # Importar Open3D

# Inicializar Pygame para capturar las entradas del teclado
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("CARLA Vehículo Control")

# Fuente para mostrar texto en pantalla
font = pygame.font.Font(None, 36)

# Función para controlar el vehículo con el teclado
def control_vehicle(vehicle):
    keys = pygame.key.get_pressed()
    
    control = carla.VehicleControl()
    control.throttle = 0.0
    control.steer = 0.0
    control.brake = 0.0
    control.hand_brake = False
    
    if keys[pygame.K_w]:  # Acelerar
        control.throttle = 1.0
    if keys[pygame.K_s]:  # Frenar
        control.brake = 1.0
    if keys[pygame.K_a]:  # Girar a la izquierda
        control.steer = -0.5
    if keys[pygame.K_d]:  # Girar a la derecha
        control.steer = 0.5
    if keys[pygame.K_SPACE]:  # Freno de mano
        control.hand_brake = True

    # Aplicar el control al vehículo
    vehicle.apply_control(control)

# Función de callback para recibir las imágenes de la cámara
def process_img(image, display):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Quitar el canal alfa
    array = array[:, :, ::-1]  # Convertir de BGRA a RGB
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))  # Dibujar en la pantalla

# Función de callback para procesar los datos del LiDAR
def process_lidar(data):
    points = np.frombuffer(data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))  # [x, y, z, intensity]

    # Comprobar la forma y el tipo de datos de points
    print("Forma de points:", points.shape)  # Debe ser (n, 4)
    print("Tipo de datos de points:", points.dtype)  # Debe ser float

    # Asegúrate de que points es un arreglo de tipo float
    if points.dtype != np.float32 and points.dtype != np.float64:
        points = points.astype(np.float32)  # Cambia el tipo de datos si es necesario

    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print("Los puntos contienen NaN o valores infinitos.")

    
    # Verificar valores NaN o infinitos en points
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print("Los puntos contienen NaN o valores infinitos.")
        return  # Salir de la función si hay datos inválidos

    # Crear una nube de puntos de Open3D
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Usar solo [x, y, z]
        print("Puntos asignados correctamente.")
    except Exception as e:
        print("Error al asignar puntos:", e)



    
# Visualizar la nube de puntos
    try:
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print("Error al visualizar la nube de puntos:", e)


def main():
    # Conectar al servidor CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Obtener el blueprint library para los vehículos
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]  # Selecciona un coche cualquiera

    # Configuración de color del vehículo
    if vehicle_bp.has_attribute('color'):
        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
        vehicle_bp.set_attribute('color', color)  # Seleccionar un color válido

    # Obtener un spawn point aleatorio
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # Spawnear el vehículo
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print('Vehículo spawneado correctamente')

        # Configuración del vehículo
        vehicle.set_autopilot(False)  # Desactivar el piloto automático

        # Modo sincrónico
        settings = world.get_settings()
        settings.synchronous_mode = True  # Habilitar modo sincrónico
        # Establecer delta de tiempo fijo (por ejemplo, 0.05 segundos)
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Configuración de la cámara en tercera persona
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{width}')
        camera_bp.set_attribute('image_size_y', f'{height}')
        camera_bp.set_attribute('fov', '110')  # Campo de visión amplio

        # Posición de la cámara detrás y encima del coche
        camera_transform = carla.Transform(carla.Location(x=-6.0, z=2.5), carla.Rotation(pitch=-15))
        
        # Crear y spawnear la cámara
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Configurar la cámara para enviar imágenes a la función process_img
        camera.listen(lambda image: process_img(image, screen))

        # Añadir un sensor LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')  # Rango del LiDAR en metros
        lidar_bp.set_attribute('channels', '32')  # Número de canales
        lidar_bp.set_attribute('points_per_second', '56000')  # Resolución del LiDAR
        lidar_bp.set_attribute('rotation_frequency', '10')  # Frecuencia de rotación en Hz

        # Colocar el LiDAR en el techo del vehículo
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        # Configurar el callback del LiDAR
        lidar.listen(lambda data: process_lidar(data))

        try:
            # Bucle principal para control del coche
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Controlar el coche con el teclado
                control_vehicle(vehicle)

                # Avanzar un tick en la simulación
                world.tick()

                # Mostrar un texto de información
                text = font.render('Controlando el vehículo', True, (255, 255, 255))
                screen.blit(text, (50, 50))

                pygame.display.flip()  # Actualizar la pantalla

                # Pequeño delay
                time.sleep(0.05)

        finally:
            # Restaurar los ajustes originales
            settings.synchronous_mode = False
            world.apply_settings(settings)

            # Destruir el vehículo, la cámara y el LiDAR
            lidar.destroy()
            camera.destroy()
            vehicle.destroy()
            print("Simulación terminada, vehículo, cámara y LiDAR destruidos.")
    
    else:
        print('Error: No se pudo spawnear el vehículo.')

if __name__ == '__main__':
    main()
