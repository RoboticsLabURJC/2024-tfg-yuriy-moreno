import carla
import pygame
import random
import time
import numpy as np
import psutil

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
        control.steer = -0.3
    if keys[pygame.K_d]:  # Girar a la derecha
        control.steer = 0.3
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
    print(f'Recibidos {points.shape[0]} puntos de LiDAR')

def get_system_performance():
    # Obtener la carga de CPU
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage


def main():
    # Conectar al servidor CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Obtener el blueprint library para los vehículos
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]  # Selecciona un coche específico que tenga un modelo de físicas realista

    #print(vehicle_bp.get_attribute())

    #vehicle_bp.set_attribute('max_steering_angle', '70')  # Aumentar el ángulo de dirección
    #vehicle_bp.set_attribute('suspension_stiffness', '30000')  # Ajustar la rigidez de la suspensión
    #vehicle_bp.set_attribute('suspension_damping', '1000')  # Ajustar el amortiguamiento de la suspensión
    #vehicle_bp.set_attribute('suspension_travel', '0.2')  # Ajustar el desplazamiento de la suspensión


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

        ##################
        # Configuracion de las colisiones del vehiculo
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

        # Ejemplo de escucha del sensor de colisión
        collision_sensor.listen(lambda event: print("Colisión detectada"))
        #########################


        # Configuración del vehículo
        vehicle.set_autopilot(False)  # Desactivar el piloto automático

        ####################### Aplicar control de física del vehiculo ###################
        # Configurar físicas del vehículo
        vehicle_physics_control = carla.VehiclePhysicsControl()
        
        # Ajustar atributos
        vehicle_physics_control.mass = 1500.0  # Masa en kg
        vehicle_physics_control.drag_coefficient = 0.3  # Coeficiente de arrastre
        
        
        vehicle_physics_control.max_rpm = 6000.0  # RPM máximo
        
        vehicle_physics_control.moi = 10.0  # Momento de inercia
        vehicle_physics_control.damping_rate_full_throttle = 0.5
        vehicle_physics_control.damping_rate_zero_throttle_clutch_engaged = 0.1
        vehicle_physics_control.damping_rate_zero_throttle_clutch_disengaged = 0.1
        
        vehicle_physics_control.use_gear_autobox = True
        vehicle_physics_control.gear_switch_time = 0.5  # Tiempo de cambio de marcha
        vehicle_physics_control.clutch_strength = 30.0  # Fuerza del embrague
        
        # Establecer el centro de masa
        vehicle_physics_control.center_of_mass = carla.Vector3D(0.0, 0.0, 0.5)  # Ejemplo de centro de masa
        
        # Curva de dirección
        #vehicle_physics_control.steering_curve = [0.0, 0.5, 1.0]  # Ejemplo simplificado

        # Aplicar las configuraciones de físicas al vehículo
        vehicle.physics_control = vehicle_physics_control

        ############################################################################################


        ####################### Aplicar control de física de las ruedas ###################
        # Crear un objeto de WheelsPhysicsControl
        wheels_control = carla.WheelPhysicsControl()
        
        # Ajustar los parámetros deseados para las ruedas
        wheels_control.front_left_tire_friction = 2.0
        wheels_control.front_right_tire_friction = 2.0
        wheels_control.rear_left_tire_friction = 2.0
        wheels_control.rear_right_tire_friction = 2.0
        
        wheels_control.suspension_stiffness = 30000  # Rigidez de la suspensión
        wheels_control.suspension_damping = 1000     # Amortiguación de la suspensión
        wheels_control.suspension_travel = 0.2       # Desplazamiento de la suspensión

        # Aplicar el control de física de las ruedas al vehículo
        vehicle.wheel_physics_control = wheels_control

        #########################################################

        ######################## Modo sincrónico
        settings = world.get_settings()
        settings.synchronous_mode = True  # Habilitar modo sincrónico

        # Ajustar el delta de tiempo fijo según el uso de CPU
        cpu_usage = get_system_performance()
        if cpu_usage < 50:  # Bajo uso de CPU
            settings.fixed_delta_seconds = 0.03  # 33.33 FPS
        elif cpu_usage < 80:  # Uso de CPU medio
            settings.fixed_delta_seconds = 0.05  # 20 FPS
        else:  # Alto uso de CPU
            settings.fixed_delta_seconds = 0.1  # 10 FPS

        # Establecer delta de tiempo fijo (por ejemplo, 0.03 segundos)
        #settings.fixed_delta_seconds = 0.03
        world.apply_settings(settings)

        # Configuración de la cámara en tercera persona
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{width}')
        camera_bp.set_attribute('image_size_y', f'{height}')
        camera_bp.set_attribute('fov', '110')  # Campo de visión amplio

        # Posición de la cámara detrás y encima del coche
        camera_transform = carla.Transform(carla.Location(x=-4.0,z=2.5), carla.Rotation(pitch=-15))
        
        # Crear y spawnear la cámara
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Configurar la cámara para enviar imágenes a la función process_img
        camera.listen(lambda image: process_img(image, screen))


        ############################################### Añadir un sensor LiDAR
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

        # Configurar el callback del LiDAR y guardar datos en disco
        #lidar.listen(lambda point_cloud: point_cloud.save_to_disk('/home/yuriy/Lidar/%.6d.ply' % point_cloud.frame))

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

                # Obtener la velocidad del vehículo
                velocity = vehicle.get_velocity()  # Devuelve un vector (x, y, z)
                speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Magnitud de la velocidad
                speed_kmh = speed * 3.6  # Convertir a km/h

                # Mostrar la velocidad en pantalla
                speed_text = font.render(f'Velocidad: {speed_kmh:.2f} km/h', True, (255, 255, 255))
                screen.blit(speed_text, (50, 50))

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