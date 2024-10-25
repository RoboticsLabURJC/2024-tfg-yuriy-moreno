import carla
import random
import time

# Conéctate al servidor de CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)  # Ajusta el tiempo de espera según sea necesario

# Obtén el mundo
world = client.get_world()

# Agregar un vehículo al azar en el mundo
blueprints = world.get_blueprint_library().filter('vehicle.*')
vehicle_bp = random.choice(blueprints)

# Define la ubicación inicial del vehículo
spawn_points = world.get_map().get_spawn_points()
if spawn_points:
    spawn_point = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
else:
    print("No hay puntos de aparición disponibles.")

# Define la ubicación del LiDAR
lidar_location = carla.Vector3D(0, 0, 2.0)  # Ajusta la altura según sea necesario

# Crea el sensor LiDAR
lidar_blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_transform = carla.Transform(lidar_location, carla.Rotation(0, 0, 0))

# Configura atributos del LiDAR
lidar_blueprint.set_attribute('range', '100')  # Rango en metros
lidar_blueprint.set_attribute('points_per_second', '100000')  # Puntos por segundo

# Spawnea el sensor LiDAR y lo adjunta al vehículo
lidar = world.spawn_actor(lidar_blueprint, lidar_transform, attach_to=vehicle)

# Función para manejar los datos del LiDAR
def lidar_callback(data):
    print('Received LiDAR data with {} points'.format(len(data)))

# Suscribirse a los datos del LiDAR
lidar.listen(lidar_callback)

try:
    # Mantén el script en ejecución
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('Interrumpido por el usuario.')
finally:
    # Detén el sensor LiDAR y limpia los recursos
    lidar.destroy()
    vehicle.destroy()
    print('LiDAR y vehículo destruidos.')
