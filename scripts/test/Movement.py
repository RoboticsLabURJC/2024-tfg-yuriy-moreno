import carla
import random
import time

import carla
import random
import time

def main():
    # Conectar al servidor CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # Tiempo de espera de 10 segundos para la conexión
    world = client.get_world()

    # Obtener el blueprint library para los vehículos
    blueprint_library = world.get_blueprint_library()

    # Seleccionar un vehículo aleatorio
    vehicle_bp = blueprint_library.filter('vehicle')[0]  # Elige el primer coche de la lista

    # Obtener un spawn point aleatorio
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # Spawnear el vehículo
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print('Vehículo spawneado correctamente')

        # Configuración básica del controlador
        vehicle.set_autopilot(False)  # Desactivar el piloto automático

        # Modo sincrónico para mejor control
        settings = world.get_settings()
        settings.synchronous_mode = True  # Habilitar modo sincrónico
        world.apply_settings(settings)

        try:
            # Bucle de control del vehículo (Ejemplo básico)
            while True:
                # Aquí puedes añadir control manual o IA, pero para fines de ejemplo lo dejamos simple
                control = carla.VehicleControl()

                # Input manual del coche: acelerar, frenar, girar (valores de ejemplo)
                control.throttle = 0.5  # Acelerar al 50%
                control.steer = 0.0     # No girar
                control.brake = 0.0     # No frenar

                # Aplicar el control al vehículo
                vehicle.apply_control(control)

                # Avanzar un tick en el simulador (solo para modo sincrónico)
                world.tick()

                time.sleep(0.05)  # Pequeño delay para evitar sobrecarga en la simulación

        finally:
            # Restaurar los ajustes al final de la simulación
            settings.synchronous_mode = False
            world.apply_settings(settings)

            # Destruir el vehículo al terminar
            vehicle.destroy()
            print("Simulación terminada, vehículo destruido.")

    else:
        print('Error: No se pudo spawnear el vehículo.')

if __name__ == '__main__':
    main()
