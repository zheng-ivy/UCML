import numpy as np
import os
import subprocess

np.random.seed(42)

configs = []

bus_systems = [
    ('case4gs', 20),
    ('case5', 20),
    ('case6ww', 20),
    ('case9', 30),
    ('case14', 50),
    ('case24_ieee_rts', 60),
    ('case30', 70),
    ('case33bw', 70),
    ('case39', 90),
    ('case57', 100),
    ('case118', 130),
    ('case145', 130),
    ('illinois200', 130),
    ('case300', 130),
    ('case1354pegase', 50), # 50
    ('case1888rte', 1),
    ('case2848rte', 1),
    ('case2869pegase', 1),
    ('case3120sp', 1),
    ('case6470rte', 1),
]

instance_id = 1

for bus_system, count in bus_systems:
    for _ in range(count):
        configs.append({
            'Instance ID': instance_id,
            'Bus System': bus_system,
            'Random Seed': instance_id
        })
        instance_id += 1

for idx, config in enumerate(configs):
    instance_id = config['Instance ID']
    bus_system = config['Bus System']
    random_seed = config['Random Seed']

    print(f"Generating Instance {instance_id}: {bus_system}")
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    command = ['python', 'generate_case.py', bus_system]
    subprocess.run(command)