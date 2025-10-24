import numpy as np
from estruturas import NetworkNode, FogNode, Service, Sensor, CloudNode
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import folium
import random as rd
import math
import sys
import os
import networkx as nx



# FUNÇÕES 

def is_point_inside_polygon(point, polygon):
    # Ray casting algorithm
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if point[1] > min(p1y, p2y):
            if point[1] <= max(p1y, p2y):
                if point[0] <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (point[1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or point[0] <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def generate_points_in_polygon(points, num_points):
    # Compute convex hull
    hull = ConvexHull(points)
    # Get vertices of convex hull
    hull_vertices = points[hull.vertices]

    # Generate random points inside convex hull
    min_x, min_y = np.min(hull_vertices, axis=0)
    max_x, max_y = np.max(hull_vertices, axis=0)

    random_points = []
    while len(random_points) < num_points:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = np.array([x, y])
        if is_point_inside_polygon(point, hull_vertices):
            random_points.append(point)

    return np.array(random_points)

def generateFogNodesParameters(bounds, very_high_percentual, high_percentual, medium_percentual, low_percentual):
  global fog_nodes_list
  global n_fog_nodes_per_region
  global current_index
  global fog_node_dictionary

  random_points = generate_points_in_polygon(bounds, n_fog_nodes_per_region)

  n_very_high = 0
  n_high = 0
  n_medium = 0
  n_low = 0

  for i in range(n_fog_nodes_per_region):
    fog_node = FogNode(current_index, random_points[i][0], random_points[i][1], None, None, None, None)

    random_number = rd.uniform(0, 1)
    power_level_fog_node_index = -1

    if(random_number <= very_high_percentual):
      n_very_high = n_very_high + 1
      power_level_fog_node_index = VERY_HIGH_POWER_INDEX
    elif(random_number > very_high_percentual and random_number <= very_high_percentual + high_percentual):
      n_high = n_high + 1
      power_level_fog_node_index = HIGH_POWER_INDEX
    elif(random_number > very_high_percentual + high_percentual and random_number <= very_high_percentual + high_percentual + medium_percentual):
      n_medium = n_medium + 1
      power_level_fog_node_index = MEDIUM_POWER_INDEX
    else:
      n_low = n_low + 1
      power_level_fog_node_index = LOW_POWER_INDEX

    fog_node.power_level = fog_node_dictionary[power_level_fog_node_index][0]
    fog_node.processing_capacity = fog_node_dictionary[power_level_fog_node_index][1]
    fog_node.memory_capacity = fog_node_dictionary[power_level_fog_node_index][2]
    fog_node.cost = fog_node_dictionary[power_level_fog_node_index][3]
    fog_node.model = fog_node_dictionary[power_level_fog_node_index][4]

    fog_nodes_list.append(fog_node)
    current_index = current_index + 1

def getHaversineDistance(node1, node2):
    # Radius of the Earth in meters
    R = 6371.0 * 1000

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(node1.latitude)
    lon1 = math.radians(node1.longitude)
    lat2 = math.radians(node2.latitude)
    lon2 = math.radians(node2.longitude)

    # Calculate the differences
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance
    distance = R * c

    return distance

def generatesubstrateNetwork(first_index, last_index):
  global fog_nodes_network
  global fog_nodes_list

  G = nx.Graph()

  for fog_node_index1 in range(first_index, last_index):
    for fog_node_index2 in range(fog_node_index1 + 1, last_index):
      random_number = rd.uniform(0, 1)
      if(random_number <= FOG_CONNECTION_PROBABILITY):
        fog_nodes_network[(fog_node_index1, fog_node_index2)] = [rd.choice(list(bandwidth_options_for_fog_nodes)), getHaversineDistance(fog_nodes_list[fog_node_index1], fog_nodes_list[fog_node_index2])]
        G.add_edge(fog_node_index1, fog_node_index2)

  while((G.number_of_nodes() == 0 and G.number_of_edges() == 0) or not nx.is_connected(G)):
    fog_node_index1 = rd.randint(first_index, last_index - 1)
    fog_node_index2 = rd.randint(first_index, last_index - 1)
    random_number = rd.uniform(0, 1)
    if(random_number <= FOG_CONNECTION_PROBABILITY and fog_node_index1 < fog_node_index2 and not G.has_edge(fog_node_index1, fog_node_index2)):
      fog_nodes_network[(fog_node_index1, fog_node_index2)] = [rd.choice(list(bandwidth_options_for_fog_nodes)), getHaversineDistance(fog_nodes_list[fog_node_index1], fog_nodes_list[fog_node_index2])]
      G.add_edge(fog_node_index1, fog_node_index2)

  return G

def getService(service_name):
  global service_dictionary

  service = Service(None, None, None, None, None)
  service.service_type = service_name
  service.processing_demand = service_dictionary[service_name][0]
  service.memory_demand = service_dictionary[service_name][1]
  service.number_of_bits = service_dictionary[service_name][2]
  service.lifetime = service_dictionary[service_name][3]

  return service

def generateSensors(bounds, waste_percentual, camera_percentual, air_percentual):
  global n_sensors_per_region
  global sensors_list
  global current_index

  random_points = generate_points_in_polygon(bounds, n_sensor_per_region)
  number_of_services = [0 for sensor_idx in range(n_sensors)]

  for i in range(len(random_points)):
    sensor = Sensor(current_index, random_points[i][0], random_points[i][1], None)
    sensor.ID = current_index
    sensor.longitude = random_points[i][0]
    sensor.latitude = random_points[i][1]

    sensor.services = [] # List of elements with two info: [Service, Last_Time_When_It_Appeared]
    number_of_services[sensor.ID] = rd.randint(1, int(0.1*n_sensors))

    for service_counter in range(number_of_services[sensor.ID]):
      random_number = rd.uniform(0, 1)
      type_of_request = ''
      if(random_number <= waste_percentual):
        type_of_request = 'waste'
      elif(random_number > waste_percentual and random_number <= waste_percentual + camera_percentual):
        type_of_request = 'camera'
      else:
        type_of_request = 'air'

      service = getService(type_of_request)
      sensor.services.append(service)

    sensors_list.append(sensor)
    current_index = current_index + 1

def getMaxHaversineDistanceFromSensorsToFogNodes(sensor):
  max_haversine_distance = -1
  for fog_node in fog_nodes_list:
    harversine_distance = getHaversineDistance(sensor, fog_node)
    if(max_haversine_distance < harversine_distance):
      max_haversine_distance = harversine_distance
  return max_haversine_distance

def determineTechnologyMiscellaneousCase(sensor):
  '''
  max_haversine_distance = getMaxHaversineDistanceFromSensorsToFogNodes(sensor);

  if(max_haversine_distance <= wireless_coverage_radius_in_meters["6G"]):
    return "6G"

  elif(max_haversine_distance <= wireless_coverage_radius_in_meters["5G"]):
    return "5G"

  elif(max_haversine_distance <= wireless_coverage_radius_in_meters["4G"]):
    return "4G"

  elif(max_haversine_distance <= wireless_coverage_radius_in_meters["3G"]):
    return "3G"

  else:
    return "2G"
  '''
  return np.random.choice(["4G", "5G", "6G"])

def getSensorToFogBandwidth(wireless_technology_chosen):
  return np.random.choice(sensor_bandwidth_options[wireless_technology_chosen])
  
def getBandwidthCost(link_config):
  return np.random.choice(sensor_bandwidth_cost_options[link_config])

def generateInfoInstance(instance_ID):
  #print(str(instance_ID) + ".txt")
  info_instance = "#begin_instance_info\n"
  info_instance += "family\t" + "antwerp\n"
  info_instance += "ID\t" + str(instance_ID) + "\n"
  info_instance += "level_of_number_of_sensors\t" + level_for_selecting_sensor + "\n"
  info_instance += "level_for_selecting_services\t" + level_for_selecting_services + "\n"
  info_instance += "fog_to_fog_network_connection_density_level\t" + fog_fog_nodes_connection_choice + "\n"
  info_instance += "fog_to_cloud_network_connection_density_level\t" + fog_cloud_nodes_connection_choice + "\n"
  info_instance += "sensors\t" + str(n_sensors) + "\n"
  info_instance += "fog_nodes\t" + str(int(n_fog_nodes)) + "\n"
  info_instance += "cloud_nodes\t" + str(n_cloud_nodes) + "\n"
  info_instance += "services\t" + str(len(service_dictionary)) + "\n"
  info_instance += "#end_instance_info\n"
  #print(info_instance)
  return info_instance

def generateInfoSensor():
  info_sensor = "#begin_sensors\n"
  for sensor in sensors_list:
    info_sensor += str(sensor.ID) + "\t" + str(sensor.longitude) + "\t" + str(sensor.latitude) + "\t"
    for service_idx in range(len(sensor.services)):
      info_sensor += str(sensor.services[service_idx].service_type)
      if(service_idx < len(sensor.services) - 1):
        info_sensor += "\t"
    info_sensor += "\n"
  info_sensor += "#end_sensors\n"
  #print(info_sensor)
  return info_sensor

def generateInfoReachFogNodes():
  connection_ID = 0
  info_reach_fog_nodes = "#begin_reach_fog_nodes\n"
  for sensor in sensors_list:
    for fog_node_ID in sensor.reachable_fog_nodes:
      harvesine = getHaversineDistance(sensor, fog_nodes_list[fog_node_ID])
      info_reach_fog_nodes += str(connection_ID) + "\t" + str(sensor.ID) + "\t" + str(fog_node_ID) + "\t" + str(sensor_bandwidth_dictionary[(sensor.ID, fog_node_ID)]) + "\t" + str(sensor_bandwidth_cost_dictionary[(sensor.ID, fog_node_ID)]) + "\t" + str(harvesine) + "\n"
      connection_ID += 1
  info_reach_fog_nodes += "#end_reach_fog_nodes\n"
  #print(info_reach_fog_nodes)
  return info_reach_fog_nodes

def generateInfoFogNodes():
  info_fog_nodes = "#begin_fog\n"
  for fog_node in fog_nodes_list:
    info_fog_nodes += str(fog_node.ID) + "\t" + str(fog_node.longitude) + "\t" + str(fog_node.latitude) + "\t" + str(fog_node.processing_capacity) + "\t" + str(fog_node.memory_capacity) + "\t" + str(fog_node.cost) + "\t" + fog_node.model + "\n"
  link_index = 0
  info_fog_nodes += str(len(fog_nodes_network)) + "\n"
  for key, value in fog_nodes_network.items():
      info_fog_nodes += str(link_index) + "\t" + str(key[0]) + "\t" + str(key[1]) + "\t" + str(value[0]) + "\t" + str(getBandwidthCost("(fog,fog)")) + "\t" + str(value[1]) + "\n"
      link_index = link_index + 1
  info_fog_nodes += "#end_fog\n"
  #print(info_fog_nodes)
  return info_fog_nodes

def generateInfoCloudNodes():
  info_cloud_nodes = "#begin_cloud\n"

  CLOUD_BANDWIDTH = 50
  fog_cloud_nodes_network = {}

  while True:
    cloud_node_selected = [False for cloud_node_index in range(len(cloud_nodes_list))]
    n_cloud_nodes_selected = 0
    for fog_node in fog_nodes_list:
      for cloud_node in cloud_nodes_list:
        random_number = rd.uniform(0, 1)
        if(random_number <= CLOUD_CONNECTION_PROBABILITY):
          fog_cloud_nodes_network[(fog_node.ID, cloud_node.ID)] = [CLOUD_BANDWIDTH, getHaversineDistance(fog_node, cloud_node)]
          if(not cloud_node_selected[cloud_node.ID]):
            cloud_node_selected[cloud_node.ID] = True
            n_cloud_nodes_selected = n_cloud_nodes_selected + 1

    if(n_cloud_nodes_selected == len(cloud_nodes_list)):
      break

    else:
      fog_cloud_nodes_network = {}

  for cloud_node in cloud_nodes_list:
    info_cloud_nodes += str(cloud_node.ID) + "\t" + str(cloud_node.longitude) + "\t" + str(cloud_node.latitude) + "\t" + str(cloud_node.processing_capacity) + "\t" + str(cloud_node.memory_capacity) + "\t" + str(cloud_node.cost) + "\t" + cloud_node.model + "\n"

  info_cloud_nodes += str(len(fog_cloud_nodes_network)) + "\n"
  link_index = 0
  for key, value in fog_cloud_nodes_network.items():
    info_cloud_nodes += str(link_index) + "\t" + str(key[0]) + "\t" + str(key[1]) + "\t" + str(value[0]) + "\t" + str(getBandwidthCost("(fog,cloud)")) + "\t" + str(value[1]) + "\n"
    link_index = link_index + 1
  info_cloud_nodes += "#end_cloud\n"
  #print(info_cloud_nodes)
  return info_cloud_nodes

def generateInfoServices():
  min_demanded_bandwidth = 1000000.0
  for sensor_ID in sensor_tech_dict:
    if(wireless_bandwidth[sensor_tech_dict[sensor_ID]] < min_demanded_bandwidth):
      min_demanded_bandwidth = wireless_bandwidth[sensor_tech_dict[sensor_ID]]

  info_services = "#begin_service\n"
  for service_key in service_dictionary:
    info_services += service_key + "\t" + str(service_dictionary[service_key][0]) + "\t" + str(service_dictionary[service_key][1]) + "\t" + str(service_dictionary[service_key][2]) + "\t"  + str(service_dictionary[service_key][3]) + "\t"+ str(min_demanded_bandwidth) + "\n"
  info_services += "#end_service\n"
  #print(info_services)
  return info_services

def selectSensors():
  global sensors_list
  global n_sensors

  number_of_sensors_selected = 0
  while not number_of_sensors_selected:
    sensors_lower_bound = int(lambda_parameter_for_selecting_sensors[0] * n_sensors)
    sensors_upper_bound = int(lambda_parameter_for_selecting_sensors[1] * n_sensors)
    lambda_avg_sensors = rd.randint(sensors_lower_bound, sensors_upper_bound)
    number_of_sensors_selected = min(np.random.poisson(lambda_avg_sensors), len(sensors_list))

  index_sample = np.random.choice(len(sensors_list), size=number_of_sensors_selected, replace=False)
  selected_sensors = [sensors_list[idx].ID for idx in index_sample]
  return selected_sensors

def generateServices(selected_sensor):
  while True:
    services_lower_bound = max(1, int(lambda_parameter_for_selecting_services[0] * len(selected_sensor.services)))
    services_upper_bound = max(1, int(lambda_parameter_for_selecting_services[1] * len(selected_sensor.services)))
    lambda_avg_services = rd.randint(services_lower_bound, services_upper_bound)
    number_of_services_selected = min(np.random.poisson(lambda_avg_services), len(selected_sensor.services))
    selected_services = np.random.choice(len(selected_sensor.services), size=number_of_services_selected, replace=False)

    if(len(selected_services) != 0):
      break
  return selected_services

def generateInfoInstants():
  info_instants = ""
  info_instants += "#begin_requests\n"
  for time_stamp in range(1,time_window+1):
    info_instants += "\t##time_instant_" + str(time_stamp) + "\n"
    selected_sensors = selectSensors()
    for sensor_index in selected_sensors:
      selected_services = generateServices(sensors_list[sensor_index])
      for service_index in selected_services:
        info_instants += "\t\t" + str(sensor_index) + "\t" + str(service_index) + "\t" + sensors_list[sensor_index].services[service_index].service_type + "\t" + str(sensors_list[sensor_index].services[service_index].lifetime) + "\n"
  #print(info_instants)
  info_instants += "#end_requests\n"
  return info_instants

#print(info_instance + info_sensor + info_reach_fog_nodes + info_fog_nodes + info_cloud_nodes + info_services + info_instants)
def generateFileInstance(data_path, instance_ID):
  with open(os.path.join(data_path, instance_ID + ".txt"), "w") as file:
    file.write(generateInfoInstance(instance_ID) + generateInfoSensor() + generateInfoFogNodes() + generateInfoReachFogNodes() + generateInfoCloudNodes() + generateInfoServices() + generateInfoInstants())
    
n_sensors = int(sys.argv[1])
instance_ID = sys.argv[2]
data_path = sys.argv[3]

time_window = 1000

avg_level_for_selecting_sensors = {
    'low': [0.1, 0.3],
    'medium': [0.3,0.7],
    'high': [0.7,0.9]
}
#level_for_selecting_sensor = sys.argv[2]
level_for_selecting_sensor = "low"
lambda_parameter_for_selecting_sensors = avg_level_for_selecting_sensors[level_for_selecting_sensor]

avg_level_for_selecting_services = {
    'low': [0.1, 0.3],
    'medium': [0.3,0.7],
    'high': [0.7,0.9]
}
#level_for_selecting_services = sys.argv[3]
level_for_selecting_services = "low"
lambda_parameter_for_selecting_services = avg_level_for_selecting_services[level_for_selecting_services]

# in Gbps
wireless_technologies = ["2G", "3G", "4G", "5G", "6G", "misc"]

wireless_bandwidth = {
    "2G": 0.00004,
    "3G": 0.001,
    "4G": 0.01,
    "5G": 0.12,
    "6G": 1.0
}

# in meters
wireless_coverage_radius_in_meters = {
    "2G": 10000,
    "3G": 5000,
    "4G": 3000,
    "5G": 600,
    "6G": 320
}

wireless_technology_chosen = "4G"
#wireless_technology_chosen = sys.argv[4]
if(wireless_technology_chosen not in wireless_technologies):
    wireless_technology_chosen = wireless_technologies[2]

probability_of_fog_cloud_nodes_connection = {
    'low': 0.1,
    'medium': 0.5,
    'high': 0.9
}
#fog_fog_nodes_connection_choice = sys.argv[5]
#fog_cloud_nodes_connection_choice = sys.argv[6]
fog_fog_nodes_connection_choice = "low"
fog_cloud_nodes_connection_choice = "low"
FOG_CONNECTION_PROBABILITY = probability_of_fog_cloud_nodes_connection[fog_fog_nodes_connection_choice]
CLOUD_CONNECTION_PROBABILITY = probability_of_fog_cloud_nodes_connection[fog_cloud_nodes_connection_choice]

coordinates = [[51.24468472323838, 4.409597070924777], [51.28850722394544, 4.393117578551932], [51.25478478844565, 4.354665429681959],
[51.22261836485071, 4.399789050587303], [51.21950044460592, 4.397557452661814], [51.22106255764579, 4.4014755576514295], [51.22365158898595, 4.410202582222291], [51.222242915305245, 4.410983280580201], [51.22143657029032, 4.409438328170246],
[51.21680503702301, 4.4254052592802156], [51.21918444072911, 4.420502350843398], [51.217347535261005, 4.421183038214594], [51.21234639567679, 4.414925822930287],
[51.19039399730772, 4.461440932566946], [51.199000353121455, 4.462991551276801], [51.20728219114087, 4.470029667790763],
[51.179361822078775, 4.4153154215211305], [51.18145433403806, 4.412948454482137], [51.17975324780296, 4.399410381235565], 	[51.157665857404005, 4.411433489841647]]

region1 = np.array([[51.24468472323838, 4.409597070924777], [51.28850722394544, 4.393117578551932], [51.25478478844565, 4.354665429681959]])
region2 = np.array([[51.22261836485071, 4.399789050587303], [51.21950044460592, 4.397557452661814], [51.22106255764579, 4.4014755576514295], [51.22365158898595, 4.410202582222291], [51.222242915305245, 4.410983280580201], [51.22143657029032, 4.409438328170246]])
region3 = np.array([[51.21680503702301, 4.4254052592802156], [51.21918444072911, 4.420502350843398], [51.217347535261005, 4.421183038214594], [51.21234639567679, 4.414925822930287]])
region4 = np.array([[51.19039399730772, 4.461440932566946], [51.199000353121455, 4.462991551276801], [51.20728219114087, 4.470029667790763]])
region5 = np.array([[51.179361822078775, 4.4153154215211305], [51.18145433403806, 4.412948454482137], [51.17975324780296, 4.399410381235565], [51.157665857404005, 4.411433489841647]])


region1_bounds = np.array([[51.24468472323838, 4.409597070924777], [51.28850722394544, 4.393117578551932], [51.25478478844565, 4.354665429681959]])
region2_bounds = np.array([[51.22617381833156, 4.399406371959821], [51.225099, 4.414169], [51.211067, 4.406015], [51.203808, 4.384214], [51.208271, 4.384729]])
region3_bounds = np.array([[51.221279, 4.417131], [51.210186, 4.408558], [51.200626, 4.414529], [51.218179, 4.428512]])
region4_bounds = np.array([[51.213430, 4.451353], [51.17932963353455, 4.4417398839160525], [51.17932963353455, 4.466630783854205], [51.21020391237721, 4.489290085866867]])
region5_bounds = np.array([[51.18962903393958, 4.410505163396242], [51.18285049873252, 4.391794073097907], [51.17364946330412, 4.389004575681392], [51.17351493162472, 4.413637983551219], [51.18290430053395, 4.416770803667099]])

n_regions = 5
n_sensor_per_region = n_fog_per_region = 0

LOW_POWER_INDEX = 0
MEDIUM_POWER_INDEX = 1
HIGH_POWER_INDEX = 2
VERY_HIGH_POWER_INDEX = 3

fog_node_dictionary = {
    # ID: [power_level, processing_capacity, memory_capacity, cost, model]
    LOW_POWER_INDEX: ["low", 8, 256, 1.0962, "HX-CPU-I4309Y"],
    MEDIUM_POWER_INDEX: ["medium", 16, 256, 1.4094, "HX-CPU-I4314"],
    HIGH_POWER_INDEX: ["high", 32, 256, 2.1402, "HX-CPU-I6314U4"],
    VERY_HIGH_POWER_INDEX: ["very high", 40, 256, 2.79936, "HX-CPU-I8380"],
}

bandwidth_options_for_fog_nodes = {1, 2, 3}

SENSOR_FOG_FACTOR = 0.5
#n_fog_nodes = max(10, int(n_sensors/SENSOR_FOG_RATIO))

n_fog_nodes = int(SENSOR_FOG_FACTOR * n_sensors)
n_fog_nodes_per_region = int(n_fog_nodes/n_regions)

fog_nodes_network = {}
fog_nodes_list = []
fog_nodes_index_by_region = {}
current_index = 0

N_SENSORS_THRESHOLD = 500

first_index_of_region = current_index
if(n_sensors > N_SENSORS_THRESHOLD):
    generateFogNodesParameters(region1_bounds, 0.2, 0.35, 0.35, 0.10)
else:
    generateFogNodesParameters(region1_bounds, 0, 0, 0.7, 0.3)
fog_nodes_graph = generatesubstrateNetwork(first_index_of_region, first_index_of_region + n_fog_nodes_per_region)
fog_nodes_index_by_region['port_area'] = [first_index_of_region, first_index_of_region + n_fog_nodes_per_region]

first_index_of_region = current_index
if(n_sensors > N_SENSORS_THRESHOLD):
    generateFogNodesParameters(region2_bounds, 0.2, 0.35, 0.35, 0.1)
else:
    generateFogNodesParameters(region2_bounds, 0, 0, 0.7, 0.3)
fog_nodes_graph = generatesubstrateNetwork(first_index_of_region, first_index_of_region + n_fog_nodes_per_region)
fog_nodes_index_by_region['grote_markt'] = [first_index_of_region, first_index_of_region + n_fog_nodes_per_region]

first_index_of_region = current_index
if(n_sensors > N_SENSORS_THRESHOLD):
    generateFogNodesParameters(region3_bounds, 0.1, 0.3, 0.3, 0.3)
else:
    generateFogNodesParameters(region3_bounds, 0, 0, 0.5, 0.5)
fog_nodes_graph = generatesubstrateNetwork(first_index_of_region, first_index_of_region + n_fog_nodes_per_region)
fog_nodes_index_by_region['zoo'] = [first_index_of_region, first_index_of_region + n_fog_nodes_per_region]

first_index_of_region = current_index
if(n_sensors > N_SENSORS_THRESHOLD):
    generateFogNodesParameters(region4_bounds, 0.3, 0.3, 0.3, 0.1)
else:
    generateFogNodesParameters(region4_bounds, 0, 0, 0.7, 0.3)
fog_nodes_graph = generatesubstrateNetwork(first_index_of_region, first_index_of_region + n_fog_nodes_per_region)
fog_nodes_index_by_region['luchthaven'] = [first_index_of_region, first_index_of_region + n_fog_nodes_per_region]

first_index_of_region = current_index
if(n_sensors > N_SENSORS_THRESHOLD):
    generateFogNodesParameters(region5_bounds, 0.1, 0.3, 0.3, 0.3)
else:
    generateFogNodesParameters(region5_bounds, 0, 0, 0.5, 0.5)
fog_nodes_graph = generatesubstrateNetwork(first_index_of_region, first_index_of_region + n_fog_nodes_per_region)
fog_nodes_index_by_region['faculty_applied_engineering'] = [first_index_of_region, first_index_of_region + n_fog_nodes_per_region]

G = nx.Graph()
for edge in fog_nodes_network:
    u = edge[0]
    v = edge[1]
    G.add_edge(u, v)

while(not nx.is_connected(G)):
    for region1 in fog_nodes_index_by_region:
        for region2 in fog_nodes_index_by_region:
            if(region1 != region2):
                random_number = rd.uniform(0, 1)
                if(random_number <= FOG_CONNECTION_PROBABILITY):
                    fog_node_1 = rd.randint(fog_nodes_index_by_region[region1][0], fog_nodes_index_by_region[region1][1] - 1)
                    fog_node_2 = rd.randint(fog_nodes_index_by_region[region2][0], fog_nodes_index_by_region[region2][1] - 1)
                    if(fog_node_1 < fog_node_2 and (fog_node_1, fog_node_2) not in fog_nodes_network):
                        fog_nodes_network[(fog_node_1, fog_node_2)] = [rd.choice(list(bandwidth_options_for_fog_nodes)), getHaversineDistance(fog_nodes_list[fog_node_1], fog_nodes_list[fog_node_2])]
                        G.add_edge(fog_node_1, fog_node_2)

service_dictionary = {
    # ID: [processing_demand, memory_demand, number_of_bits, lifetime]
    'waste': [0.2125, 0.375, 296, 50],
    'camera': [0.35, 0.475, 12000, 10],
    'air': [0.25, 0.3125, 744, 100]
}

n_sensor_per_region = int(n_sensors/n_regions)
current_index = 0
sensors_list = []
generateSensors(region1_bounds, 0.4, 0.2, 0.4)
generateSensors(region2_bounds, 0.4, 0.4, 0.2)
generateSensors(region3_bounds, 0.65, 0.25, 0.1)
generateSensors(region4_bounds, 0.2, 0.4, 0.4)
generateSensors(region5_bounds, 0.25, 0.65, 0.1)

sensor_bandwidth_dictionary = {}
sensor_bandwidth_options = {
    "2G": [0.00004],
    "3G": [0.00004, 0.001],
    "4G": [0.00004, 0.001, 0.01],
    "5G": [0.00004, 0.001, 0.01, 0.12],
    "6G": [0.00004, 0.001, 0.01, 0.12, 1.0]
}

sensor_bandwidth_cost_dictionary = {}
sensor_bandwidth_cost_options = {
    "(sensor,fog)": [0.0092, 0.046],
    "(fog,fog)": [0.046, 0.064, 0.078],
    "(fog,cloud)": [0.083]
}

sensor_tech_dict = {}
for sensor in sensors_list:
    if(wireless_technology_chosen == "misc"):
        sensor_tech_dict[sensor.ID] = determineTechnologyMiscellaneousCase(sensor)
    else:
        sensor_tech_dict[sensor.ID] = wireless_technology_chosen
    sensor.reachable_fog_nodes = []

    coverage_ratio_in_meters = wireless_coverage_radius_in_meters[sensor_tech_dict[sensor.ID]]
    for fog_node in fog_nodes_list:
        harversine_distance = getHaversineDistance(sensor, fog_node)
        if(harversine_distance <= coverage_ratio_in_meters):
            sensor.reachable_fog_nodes.append(fog_node.ID)
            #sensor_bandwidth_dictionary[(sensor.ID, fog_node.ID)] = getSensorToFogBandwidth(sensor_tech_list[sensor.ID])
            sensor_bandwidth_dictionary[(sensor.ID, fog_node.ID)] = sensor_bandwidth_options[sensor_tech_dict[sensor.ID]][-1]
            sensor_bandwidth_cost_dictionary[(sensor.ID, fog_node.ID)] = getBandwidthCost("(sensor,fog)")

    if(len(sensor.reachable_fog_nodes) == 0):
        print(sensor.ID, " ", len(sensor.reachable_fog_nodes))

n_cloud_nodes = 3

LOW_POWER_CLOUD_INDEX = 0
MEDIUM_POWER_CLOUD_INDEX = 1
HIGH_POWER_CLOUD_INDEX = 2

cloud_node_dictionary = {
    # ID: [power_level, processing_capacity, memory_capacity, cost, model]
    LOW_POWER_INDEX: ["low", 48.0, 192.0, 2.94, "m5d.12xlarge"],
    MEDIUM_POWER_CLOUD_INDEX: ["medium", 96.0, 384.0, 8.50, "g4dn.metal"],
    HIGH_POWER_CLOUD_INDEX: ["high", 192.0, 768.0, 10.49, "m7i.48xlarge"]
}

cloud_nodes_list = []
current_index = 0

for cloud_node_idx in range(n_cloud_nodes):
    cloud_node = CloudNode(cloud_node_idx, 48.92744292889152, 2.353369482741328, None, None, None, None)
    cloud_node.power_level = cloud_node_dictionary[cloud_node_idx][0]
    cloud_node.processing_capacity = cloud_node_dictionary[cloud_node_idx][1]
    cloud_node.memory_capacity = cloud_node_dictionary[cloud_node_idx][2]
    cloud_node.cost = cloud_node_dictionary[cloud_node_idx][3]
    cloud_node.model = cloud_node_dictionary[cloud_node_idx][4]
    cloud_nodes_list.append(cloud_node)

generateFileInstance(data_path, instance_ID)
  
# ------------------------------------


