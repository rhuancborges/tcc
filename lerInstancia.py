from estruturas import Grafo, Sensor, FogNode, CloudNode, Service, Request
import os

def run(pathfile):
    grafo = Grafo()
    with open(pathfile, "r", encoding="utf-8") as f:
        # Primeiras 7 linhas do arquivo de instância ignoradas
        for i in range(7):
            f.readline()

        # sensors X
        linha = f.readline().strip()
        num_sensores = int(linha.split()[1])

        # fog_nodes X
        linha = f.readline().strip()
        num_fog = int(linha.split()[1])

        # cloud_nodes X
        linha = f.readline().strip()
        num_cloud = int(linha.split()[1])

        # services X
        linha = f.readline().strip()
        num_servicos = int(linha.split()[1])

        # Ignora as próximas duas linhas
        f.readline() # #end_instance_info
        f.readline() # #begin_sensors

        # Lê os Nós Sensores
        sensores = []
        for i in range(num_sensores):
            linha = f.readline().strip().split()
            # Sensores -> index | longitude | latitude | services
            sensor = Sensor(int(linha[0]), linha[1], linha[2], linha[3::])
            sensores.append(sensor)
            grafo.add_vertice(sensor)

        # Ignora as próximas duas linhas
        f.readline() # #end_sensors
        f.readline() # #begin_fog

        # Lê os Nós Fog
        fogs = []
        for i in range(num_fog):
            linha = f.readline().strip().split()
            # Fog -> index | longitude | latitude | processing capacity | memory capacity | cost | model
            fog = FogNode(int(linha[0]), float(linha[1]), float(linha[2]), float(linha[3]), float(linha[4]), float(linha[5]), linha[6])
            fogs.append(fog)
            grafo.add_vertice(fog)

        # Lê as arestas entre Nós Fog
        num_fogfog = int(f.readline().strip())
        for i in range(num_fogfog):
            linha = f.readline().strip().split()
            # Arestas -> node i | node j | bandwidth i-j | bandwidth cost (US$/Gbps) | haversine distance i-j
            grafo.add_aresta(fogs[int(linha[1])], fogs[int(linha[2])], float(linha[3]), float(linha[4]), float(linha[5]))

        # Ignora as próximas duas linhas
        f.readline() # #end_fog
        f.readline() # #begin_reach_fog_nodes

        # Lê as arestas entre Nós Sensores e Nós Fog
        linha = f.readline().strip()
        while linha != "#end_reach_fog_nodes":
            linha = linha.split()
            # Arestas -> node i | node j | bandwidth i-j | bandwidth cost (US$/Gbps) | haversine distance i-j
            grafo.add_aresta(sensores[int(linha[1])], fogs[int(linha[2])], float(linha[3]), float(linha[4]), float(linha[5]))
            linha = f.readline().strip()

        # Ignora a próxima linha
        f.readline() # #begin_cloud

        # Lê os Nós Cloud
        clouds = []
        for i in range(num_cloud):
            linha = f.readline().strip().split()
            # Cloud -> index | longitude | latitude | processing capacity | memory capacity | cost | model
            cloud = CloudNode(int(linha[0]), float(linha[1]), float(linha[2]), float(linha[3]), float(linha[4]), float(linha[5]), linha[6])
            clouds.append(cloud)

        # Lê as arestas entre Nós Fog e Nós Cloud
        num_fogcloud = int(f.readline().strip())
        for i in range(num_fogcloud):
            linha = f.readline().strip().split()
            # Arestas -> node i | node j | bandwidth i-j | bandwidth cost (US$/Gbps) | haversine distance i-j
            grafo.add_aresta(fogs[int(linha[1])], clouds[int(linha[2])], float(linha[3]), float(linha[4]), float(linha[5]))

        # Ignora as próximas duas linhas
        f.readline() # #end_cloud
        f.readline() # #begin_service

        # Lê os serviços
        servicos = {}
        for i in range(num_servicos):
            linha =  f.readline().strip().split()
            # ID: [processing_demand, memory_demand, number_of_bits, lifetime]
            servicos[linha[0]] = Service(linha[0], float(linha[1]), float(linha[2]), int(linha[3]), int(linha[4]))

        # Ignora as próximas duas linhas
        f.readline() # #end_services
        f.readline() # #begin_requests

        requisicoes = []
        i = 0
        linha = f.readline().strip()
        while linha != "#end_requests":
            instante = int(linha.split("_")[2])
            linha = f.readline().strip()  # Lê uma requisição
            # Lê as requisições daquele instante
                # sensor ID | service index | type of service | request_lifetime
            while (linha.split("_")[0] != "##time") and (linha != "#end_requests"):
                linha = linha.split()
                i += 1
                requisicoes.append(Request(i, sensores[int(linha[0])], servicos[linha[2]], instante))
                linha = f.readline().strip()
        # Fim da leitura do arquivo

        #print(fogs)
        #print(sensores)
        #print(servicos)
        #print(requisicoes)
        return grafo, requisicoes, fogs, sensores
    
if __name__ == "__main__":
    instance_file = "1.txt"
    grafo, requisicoes, fogs, sensores = run(os.path.join("instances", instance_file))