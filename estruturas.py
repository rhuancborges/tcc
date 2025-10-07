# CLASSE NETWORKNODE - Define um objeto "Nó" com características mais gerais
class NetworkNode:
    def __init__(self, ID, latitude, longitude):
        self.ID = ID
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f"{self.__class__.__name__}({self.ID})"

    def __hash__(self):
        return hash(self.ID)

    def __eq__(self, other):
        return (self.ID == other.ID) and (type(self) is type(other))

# CLASSE FOGCLOUDNODE - Especifica um tipo de "Nó" em "Nó Fog" ou "Nó Cloud"
class FogCloudNode(NetworkNode):
    def __init__(self, ID, longitude, latitude, processing_capacity, memory_capacity, cost, model):
        super().__init__(ID, latitude, longitude)
        self.model = model
        self.processing_capacity = processing_capacity
        self.memory_capacity = memory_capacity
        self.cost = cost
        self.requisicoes = 0

# CLASSE FOGNODE - Especifica um "Nó Fog"
# Compartilha todas as características da classe pai
class FogNode(FogCloudNode):
    pass

# CLASSE CLOUDNODE - Especifica um "Nó Cloud"
# Compartilha todas as características da classe pai
class CloudNode(FogCloudNode):
    pass

# CLASSE SENSOR - Especifica um tipo de "Nó" em "nó Sensor"
class Sensor(NetworkNode):
    def __init__(self, ID, longitude, latitude, services):
        super().__init__(ID, latitude, longitude)
        self.services = services

# CLASSE ARESTA - Define um objeto para representar os arcos/arestas do grafo do problema
# Foi necessária a criação desse tipo de objeto devido aos seus múltiplos atributos
class Aresta:
    def __init__(self, u, v, largura_banda, custo, tempo):
        self.u = u
        self.v = v
        self.largura_banda = largura_banda
        self.custo = custo
        self.tempo = tempo
        self.quantidade_uso = 0

    def __repr__(self):
        return f"{self.__class__.__name__}({self.u}, {self.v})"

# CLASSE GRAFO - Define um objeto para representar o grafo do problema
class Grafo:
    def __init__(self):
        self.adj = {}  # Lista de Adjacências
        self.adj_ = {} # Lista de Arestas
        self.n_vertices = 0
        self.n_arestas = 0

    def add_vertice(self, noh):
        if noh not in self.adj:
            self.adj[noh] = []
            self.n_vertices += 1

    def add_aresta(self, u, v, largura_banda, custo, dist_H):
        self.add_vertice(u)
        self.add_vertice(v)
        # Em uma aresta, o atributo tempo significa o tempo de propagação dos dados nessa aresta
        # O tempo é definido como o tempo de atraso (numero de bits/velocidade de transmissão) somado ao tempo de propagação (distância Harvesine/velocidade de propagação)
            # O tempo de atraso para cada tipo de requisição, para a tecnologia 4G, é:
                # waste: 0.0000296
                # camera: 0.0012
                # air: 0.0000744
            # Média aritmética: 0.0004346
        tempo = 0.001 + float(dist_H)/(2*10**8)
        self.adj[u].append((v, largura_banda, custo, tempo))
        aresta = Aresta(u, v, largura_banda, custo, tempo)
        #aresta.largura_banda = float(largura_banda)
        self.adj_[f"({u},{v})"] = aresta
        self.n_arestas += 1

class Request:
    def __init__(self, ID, sensor, service, instante):
        self.ID = ID
        self.sensor = sensor
        self.service = service
        self.instante = instante
    def __repr__(self):
        return f"{self.__class__.__name__}({self.ID})"


class Service:
    def __init__(self, id, processing_demand, memory_demand, number_of_bits, lifetime):
        self.id = id
        self.processing_demand = processing_demand
        self.memory_demand = memory_demand
        self.number_of_bits = number_of_bits
        self.lifetime = lifetime