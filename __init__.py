import logging, os, sys
import subprocess, pickle, CMC, lerInstancia
from main import gerarGraficos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Noh:
    
    def __init__(self, id):
        self.id = id
    
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

class FogNode(Noh):
    
    def __init__(self, id, processing_capacity, memory_capacity):
        super().__init__(id)
        self.processing_capacity = processing_capacity
        self.memory_capacity = memory_capacity
        self.requisicoes = 0

class CloudNode(Noh):
    def __init__(self, id, processing_capacity, memory_capacity):
        super().__init__(id)
        self.processing_capacity = processing_capacity
        self.memory_capacity = memory_capacity
        self.requisicoes = 0

class Request:

    def __init__(self, id, processing_demand, memory_demand, nbits, lifetime):
        self.id = id
        self.processing_demand = processing_demand
        self.memory_demand = memory_demand
        self.lifetime = lifetime
        self.number_of_bits = nbits

    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.id})"

class Grafo:

    def __init__(self, vertices, arestas):
        self.adj = {}
        self.adj_ = {}
        for v in vertices:
            self.adj[v] = []
        self.adj_ = {}
        for aresta in arestas:
            self.adj_[f"({aresta.u},{aresta.v})"] = aresta

class Aresta:

    def __init__(self, u, v, largura_banda, tempo):
        self.u = u
        self.v = v
        self.largura_banda = largura_banda
        self.tempo = tempo
        self.bdp = largura_banda*tempo
        self.custo = 1

    def __hash__(self):
        return hash(self.id)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.u}, {self.v})"

def processar_caminho(caminho, requisicao, dist, fog, grafo):
    arcos = []
    band_tot = 0
    custo = 0
    selecionado = None
    tempo = 0
    motivo= f"No caminho {caminho}: "
       
    # Percorre o caminho por indices
    for i in range(len(caminho)-1):
        vertice_atual = caminho[i]
        vizinho = caminho[i+1]
        gasto_req = requisicao.number_of_bits/(10**9) # Converte de bits para Gbits
        tempo += dist[vertice_atual][vizinho]
        if (gasto_req <= grafo.adj_[f"({vertice_atual},{vizinho})"].bdp):
          if(tempo <= requisicao.lifetime):
            selecionado = vizinho
            largura_banda = gasto_req/grafo.adj_[f"({vertice_atual},{vizinho})"].tempo
            band_tot += largura_banda
            arcos.append((f"({vertice_atual},{selecionado})", gasto_req))
            custo += largura_banda * grafo.adj_[f"({vertice_atual},{selecionado})"].custo
          else:
            motivo += f"Requisição estourou o tempo de vida em {vizinho}\n"
            break
        else:
            motivo += f"Requisição estourou a largura de banda indo de {vertice_atual} para {vizinho}\n"
            break

    if selecionado is None:
        return selecionado, False, arcos, band_tot, custo, motivo
    
    # Testa se o vizinho selecionado é apto para processar a requisicao, com base na capacidade de processamento e de memória do Nó Fog
    if (selecionado == fog):
        if(requisicao.processing_demand <= selecionado.processing_capacity):
            if (requisicao.memory_demand <= selecionado.memory_capacity):
                selecionado.processing_capacity -= requisicao.processing_demand
                selecionado.memory_capacity -= requisicao.memory_demand
                selecionado.requisicoes += 1
                if(isinstance(selecionado, FogNode)): 
                    return selecionado, True, arcos, band_tot, custo, motivo
                else:
                    motivo += "Requisição foi processada no CloudNode\n"
                    return selecionado, False, arcos, band_tot, custo, motivo
            else:
                motivo += f"Requisição estourou a memória em {selecionado}\n"
        else:
            motivo += f"Requisição estourou o processamento em {selecionado}\n"
    else:
        motivo += f"Requisição não processada em {selecionado}, pois não é o nó destino\n"
    
    if isinstance(fog, CloudNode):
        motivo += "Nó destino é CloudNode.\n"
        return fog, False, arcos, band_tot, custo, motivo
    
    return selecionado, False, arcos, band_tot, custo, motivo

if __name__ == "__main__":
    gerarGraficos("resultados.csv")
    sys.exit(0)
    sensor = Noh(1)
    fog1 = FogNode(2, 10, 10)
    fog2 = FogNode(3, 10, 10)
    cloud = CloudNode(4, float('inf'), float('inf'))
    dist = {
        sensor: {
            sensor: 0,
            fog1: 1,
            fog2: 3,
            cloud: 8
        },
        fog1: {
            sensor: float('inf'),
            fog1: 0,
            fog2: 2,
            cloud: 7
        },
        fog2: {
            sensor: float('inf'),
            fog1: float('inf'),
            fog2: 0,
            cloud: 5
        }
    }
    caminho = [sensor, fog1, fog2, cloud]
    requisicao = Request(1, 5, 5, 2000000000, 10)
    arestas = [Aresta(sensor, fog1, 2, 1), Aresta(fog1, fog2, 2, 2), Aresta(fog2, cloud, 50, 5)]
    grafo = Grafo(caminho, arestas)
    print(processar_caminho(caminho, requisicao, dist, cloud, grafo))