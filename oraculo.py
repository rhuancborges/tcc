import sys
import numpy as np
from Environment import RKOEnvAbstract
import os 
from CMC import reconstruir_caminho, floyd_warshall
import RKO, lerInstancia
current_directory = os.path.dirname(os.path.abspath(__file__))


class FogEnv(RKOEnvAbstract):
    """
    Environment for the Fog Computing Request Assignment Problem.
    Sensors generate requests that must be assigned to servers, respecting:
    - Request lifetime (max time to reach server)
    - Server resource constraints (CPU, memory)
    - Bandwidth is not a constraint
    The decoder interprets a random-key vector as a sequence of requests and servers.
    """
    def __init__(self, grafo, requisicoes, servers):
        super().__init__()
        self.grafo = grafo
        self.requisicoes = requisicoes
        self.servers = servers
        self.dist, self.prev = floyd_warshall(self.grafo)
        self.reconstruir_caminho = reconstruir_caminho  # Callable
        self.tam_solucao = len(self.requisicoes) + len(self.servers)
        self.LS_type = 'Best'
        self.dict_best = {}
        self.instance_name = 'fog_instance'
        self.save_q_learning_report = False
        # Parameter dicts can be customized as needed

    def decoder(self, keys: np.ndarray):
        """
        Decodifica o vetor de chaves conforme a lógica:
        - Se o último item não for servidor, rotaciona até que seja.
        - Percorre a lista ordenada, acumulando requisições e atribuindo-as ao próximo servidor encontrado.
        Retorna lista de (request_idx, server_idx).
        """
        n_req = len(self.requisicoes)
        n_srv = len(self.servers)
        key_tuples = [(i, 'req') for i in range(n_req)] + [(i, 'srv') for i in range(n_srv)]
        zipped = list(zip(keys, key_tuples))
        #print("Initial keys:", zipped)
        zipped.sort(key=lambda x: x[0])
        #print("Sorted keys:", zipped)

        # Garante que o último item seja um servidor (rotaciona se necessário)
        def is_server(item):
            return item[1][1] == 'srv'
        if not is_server(zipped[-1]):
            # Rotaciona até o último ser servidor
            for shift in range(1, len(zipped)+1):
                rotated = zipped[-shift:] + zipped[:-shift]
                if is_server(rotated[-1]):
                    zipped = rotated
                    break

        assignments = []
        pending_requests = []
        for val, (idx, typ) in zipped:
            if typ == 'req':
                pending_requests.append(idx)
            elif typ == 'srv':
                for req_idx in pending_requests:
                    assignments.append((self.requisicoes[req_idx], self.servers[idx]))
                pending_requests = []
        return assignments

    def cost(self, solution, final_solution: bool = False) -> float:
        """
        Conta o número de requisições processadas com sucesso.
        Para cada (request, server) na solução, chama processar_caminho.
        Se processar_caminho retorna True, incrementa o contador de processadas.
        Retorna o negativo do número de requisições processadas (minimização no RKO).
        """
        processed_count = 0
        penalty = 0
        for req, srv in solution:
            path = self.reconstruir_caminho(self.prev, req.sensor, srv)
            if (path is None) or (path[0] != req.sensor) or (len(path) < 2):
                penalty += 1
                continue  # Não processa se não há caminho
            # Chama função externa para tentar processar a requisição
            
            processed = self.processavel(path, req.service, srv)
            if processed:
                processed_count += 1
        # Como o RKO minimiza, retornamos o negativo do número de processadas
        return -(processed_count-penalty)
    
    def processavel(self, caminho, requisicao, fog):
        tempo = 0
        for i in range(len(caminho)-1):
            vertice_atual = caminho[i] 
            vizinho = caminho[i+1]
            tempo += self.dist[vertice_atual][vizinho]
            if(tempo <= requisicao.lifetime) and (vizinho == fog):
                return True            
        return False

def run(grafo, requisicoes, fogs):
    env = FogEnv(grafo, requisicoes, fogs)
    solver = RKO.RKO(env, True)
    cost, solution, time = solver.solve(time_total=300, brkga=1, vns=1, ms=1, runs=10)
    return env.decoder(solution)
   
if __name__ == "__main__":
    instance_file = "1.txt"
    grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join("instances", instance_file))
    solution= run(grafo, requisicoes, fogs)
    print("Solução encontrada:")