import pickle
import sys
from lerInstancia import run
import os, logging
from estruturas import Grafo, Sensor, FogNode, CloudNode, Request, Service
import lerInstancia


# FUNÇÃO FLOYD-WARSHALL - Função para calcular todos os caminhos do grafo, considerando o tempo como peso
# dist[u][v] - "Qual o tempo gasto para chegar em v no caminho que parte de u?"
# prev[u][v] - "Qual o nó anterior ao v no caminho que parte de u?"
def floyd_warshall(grafo):
    dist = {u: {v: float('inf') for v in grafo.adj.keys()} for u in grafo.adj.keys()}
    prev = {u: {v: None for v in grafo.adj.keys()} for u in grafo.adj.keys()}

    for u in grafo.adj.keys():
        dist[u][u] = 0
        for v, _, _, t in grafo.adj[u]:
            dist[u][v] = t
            prev[u][v] = u

    for k in grafo.adj.keys():
        for i in grafo.adj.keys():
            for j in grafo.adj.keys():
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]
    return dist, prev

# FUNÇÃO RECONSTRUIR CAMINHO
# Dada uma origem e um destino, a função utiliza a estrutura prev para reconstruir o caminho desde o destino até a origem
def reconstruir_caminho(prev, origem, destino):
    caminho = []
    atual = destino
    while atual is not None:
        caminho.append(atual)
        atual = prev[origem][atual]
    caminho.reverse()
    return caminho

# FUNÇÃO PROCESSAR CAMINHO
# Dado um caminho no grafo e uma requisição, essa função tenta alocar a requisição no nó Fog destino
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
        gasto_req = requisicao.number_of_bits/((10**9)*grafo.adj_[f"({vertice_atual},{vizinho})"].tempo) # Converte de bits para Gbps
        tempo += dist[vertice_atual][vizinho]
        #print(f"Largura de banda aresta ({vertice_atual},{vizinho}): {grafo.adj_[f'({vertice_atual},{vizinho})'].largura_banda} Gbps")
        #print(f"Gasto requisição: {gasto_req} Gbits")
        if (gasto_req <= grafo.adj_[f"({vertice_atual},{vizinho})"].largura_banda):
          if(tempo <= requisicao.lifetime):
            selecionado = vizinho
            band_tot += gasto_req
            arcos.append((f"({vertice_atual},{selecionado})", gasto_req))
            custo += gasto_req*grafo.adj_[f"({vertice_atual},{selecionado})"].custo
          else:
            motivo += f"Requisição estourou o tempo de vida em {vizinho}\n"
        else:
            motivo += f"Requisição estourou a largura de banda indo de {vertice_atual} para {vizinho}\n"

    # Se não encontrou vizinho válido
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
    motivo += "Requisição percorreu todo o caminho e não foi processada\n"
    return selecionado, False, arcos, band_tot, custo, motivo

# FUNÇÃO LER INSTÂNCIA
# Chamada para cada instância gerada
# index - índice das instâncias e também dos arquivos de log e mapa
    # Facilita a correspondência entre os arquivos
def run(grafo, requisicoes, oracle, index, usaOraculo):
    # Estrutura para controlar a disponibilidade de Processamento e Memória para os nós fog
    # i: (v,p,m) - Ao instante "i", uma quantidade "p" de processamento e "m" de memória volta a estar disponível no nó "v"
    temporal = {i: [] for i in range(1, 1001)}
    

    # Estrutura para controlar a disponibilidade de largura de banda para as arestas
    # i: (aresta, band) - Ao instante i, uma quantidade "band" de largura de banda volta a estar disponível na aresta "aresta"
    temporal_arestas = {i: [] for i in range(1, 1002)}

    tot_req = 0
    quant_req = 0
    set_arcos = []
    quant_band = 0
    quant_custo = 0

    # Chama a função Floyd-Warshall para já deixar pré-processado todos os caminhos do grafo
    dist, prev = floyd_warshall(grafo)

    logger.info(f"CMC - Iniciando processamento da instância {index} com {len(requisicoes)} requisições...")
    trocouInstante = False
    instante_anterior = -1
    # Começa a ler as requisições
    for req in requisicoes: 
        instante = req.instante

        # Faz as atualizações no grafo uma vez por instante
        for (v, p, m) in temporal[instante]:
            v.processing_capacity += p
            v.memory_capacity += m
        for aresta, band in temporal_arestas[instante]:
            aresta.largura_banda += band

        temporal[instante] = []
        temporal_arestas[instante] = []
       
        tot_req += 1
        sensor = req.sensor
        logger.info(f"CMC - Processando requisição {tot_req} do instante {instante}...")
        

        # Para um dado sensor, seleciona todos os possíveis Nós fog a serem alcançados, ordenados crescentemente pelo tempo de alcance
        tempo_dict = dist[sensor]
        candidatos = sorted([(v, t) for v, t in tempo_dict.items() if isinstance(v, FogNode) and (t != float('inf'))], key=lambda x: x[1])
        
        quant_testados = 1 # Variável para contar o número de caminhos testados até processar uma requisição
        candidatos += sorted([(v, t) for v, t in tempo_dict.items() if isinstance(v, CloudNode) and (t != float('inf'))], key=lambda x: x[1])
        
        if usaOraculo:
            fog_oracle = oracle[sensor][req.service.id]
            candidatos = [(fog_oracle, dist[sensor][fog_oracle])] + candidatos
            logger.info("CMC - Adicionando previsão do oráculo como primeira tentativa de destino")
            quant_testados -= 1
        else:
            logger.info("CMC - Segue apenas com o Caminho Mais Curto")

        # Varre a lista de Nós Fog candidatos, selecionando um por vez como destino do caminho
        for destino, _ in candidatos:

            if (index <= 15):
                if (quant_testados > 3):
                    logger.info(f"CMC - Testou-se o limite de 3 caminhos\n")
                    destino = candidatos[-3][0]
            else:
                if (quant_testados > 6):
                    logger.info(f"CMC - Testou-se o limite de 6 caminhos\n")
                    destino = candidatos[-3][0]

            # Reconstrói o caminho
            caminho = reconstruir_caminho(prev, sensor, destino)
            
            # Tenta processar o caminho
            selecionado, processou, arcos, band, c, motivo = processar_caminho(caminho, req.service, dist, destino, grafo)

            # Entra no if apenas se a requisicao foi processada no caminho passado como parâmetro
            if processou:
                if quant_testados == 1:
                    logger.info(f"Requisicao {tot_req} processada no primeiro caminho testado!\n")
                logger.info(f"{quant_testados} caminhos testados | Caminho selecionado: {caminho} | Nó selecionado: {selecionado}\n")
                
                if(isinstance(selecionado, FogNode)):
                    quant_req += 1  # Número de requisicoes incrementado
                
                set_arcos += arcos # Conjunto de arcos percorridos atualizado
                
                for arco, larg in arcos:
                    grafo.adj_[arco].quantidade_uso += 1 # Uso de arcos incrementado
                    grafo.adj_[arco].largura_banda -= larg
                    if(instante+1)<=1000:
                        # Torna a quantidade de largura de banda da requisição "indisponível" à aresta apenas por um instante (visto que o tempo de propagação não é tão significativo)
                        temporal_arestas[instante+1].append((grafo.adj_[arco], larg))
                
                quant_band += band # Largura de banda total incrementada
                quant_custo += c # Custo total incrementado

                # Adicionando na estrutura temporal quando as quantidades de processamento e memória estaraõ disponíveis novamente para o nó em que foi processada a requisição
                #tX: [(Nó, processing_demand, memory_demand)]
                if (instante+req.service.lifetime+1) <= 1000:
                    temporal[instante+req.service.lifetime+1].append((selecionado, req.service.processing_demand, req.service.memory_demand))

                break # Uma vez processada a requisição, não se testa outros caminhos
            else:
                logger.info(f"Requisicao {tot_req} foi rejeitada porque {motivo}\n")
                if(isinstance(destino, CloudNode)):
                    break

            quant_testados+=1 # Se não foi processada, incrementa a quantidade de caminhos testada

    set_arcos = set(set_arcos)
    quant_arcos = len(set_arcos)   
    return quant_req, (quant_req/tot_req)*100.0, (quant_arcos/grafo.n_arestas)*100.0, quant_band, quant_custo, grafo


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__=="__main__":
    instance_file = "1_training_250.txt"
    grafo, requisicoes, fogs, sensores =  lerInstancia.run(os.path.join("instances", "rko_training", instance_file))
    """  with open(os.path.join("instances", "rko_training",  f"oracle_{20}_sensors.pkl"), "rb") as f:
            oracle = pickle.load(f)
    print(oracle)  
    for fog in fogs:
        caminho = reconstruir_caminho(floyd_warshall(grafo)[1], sensores[1], fog)
        print(f"{caminho} -> {caminho[0] == sensores[0]}")
    sys.exit(0) 
    """
    
    
    #req, perc_req, perc_arcos, largura_banda, custo, _ = run(grafo, requisicoes, None, 1, False)
    with open(os.path.join("instances", "rko_training",  f"oracle_{500}_sensors.pkl"), "rb") as f:
           oracle = pickle.load(f)
   
    #req1, perc_req1, perc_arcos1, largura_banda1, custo1, _ = run(grafo, requisicoes, oracle, 1, True)
    print("SEM ORÁCULO")
    print(f"Número de requisições processadas: {req}")
    print(f"Percentual de requisições processadas: {perc_req}%")
    print(f"Percentual de arcos usados: {perc_arcos}%")
    print(f"Largura de banda total usada: {largura_banda} Gbps")
    print(f"Custo total: US$ {custo}")
    print("\nCOM ORÁCULO")
    print(f"Número de requisições processadas: {req1}")
    print(f"Percentual de requisições processadas: {perc_req1}%")
    print(f"Percentual de arcos usados: {perc_arcos1}%")
    print(f"Largura de banda total usada: {largura_banda1} Gbps")
    print(f"Custo total: US$ {custo1}")
   