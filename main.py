import sys, pickle
import lerInstancia, CMC, os, logging, folium, subprocess, oraculo
from estruturas import CloudNode, FogNode
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np

def criarHeatMap(fogs, sensores, index, grafo):
    # Inicializa o mapa centralizado na média das coordenadas
    avg_lat = sum(node.latitude for node in fogs) / len(fogs)
    avg_lon = sum(node.longitude for node in fogs) / len(fogs)
    mapa = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)

    # Constrói os dados para o mapa de calor
    heat_data = [
        [node.latitude, node.longitude, node.requisicoes]
        for node in fogs if node.requisicoes > 0
    ]

    # Adiciona camada de calor
    HeatMap(heat_data).add_to(mapa)

    # Desenha as arestas do grafo. Apenas as arestas "usadas" (por onde passaram requisições) e excluindo as arestas que se conectam aos Nós Cloud
    for arco in grafo.adj_.values():
        u = arco.u
        v = arco.v
        uso = arco.quantidade_uso
        if uso > 0 and not (isinstance(u, CloudNode) or isinstance(v, CloudNode)):
            folium.PolyLine(
                locations=[[float(u.latitude), float(u.longitude)], [float(v.latitude), float(v.longitude)]],
                color="orange",
                weight=3,
                opacity=0.8,
                tooltip=f"{u.ID} → {v.ID}: {uso} usos"
            ).add_to(mapa)

    # Desenha os Nós Fog
    for node in fogs:
        folium.Marker(
            location=[node.latitude, node.longitude],
            popup=f"Fog {node.ID} - {node.requisicoes} reqs",
            icon=folium.Icon(color="blue" if node.requisicoes > 0 else "gray")
        ).add_to(mapa)

    # Desenha os Nós Sensores
    for node in sensores:
        folium.Marker(
            location=[node.latitude, node.longitude],
            popup=f"Sensor {node.ID}",
            icon=folium.Icon(color="red")
        ).add_to(mapa)

    # Salvar como HTML interativo
    mapa.save(f"heatmaps/mapa_calor_{index}.html")
    
def gerarGraficos(instancias, percent_requisicoes_processadas, percent_arcos_usados, custo_total, largura_banda_total):
     # Figura geral com 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análise Comparativa entre Instâncias", fontsize=16, fontweight='bold')

    # Gráfico 1: % Requisições Processadas
    axs[0, 0].bar(instancias, percent_requisicoes_processadas, color='mediumseagreen')
    axs[0, 0].set_title('% Requisições Processadas')
    axs[0, 0].set_ylabel('%')
    axs[0, 0].set_ylim(0, 100)

    # Gráfico 2: % Arcos Usados
    axs[0, 1].bar(instancias, percent_arcos_usados, color='steelblue')
    axs[0, 1].set_title('% Arcos Utilizados')
    axs[0, 1].set_ylabel('%')
    axs[0, 1].set_ylim(0, 100)

    # Gráfico 3: Custo Total
    axs[1, 0].plot(instancias, custo_total, marker='o', linestyle='-', color='indianred')
    axs[1, 0].set_title('Custo Total Gasto')
    axs[1, 0].set_ylabel('US$')
    axs[1, 0].grid(True)

    # Gráfico 4: Largura de Banda Total
    axs[1, 1].plot(instancias, largura_banda_total, marker='s', linestyle='--', color='darkorange')
    axs[1, 1].set_title('Largura de Banda Total Gasta')
    axs[1, 1].set_ylabel('Gbps')
    axs[1, 1].grid(True)

    # Ajustar layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def construirAmostra(taxa, requisicoes):
    rebuilt_requests = []
    for instante in range(1, 1001):
        if instante % taxa == 0:
            reqs = [req for req in requisicoes if req.instante == instante]
            rebuilt_requests.append(np.random.choice(reqs))
    return rebuilt_requests

def construirOraculoED(requests, sensors, fogs):
    n_fogs = len(fogs)
    oracle = {i: {"waste": [0 for i in range(n_fogs)], "air": [0 for i in range(n_fogs)], "camera": [0 for i in range(n_fogs)]} for i in sensors}
    for req, fog in requests:
        sensor_id = req.sensor
        service_type = req.service.id
        if service_type in oracle[sensor_id]:
            oracle[sensor_id][service_type][fog.ID] += 1
    for sensor_id in oracle:
        for service_type in oracle[sensor_id]:
            list = oracle[sensor_id][service_type]
            index_max = list.index(max(list))
            oracle[sensor_id][service_type] = fogs[index_max]
    return oracle

pasta_saida = "instances"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__=="__main__":
    n_sensores = int(sys.argv[1])
    k = int(sys.argv[2])  # Índice inicial para geração de instâncias de teste


    # PARTE 1 - Treinar o RKO
    pasta_rko = os.path.join(pasta_saida, "rko_training")
    os.makedirs(pasta_rko, exist_ok=True)

    training_requests = []
    logger.info("Iniciando geração de instâncias de treinamento para o RKO...")
    if not os.path.exists(os.path.join(pasta_rko, f"oracle_{n_sensores}_sensors.pkl")):
        for i in range(10):
            instance_ID = str(i) + "_training"
            if not os.path.exists(os.path.join(pasta_rko, f"{instance_ID}.txt")):
                subprocess.run(["python", "gerarInstancia.py", str(n_sensores), instance_ID, pasta_rko])
            logger.info(f"Instância {instance_ID} gerada.")

        for i in range(10):
            instance_ID = str(i) + "_training"
            grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join(pasta_rko, f"{instance_ID}.txt"))
            logger.info(f"Instância {instance_ID} carregada com {len(requisicoes)} requisições.")
            reqs = construirAmostra(int(1000/(len(requisicoes)*0.01)), requisicoes)
            logger.info(f"Instância {instance_ID} reduzida para {len(reqs)} requisições para treinamento.")
            logger.info(f"Executando RKO na instância {instance_ID}...")
            predictor = oraculo.run(grafo, reqs, fogs)
            training_requests.extend(predictor)

        oracle = construirOraculoED(training_requests, sensores, fogs)
        with open(os.path.join(pasta_rko, f"oracle_{n_sensores}_sensors.pkl"), "wb") as f:
            pickle.dump(oracle, f)
        logger.info(f"Oráculo construído e salvo em {os.path.join(pasta_rko, f'oracle_{n_sensores}_sensors.pkl')}.")
    else:
        logger.info(f"Oráculo já existe em {os.path.join(pasta_rko, f'oracle_{n_sensores}_sensors.pkl')}. Carregando oráculo...")
        with open(os.path.join(pasta_rko, f"oracle_{n_sensores}_sensors.pkl"), "rb") as f:
            oracle = pickle.load(f)

    sys.exit(0)
    
    # PARTE 2 - GERAR INSTÂNCIAS DE TESTE
    for i in range(k, k+3):
        instance_ID = str(i)
        if not os.path.exists(os.path.join(pasta_saida, f"{instance_ID}.txt")):
            subprocess.run(["python", "gerarInstancia.py", str(n_sensores), instance_ID, pasta_saida])
            logger.info(f"Instância {instance_ID} gerada.")
        else:
            logger.info(f"Instância {instance_ID} já existe.")

    # PARTE 3 - EXECUTAR INSTÂNCIAS E GERAR GRÁFICOS E HEATMAPS
    percent_requisicoes_processadas = []
    percent_arcos_usados = []
    custo_total = []
    largura_banda_total = []
   
    os.makedirs("heatmaps", exist_ok=True)

    index = 0
    instance_file = f"{index}.txt"
    grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join("instances", instance_file))
    logger.info(f"Instância {index} carregada com {len(requisicoes)} requisições, {len(fogs)} nós fog e {len(sensores)} sensores.")

    req, arc, band, c = CMC.run(grafo, requisicoes, oracle, index)
    logger.info(f"Instância {index} processada: \n\t{len(requisicoes)} requisições (total)\n\t{req:.2f}% requisições processadas\n\t{arc}% arcos\n\t{band} Gbps\n\tUS$ {c}.")
    
    criarHeatMap(fogs, sensores, index, grafo)
    logger.info(f"Heatmap salvo como heatmaps/mapa_calor_{index}.html")
    
    percent_requisicoes_processadas.append(req)
    percent_arcos_usados.append(arc)
    largura_banda_total.append(band)
    custo_total.append(c)

    #instancias = [f"Inst {i}" for i in range(1, 19)]
    instancias = [f"Inst {index}"]
    gerarGraficos(instancias, percent_requisicoes_processadas, percent_arcos_usados, custo_total, largura_banda_total)
    logger.info("Gráficos gerados.")