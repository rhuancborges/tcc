import sys, pickle
import lerInstancia, CMC, os, logging, folium, subprocess, oraculo
from estruturas import CloudNode, FogNode
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

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
    
def gerarGraficos(path):

    os.makedirs("graficos", exist_ok=True)
    image_path = "graficos"

    # Carrega os resultados
    df = pd.read_csv(path)

    # Define as cores das abordagens
    cores = {"Sem oráculo": "steelblue", "Com oráculo": "orange"}

    # 1) Percentual de requisições processadas x Instância
    plt.figure()
    for abordagem in df["Abordagem"].unique():
        dados = df[df["Abordagem"] == abordagem]
        barras = plt.bar(dados["Instância"] + (0.2 if abordagem == "Com oráculo" else -0.2),
                dados["% Requisições processadas"],
                width=0.4, label=abordagem, color=cores[abordagem])
        for bar in barras:
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # centro da barra
                altura,  # posição vertical no topo da barra
                f"{altura:.1f}",  # texto (1 casa decimal)
                ha='center', va='bottom', fontsize=6
            )
    plt.xlabel("Instância")
    plt.ylabel("% Requisições processadas")
    plt.title("Percentual de Requisições Processadas x Instância")
    plt.xticks(ticks=range(1, 19), labels=[str(i) for i in range(1, 19)])
    plt.legend()
    plt.savefig(os.path.join(image_path, "reqProcessadas.png"))
    #plt.show()

    # 2) Percentual de arcos utilizados x Instância
    plt.figure()
    for abordagem in df["Abordagem"].unique():
        dados = df[df["Abordagem"] == abordagem]
        barras = plt.bar(dados["Instância"] + (0.2 if abordagem == "Com oráculo" else -0.2),
                dados["% Arcos utilizados"],
                width=0.4, label=abordagem, color=cores[abordagem])
        for bar in barras:
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # centro da barra
                altura,  # posição vertical no topo da barra
                f"{altura:.1f}",  # texto (1 casa decimal)
                ha='center', va='bottom', fontsize=6
            )
    plt.xlabel("Instância")
    plt.ylabel("% Arcos utilizados")
    plt.title("Percentual de Arcos Utilizados x Instância")
    plt.xticks(ticks=range(1, 19), labels=[str(i) for i in range(1, 19)])
    plt.legend()
    plt.savefig(os.path.join(image_path, "arcosUtilizados.png"))
    #plt.show()

    # 3) Largura de banda total gasta x Instância
    plt.figure()
    for abordagem in df["Abordagem"].unique():
        dados = df[df["Abordagem"] == abordagem]
        barras = plt.bar(dados["Instância"] + (0.2 if abordagem == "Com oráculo" else -0.2), 
                 dados["Largura de banda total (Gbps)"],
                width=0.4, label=abordagem, color=cores[abordagem])
        for bar in barras:
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # centro da barra
                altura,  # posição vertical no topo da barra
                f"{altura:.1f}",  # texto (1 casa decimal)
                ha='center', va='bottom', fontsize=6
            )
    plt.xlabel("Instância")
    plt.ylabel("Largura de banda total (Gbps)")
    plt.title("Largura de Banda Total Gasta x Instância")
    plt.xticks(ticks=range(1, 19), labels=[str(i) for i in range(1, 19)])
    plt.legend()
    plt.savefig(os.path.join(image_path, "larguraBandaTotal.png"))
    #plt.show()

    # 4) Custo total gasto x Instância
    plt.figure()
    for abordagem in df["Abordagem"].unique():
        dados = df[df["Abordagem"] == abordagem]
        barras = plt.bar(dados["Instância"] + (0.2 if abordagem == "Com oráculo" else -0.2), 
                 dados["Custo total (US$)"],
                width=0.4, label=abordagem, color=cores[abordagem])
        for bar in barras:
            altura = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # centro da barra
                altura,  # posição vertical no topo da barra
                f"{altura:.1f}",  # texto (1 casa decimal)
                ha='center', va='bottom', fontsize=6
            )
    plt.xlabel("Instância")
    plt.ylabel("Custo total (US$)")
    plt.title("Custo Total Gasto x Instância")
    plt.xticks(ticks=range(1, 19), labels=[str(i) for i in range(1, 19)])
    plt.legend()
    plt.savefig(os.path.join(image_path, "custoTotalGasto.png"))
    #plt.show()

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
    #n_sensores_list = [20, 40, 60, 100, 250, 500]
    n_sensores_list = [500]
    k = 17
    # Estrutura de dados
    colunas = [
        "Instância",
        "Abordagem",
        "Tempo de execução (s)",
        "Nº de sensores",
        "Nº de servidores",
        "Total de requisições",
        "Requisições processadas",
        "% Requisições processadas",
        "% Arcos utilizados",
        "Largura de banda total (Gbps)",
        "Custo total (US$)"
    ]

    if not os.path.exists("resultados.csv"):
        pd.DataFrame(columns=colunas).to_csv("resultados.csv", index=False)

    for n_sensores in n_sensores_list:
        # PARTE 1 - Treinar o RKO
        pasta_rko = os.path.join(pasta_saida, "rko_training")
        os.makedirs(pasta_rko, exist_ok=True)

        training_requests = []
        logger.info("Iniciando geração de instâncias de treinamento para o RKO...")
        if not os.path.exists(os.path.join(pasta_rko, f"oracle_{n_sensores}_sensors.pkl")):
            for i in range(1, 11):
                instance_ID = f"{i}_training_{n_sensores}"
                if not os.path.exists(os.path.join(pasta_rko, f"{instance_ID}.txt")):
                    subprocess.run(["python", "gerarInstancia.py", str(n_sensores), instance_ID, pasta_rko])
                logger.info(f"Instância {instance_ID} gerada.")

            for i in range(1, 11):
                instance_ID = f"{i}_training_{n_sensores}"
                grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join(pasta_rko, f"{instance_ID}.txt"))
                logger.info(f"Instância {instance_ID} carregada com {len(requisicoes)} requisições.")
                if n_sensores <= 100:
                    percent = 0.01
                else:
                    percent = 0.001
                reqs = construirAmostra(int(1000/(len(requisicoes)*percent)),requisicoes)
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

    
        # PARTE 2 - GERAR INSTÂNCIAS DE TESTE
        for i in range(k, k+2):
            instance_ID = str(i)
            if not os.path.exists(os.path.join(pasta_saida, f"{instance_ID}.txt")):
                subprocess.run(["python", "gerarInstancia.py", str(n_sensores), instance_ID, pasta_saida])
                logger.info(f"Instância {instance_ID} gerada.")
            else:
                logger.info(f"Instância {instance_ID} já existe.")

        # PARTE 3 - EXECUTAR INSTÂNCIAS E GERAR GRÁFICOS E HEATMAPS
            
        # Exemplo de inserção de dados após rodar uma instância
        for i in range(k, k+2):
            instance_file = f"{i}.txt"
            for abordagem in ["Sem oráculo", "Com oráculo"]:
                grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join("instances", instance_file))
                inicio = time.time()
                if abordagem == "Sem oráculo":
                    req, req_p, arc, band, c, grafo = CMC.run(grafo, requisicoes, oracle, i, usaOraculo=False)
                else:
                    req, req_p, arc, band, c, grafo = CMC.run(grafo, requisicoes, oracle, i, usaOraculo=True)
                fim = time.time()
                criarHeatMap(fogs, sensores, i, grafo)
                logger.info(f"Heatmap para a instância {i} criado.")
                resultados = [
                    i,
                    abordagem,
                    round(fim - inicio, 3),
                    len(sensores),  # Exemplo: número de sensores
                    len(fogs),   # Exemplo: número de servidores
                    len(requisicoes), # Total de requisições
                    req, # Processadas
                    req_p, # Processadas
                    arc,
                    band,
                    c
                ]
                df = pd.DataFrame([resultados], columns=colunas)
                df.to_csv("resultados.csv", mode='a', header=False, index=False)
                logger.info(f"Resultados para a instância {i} com abordagem '{abordagem}' salvos.")
        k += 3  # Incrementa o índice para evitar sobrescrever as instâncias geradas anteriormente
    gerarGraficos("resultados.csv")

