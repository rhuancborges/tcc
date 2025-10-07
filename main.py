import lerInstancia, CMC, os, logging, folium
from estruturas import CloudNode
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

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

if __name__=="__main__":
    percent_requisicoes_processadas = []
    percent_arcos_usados = []
    custo_total = []
    largura_banda_total = []
    

    logging.basicConfig(level=logging.INFO)
    os.makedirs("heatmaps", exist_ok=True)

    index = 0
    instance_file = f"{index}.txt"
    grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join("instances", instance_file))
    logging.info(f"Instância {index} carregada com {len(requisicoes)} requisições, {len(fogs)} nós fog e {len(sensores)} sensores.")

    req, arc, band, c = CMC.run(grafo, requisicoes, fogs, index)
    logging.info(f"Instância {index} processada: {req}% requisições, {arc}% arcos, {band} Gbps, US$ {c}.")

    criarHeatMap(fogs, sensores, index, grafo)
    logging.info(f"Heatmap salvo como heatmaps/mapa_calor_{index}.html")
    percent_requisicoes_processadas.append(req)
    percent_arcos_usados.append(arc)
    largura_banda_total.append(band)
    custo_total.append(c)

    #instancias = [f"Inst {i}" for i in range(1, 19)]
    instancias = [f"Inst {index}"]
    gerarGraficos(instancias, percent_requisicoes_processadas, percent_arcos_usados, custo_total, largura_banda_total)
    logging.info("Gráficos gerados.")