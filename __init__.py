import logging, os
import subprocess, pickle, CMC, lerInstancia

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    instance_file = os.path.join("instances/rko_training", "oracle_20_sensors.pkl")
    with open(instance_file, "rb") as f:
        oracle = pickle.load(f)
    grafo, requisicoes, fogs, sensores = lerInstancia.run(os.path.join("instances", "1.txt"))
    dist, prev = CMC.floyd_warshall(grafo)
    caminho = CMC.reconstruir_caminho(prev, sensores[0], oracle[sensores[0]]["waste"])
    print(grafo.adj[sensores[0]])
    for v in grafo.adj[sensores[0]]:
        print(grafo.adj[v[0]])
        for x in grafo.adj[v[0]]:
            print(grafo.adj[x[0]])
    print(sensores[0], "->", oracle[sensores[0]]["waste"])
    print(prev[sensores[0]])
    print("Caminho reconstruído:", caminho)
