import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

n_sensores = [20, 40, 60, 100, 250, 500]
k = 1  # Índice inicial para geração de instâncias de teste

for n in n_sensores:
    subprocess.run(["python", "main.py", str(n), str(k)])
    k += 3  # Incrementa o índice para evitar sobrescrever as instâncias geradas anteriormente
    logger.info(f"Execução concluída para {n} sensores.")
