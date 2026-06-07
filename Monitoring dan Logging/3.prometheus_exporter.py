import time
import psutil
from prometheus_client import start_http_server, Gauge

# Membuat alat ukur (Gauge) untuk Prometheus
cpu_usage = Gauge('system_cpu_usage', 'Penggunaan CPU saat ini dalam persen')
ram_usage = Gauge('system_ram_usage', 'Penggunaan RAM saat ini dalam persen')
disk_usage = Gauge('system_disk_usage', 'Penggunaan Disk saat ini dalam persen')

def collect_metrics():
    while True:
        cpu_usage.set(psutil.cpu_percent(interval=1))
        ram_usage.set(psutil.virtual_memory().percent)
        disk_usage.set(psutil.disk_usage('/').percent)
        time.sleep(5)

if __name__ == '__main__':
    print("Memulai Prometheus Exporter (Hardware) di http://127.0.0.1:8001")
    start_http_server(8001)
    collect_metrics()