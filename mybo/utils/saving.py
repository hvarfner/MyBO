import os
from os.path import join, dirname, abspath

from ax.service.ax_client import AxClient


def save_run(save_path: str, ax_client: AxClient) -> None:
    os.makedirs(dirname(save_path), exist_ok=True) 
    client_path = join(save_path + '_ax_client.json')
    results_path = join(save_path + '_run.csv')
    ax_client.save_to_json_file(client_path)
    ax_client.get_trials_data_frame().to_csv(results_path, index=False)