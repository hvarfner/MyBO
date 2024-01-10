import os
from argparse import ArgumentParser

import pandas as pd
from ax.service.ax_client import AxClient
from mybo.interface import _get_client, append_to_client



if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Append data to client',
        description="Provide the path to the data, and the (possibly empty) client to append it to.",
    )

    parser.add_argument('-c', '--client')
    parser.add_argument('-d', '--data')
    args = parser.parse_args()

    print(f"Trying to append {args.data} to {args.client}.")
    client = _get_client(args.client)
    data_df = pd.read_csv(args.data)

    append_to_client(client, data_df, path=args.client)
