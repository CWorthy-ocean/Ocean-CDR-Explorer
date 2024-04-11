import os

import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

USER = os.environ['USER']
try:
    TMPDIR = os.environ['TMPDIR']
except KeyError:   
    print('TMPDIR not set; set TMPDIR environment variable')
    raise

path_to_here = os.path.dirname(os.path.realpath(__file__))


def get_ClusterClient(memory="25GB", project='NCGD0011', walltime='06:00:00'):
    """return client and cluster"""
    cluster = PBSCluster(
        cores=1,
        memory=memory,
        processes=1,
        queue='casper',
        local_directory=f'/glade/work/{USER}/dask-workers',
        log_directory=f'/glade/work/{USER}/dask-workers',
        resource_spec=f'select=1:ncpus=1:mem={memory}',
        project=project,
        walltime=walltime,
        interface='ext', # try 'lo', 'mgt', 'ext', 'eno2'
    )

    jupyterhub_server_name = os.environ.get('JUPYTERHUB_SERVER_NAME', None)
    dashboard_link = 'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    if jupyterhub_server_name:
        dashboard_link = (
            'https://jupyterhub.hpc.ucar.edu/stable/user/'
            + '{USER}'
            + f'/{jupyterhub_server_name}/proxy/'
            + '{port}/status'
        )
    dask.config.set({'distributed.dashboard.link': dashboard_link})
    client = Client(cluster)
    return cluster, client
