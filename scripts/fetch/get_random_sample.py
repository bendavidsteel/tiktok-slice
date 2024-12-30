import asyncio
import collections
from concurrent.futures import ThreadPoolExecutor
import datetime
import io
import itertools
import json
import logging
import os
import subprocess
import time
import traceback

import asyncssh
import brotli
import certifi
from dask.distributed import Client as DaskClient
from dask.distributed import LocalCluster as DaskLocalCluster
from dask.distributed import as_completed as dask_as_completed
from dask_cloudprovider.aws import FargateCluster as DaskFargateCluster
from dask_jobqueue import SLURMCluster as DaskSLURMCluster
import dotenv
import numpy as np
import polars as pl
import pycurl
import randmac
from random_user_agent import user_agent as rug_user_agent
from random_user_agent import params as rug_params
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from dask_extensions import UnreliableSSHCluster
from map_funcs import _amap
from setup_pis import change_mac_addresses, get_hosts_with_retries, start_wifi_connections, restart_down_slurm_nodes, stop_stale_workers

def ensure_wifi_connection(wlan_username, wlan_password, force_start=False, password=None):
    num_tries = 0
    max_tries = 3
    # TODO use google dns64 server to allow ipv6 connections
    # https://developers.google.com/speed/public-dns/docs/using#linux
    # https://developers.google.com/speed/public-dns/docs/dns64
    # prepend = ["sudo"]
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if r.returncode == 1:
        prepend = ["echo", password, "|", "sudo", "-S"]
    else:
        prepend = ["sudo"]

    def create_wifi_connection():
        command = prepend + ["nmcli", "con", "add", "type", "wifi", "con-name", '"eduroam"', "ifname", '"wlan0"', "ssid", '"eduroam"', "wifi-sec.key-mgmt", '"wpa-eap"', "802-1x.identity", wlan_username, "802-1x.password", wlan_password, "802-1x.system-ca-certs", '"yes"', "802-1x.eap", '"peap"', "802-1x.phase2-auth", '"mschapv2"']
        r = subprocess.run(command, capture_output=True)
        assert r.returncode == 0, f"Error creating connection: {r.stderr}"

    def get_wifi_connections():
        try:
            r = subprocess.run(["nmcli", "con"], capture_output=True)
        except Exception as e:
            if e.stderr == "Error: NetworkManager is not running.":
                r = subprocess.run(["sudo", "systemctl", "start", "NetworkManager"], capture_output=True)
                r = subprocess.run(["nmcli", "con"], capture_output=True)
            else:
                raise
        connections = r.stdout.decode('utf-8').split('\n')
        return connections

    exceptions = []
    while num_tries < max_tries:
        num_tries += 1
        try:
            # check if wifi connection exists
            connections = get_wifi_connections()
            
            eduroam_lines = [c for c in connections if c.startswith('eduroam')]
            if len(eduroam_lines) == 0:
                # create wifi connection
                create_wifi_connection()
                connections = get_wifi_connections()
            elif len(eduroam_lines) > 1:
                # delete duplicate connections
                # find if any all valid
                valid_eduroam_line = None
                for eduroam_line in eduroam_lines:
                    reqs = [' wifi ', ' wlan0 ']
                    if all(req in eduroam_line for req in reqs):
                        valid_eduroam_line = eduroam_line
                if valid_eduroam_line:
                    invalid_eduroam_lines = [eduroam_line for eduroam_line in eduroam_lines if eduroam_line != valid_eduroam_line]
                    # get uuids
                    uuids = [line.split(' ')[1] for line in invalid_eduroam_lines]
                    # delete invalid connections
                    for uuid in uuids:
                        r = subprocess.run(prepend + ["nmcli", "con", "delete", uuid], capture_output=True)
                else:
                    # delete all connections
                    for eduroam_line in eduroam_lines:
                        uuid = eduroam_line.split(' ')[1]
                        r = subprocess.run(prepend + ["nmcli", "con", "delete", uuid], capture_output=True)
                    # create wifi connection
                    create_wifi_connection()
                connections = get_wifi_connections()

            elif len(eduroam_lines) == 1:
                eduroam_line = eduroam_lines[0]
                reqs = [' wifi ', ' wlan0 ']
                if not all(req in eduroam_line for req in reqs):
                    # delete existing connection
                    r = subprocess.run(prepend + ["nmcli", "con", "delete", "eduroam"], capture_output=True)
                    # create wifi connection
                    create_wifi_connection()
                    connections = get_wifi_connections()
            assert any(c.startswith('eduroam') for c in connections), "No eduroam connection"
            eduroam_lines = [c for c in connections if c.startswith('eduroam')]
            assert len(eduroam_lines) == 1, "Duplicate eduroam connections"
            eduroam_line = eduroam_lines[0]
            if not ('wifi' in eduroam_line and 'wlan0' in eduroam_line):
                raise Exception("Eduroam connection not setup correctly")
            # check if connection is up
            r = subprocess.run(["nmcli", "-f", "GENERAL.STATE", "con", "show", "eduroam"], capture_output=True)
            if 'activated' not in r.stdout.decode('utf-8') or force_start:
                # start wifi connection
                r = subprocess.run(prepend + ["nmcli", "connection", "up", "eduroam"], capture_output=True)
        except Exception as e:
            # delete connection and try again
            exceptions.append(e)
            r = subprocess.run(prepend + ["nmcli", "con", "delete", "eduroam"], capture_output=True)
        else:
            break
    else:
        raise Exception(f"Failed to create wifi connection: {exceptions}")
    
def change_mac_address(wlan_username, wlan_password, password=None):
    random_mac = str(randmac.RandMac())
    # ensure eduroam connection exists
    ensure_wifi_connection(wlan_username, wlan_password, password=password)

    # prepend = ["sudo"]
    r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if r.returncode == 1:
        prepend = ["echo", password, "|", "sudo", "-S"]
    else:
        prepend = ["sudo"]
    change_mac_cmd = prepend + ["nmcli", "con", "modify", "--temporary", "eduroam", "802-11-wireless.cloned-mac-address", random_mac]
    r = subprocess.run(" ".join(change_mac_cmd), capture_output=True, shell=True)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, " ".join(change_mac_cmd), stderr=r.stderr, output=r.stdout)
    ensure_wifi_connection(wlan_username, wlan_password)
    r = subprocess.run(prepend + ["nmcli", "connection", "up", "eduroam"], capture_output=True)
    

def thread_map(*args, function=None, num_workers=10):
    assert function is not None, "function must be provided"
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(function, *args))


def wait_until(condition, interval=0.1, timeout=1, *args):
    start = time.time()
    while not condition(*args) and time.time() - start < timeout:
        time.sleep(interval)
    if time.time() - start >= timeout:
        raise TimeoutError("Timed out waiting for condition")


class ClusterManager:
    def __init__(self):
        self.potential_usernames = [
            # most reliable batch
            'hoare', 'miacli', 'fred', 'frances',
            'geoffrey', 'edmund', 'marvin', 'barbara',
            'cook', 'goldwasser', 'milner', 'conway',
            'hemming', 'lee',
            'juris', 'floyd', 'lovelace', 'edwin',
            'neumann', 'beauvoir', 'satoshi', 'putnam', 
            'shannon', 'chowning',
            'tegmark', 'hanson', 'chomsky', 'keynes',
            # next most reliable
            'edward', 'buterin',
            'arendt', 'chan', 'sutskever', 'herbert',
            'mordvintsev',
            # to set up
            'edsger', 'fernando', 'rivest', 'tarjan', 'turing',
            # unreliable
            'ivan'
        ]
        self.max_latency = 5

    async def get_hosts(self):
        self.hosts, usernames = await get_hosts_with_retries(self.potential_usernames, max_tries=2, progress_bar=True, timeout=self.max_latency)
        self.connect_options = self.get_connect_options(usernames)
        return self.hosts, self.connect_options

    async def stop_stale_workers(self):
        return await stop_stale_workers(self.hosts, self.connect_options, timeout=self.max_latency)
    
    async def start_wifi_connections(self):
        self.hosts, self.connect_options = await start_wifi_connections(self.hosts, self.connect_options, progress_bar=True, timeout=self.max_latency)
        return self.hosts, self.connect_options
    
    async def change_mac_addresses(self):
        await change_mac_addresses(self.hosts, self.connect_options, interface='wlan0', progress_bar=True)

    async def restart_down_slurm_nodes(self):
        await restart_down_slurm_nodes(self.hosts, self.connect_options, progress_bar=True)

    def get_connect_options(self, usernames):
        raspi_password = os.environ['RASPI_PASSWORD']
        return [dict(username=un, password=raspi_password, known_hosts=None) for un in usernames]


class DaskCluster:
    def __init__(self, cluster_type, manager, worker_nthreads=1, worker_cpu=256, worker_mem=512):
        self.cluster_type = cluster_type
        self.manager = manager
        self.worker_nthreads = worker_nthreads
        self.worker_cpu = worker_cpu
        self.worker_mem = worker_mem

    async def __aenter__(self):
        subprocess.run('ulimit -n 100000', shell=True, capture_output=True)

        if self.cluster_type == 'fargate':
            self.cluster = DaskFargateCluster(
                fargate_spot=True,
                image="daskdev/dask:latest-py3.10", 
                environment={'EXTRA_PIP_PACKAGES': 'httpx==0.27.0 brotlipy==0.7.0 tqdm==4.66.2 lz4==4.3.3 msgpack==1.0.8 toolz==0.12.1'},
                worker_cpu=self.worker_cpu,
                worker_nthreads=self.worker_nthreads,
                worker_mem=self.worker_mem,
                aws_access_key_id=os.environ['AWS_ACCESS_KEY'],
                aws_secret_access_key=os.environ['AWS_SECRET_KEY'],
                cluster_arn=os.environ['ECS_CLUSTER_ARN'],
                scheduler_task_definition_arn=os.environ['SCHEDULER_TASK_DEFINITION_ARN'],
                worker_task_definition_arn=os.environ['WORKER_TASK_DEFINITION_ARN'],
                execution_role_arn=os.environ['EXECUTION_ROLE_ARN'],
                task_role_arn=os.environ['TASK_ROLE_ARN'],
                security_groups=[os.environ['SECURITY_GROUP_ID']],
                skip_cleanup=True,
                region_name='ca-central-1'
            )
        elif self.cluster_type == 'local':
            self.cluster = DaskLocalCluster()
        elif self.cluster_type == 'ssh':
            print("Finding hosts...")
            hosts, connect_options = await self.manager.get_hosts()
            
            await self.manager.stop_stale_workers()
            print("Starting wifi connections...")
            hosts, connect_options = await self.manager.start_wifi_connections(hosts, connect_options)

            # append client/scheduler
            remote_python='~/ben/tiktok/venv/bin/python'
            scheduler_password = os.environ['SCHEDULER_PASSWORD']
            all_hosts = ['localhost'] + hosts
            all_connect_options = [dict(username='bsteel', password=scheduler_password)] + connect_options
            max_tries = 3
            num_tries = 0
            while num_tries < max_tries:
                num_tries += 1
                try:
                    # TODO use the task stream plot at the cluster status page to diagnose issues
                    self.cluster = await asyncio.wait_for(UnreliableSSHCluster(
                        all_hosts,
                        connect_options=all_connect_options,
                        worker_options={ 'nthreads': self.worker_nthreads },
                        remote_python=remote_python,
                        asynchronous=True
                    ), timeout=120)
                except asyncio.TimeoutError as ex:
                    # reboot workers to get rid of stale connections
                    # _, connect_options = await reboot_workers(hosts, connect_options, timeout=max_latency)
                    # potential_usernames = [co['username'] for co in connect_options]
                    # hosts, usernames = await get_hosts_with_retries(potential_usernames, max_tries=2, progress_bar=True, timeout=max_latency)
                    # connect_options = [dict(username=un, password=raspi_password, known_hosts=None) for un in usernames]
                    print("Timed out creating cluster, trying again...")
                except asyncssh.misc.ChannelOpenError:
                    pass # simply retry
                    print("Channel open error, trying again...")
                else:
                    break
            else:
                raise asyncio.TimeoutError("Timed out creating cluster...")
        elif self.cluster_type == 'slurm':
            remote_python='~/tiktok/venv/bin/python'
            account = 'bsteel'
            # TODO could run this via slurm
            hosts, connect_options = await self.manager.get_hosts()
            await self.manager.start_wifi_connections()
            await self.manager.restart_down_slurm_nodes()

            host_address = subprocess.check_output(['hostname', '-I']).decode().strip()
            worker_network_interface = 'eth0'
            scheduler_network_interface = 'eno1'
            num_workers = 37
            self.cluster = DaskSLURMCluster(
                queue='debug',
                account=account,
                cores=4,
                processes=1,
                interface=worker_network_interface,
                memory="3200 MB",
                n_workers=num_workers, # TODO ensure this gives us max workers
                worker_extra_args=['--worker-port', '9000'],
                walltime='00:30:00',
                python=remote_python,
                scheduler_options={'host': host_address},#, 'interface': scheduler_network_interface},
                local_directory=f'/home/{account}/tiktok',
                log_directory=f'/home/{account}/tiktok',
                shared_temp_directory=f'/home/{account}/tiktok',
                job_extra_directives=[f'-D /home/{account}/tiktok'],  # Ensure tasks run in directory they have permissions in
                asynchronous=True
            )
            await self.cluster.scale(num_workers)

        return self.cluster
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.cluster is not None:
            await self.cluster.close()

async def process_future(f, batch_tasks_lookup, timeout, max_task_tries, tasks_progress_bar, exception_counter, cancel_if_unfinished=False):
    batch_tasks, processed = batch_tasks_lookup[f.key]
    if processed:
        return
    try:
        if cancel_if_unfinished and f.status != "finished":
            await f.cancel()
            batch_results = [{'return': None, 'exception': TimeoutError(f"Task timed out after {timeout} seconds"), 'pre_time': None, 'post_time': datetime.datetime.now()} for _ in batch_tasks]
        else:
            batch_results = await f.result()
    except Exception as e:
        batch_results = [{'return': None, 'exception': e, 'pre_time': None, 'post_time': datetime.datetime.now()} for _ in batch_tasks]
    assert len(batch_tasks) == len(batch_results), "Number of tasks and results must match"
    for t, r in zip(batch_tasks, batch_results):
        if r['exception'] is not None:
            t.exceptions.append(r)
            exception_counter.add(1)
            if len(t.exceptions) >= max_task_tries:
                t.completed = True
                tasks_progress_bar.update(1)
        else:
            t.result = r
            t.completed = True
            tasks_progress_bar.update(1)
    batch_tasks_lookup[f.key] = (batch_tasks, True)

async def get_results(task_futures, batch_tasks_lookup, timeout, max_task_tries, tasks_progress_bar, exception_counter):
    async for f in dask_as_completed(task_futures, raise_errors=False):
        await process_future(f, batch_tasks_lookup, timeout, max_task_tries, tasks_progress_bar, exception_counter)

class Counter:
    def __init__(self):
        self.count = 0
    def add(self, n):
        self.count += n

class TaskDataset:
    def __init__(self):
        self.tasks = pl.DataFrame(columns=['args', 'result', 'exceptions', 'completed'])

    def add_potential_ids(self, args):
        new_tasks = pl.DataFrame([
            {'args': arg, 'result': None, 'exceptions': [], 'completed': False} for arg in args
        ])
        self.tasks = pl.concat([self.tasks, new_tasks])

    def load_existing_df(self, df):
        df = df.with_columns(pl.col('result').map_elements(lambda r: r is not None and 'return' in r and r['return'] is not None, pl.Boolean).alias('completed'))
        self.tasks = pl.concat([self.tasks, df])

    def get_batch(self, batch_size):
        task_rows = self.tasks.filter(pl.col('completed') == False).head(batch_size)
        def create_task_from_row(row):
            t = DaskTask(row['args'])
            t.completed = row['completed']
            t.result = row['result']
            if isinstance(row['exceptions'], np.ndarray):
                t.exceptions = row['exceptions'].tolist()
            else:
                t.exceptions = row['exceptions']
            return t
        tasks = [create_task_from_row(row) for row in task_rows.to_dicts()]
        return tasks

    def update_tasks(self, tasks):
        # Create a DataFrame from the tasks
        updates_df = pl.DataFrame({
            "args": [t.args for t in tasks],
            "result": [t.result for t in tasks],
            "exceptions": [t.exceptions for t in tasks],
            "completed": [t.completed for t in tasks]
        })
        
        # Update the existing DataFrame using join and coalesce
        self.tasks = self.tasks.join(
                updates_df,
                on="args",
                how="left"
            )\
            .with_columns([
                pl.col("result_right").alias("result").fill_null(pl.col("result")),
                pl.col("exceptions_right").alias("exceptions").fill_null(pl.col("exceptions")),
                pl.col("completed_right").alias("completed").fill_null(pl.col("completed"))
            ])\
            .drop(["result_right", "exceptions_right", "completed_right"])
        
    
    def num_left(self):
        return len(self.tasks.filter(pl.col('completed') == False))
    
    def __len__(self):
        return len(self.tasks)

async def dask_map(function, dataset, num_workers=16, reqs_per_ip=1000, batch_size=100000, task_batch_size=1000, max_task_tries=5, task_nthreads=1, task_timeout=10, worker_cpu=256, worker_mem=512, cluster_type='local'):
    network_interfaces = ['eth0'] if cluster_type in ['ssh', 'slurm'] else [None]
    interface_ratios = [1.0] if cluster_type in ['ssh', 'slurm'] else [1]
    assert all(int(ratio * task_nthreads) > 0 for ratio in interface_ratios), "Must have at least one thread per network interface"
    function = MultiNetworkInterfaceFunc(DaskFunc(function), network_interfaces=network_interfaces, ratios=interface_ratios, task_nthreads=task_nthreads)
    # network_interface = None # 'wlan0' if cluster_type == 'raspi' else None
    # function = BatchNetworkInterfaceFunc(DaskFunc(function), network_interface=network_interface, task_nthreads=task_nthreads)
    dotenv.load_dotenv()
    tasks_progress_bar = tqdm(total=dataset.num_left(), desc="All Tasks")
    batch_progress_bar = tqdm(total=min(batch_size, dataset.num_left()), desc="Batch Tasks", leave=False)

    wlan_username = os.environ['EDUROAM_USERNAME']
    wlan_password = os.environ['EDUROAM_PASSWORD']
    raspi_password = os.environ['RASPI_PASSWORD']
    slurm_account_password = os.environ['SCHEDULER_PASSWORD']

    num_cluster_errors = 0
    max_cluster_errors = 3

    while dataset.num_left() > 0:
        try:
            cluster_manager = ClusterManager()
            async with DaskCluster(cluster_type, cluster_manager, worker_cpu=worker_cpu, worker_mem=worker_mem) as cluster:
                async with DaskClient(cluster) as client:
                    if isinstance(cluster, DaskFargateCluster):
                        cluster.adapt(minimum=1, maximum=num_workers)
                        # wait for workers to start
                        client.wait_for_workers(1, timeout=120)
                    num_reqs_for_current_ips = 0
                    num_exceptions_for_current_ips = 0
                    while dataset.num_left() > 0:
                        try:
                            current_batch_size = min(batch_size, dataset.num_left())

                            # prepping args for mapping 
                            # batching tasks as we want to avoid having dask tasks that are too small
                            all_batch_tasks = dataset.get_batch(current_batch_size)
                            batch_tasks = []
                            for i in range(0, len(all_batch_tasks), task_batch_size):
                                batch_tasks.append(all_batch_tasks[i:i+task_batch_size])
                            batch_args = [[t.args for t in batch] for batch in batch_tasks]

                            num_reqs_for_current_ips += current_batch_size

                            # reset the progress bar
                            batch_progress_bar.reset(total=current_batch_size)

                            # start the timeout timer
                            total_time = sum(len(b) for b in batch_tasks) * task_timeout
                            num_actual_workers = len(cluster.workers)
                            if num_actual_workers == 0:
                                num_actual_workers = 1
                            timeout = total_time / (num_actual_workers * task_nthreads)

                            # send out the tasks
                            task_futures = client.map(function, batch_args)
                            batch_tasks_lookup = {f.key: (mini_batch_tasks, False) for mini_batch_tasks, f in zip(batch_tasks, task_futures)}
                            for f in task_futures:
                                # TODO update with specific task arg size, not larger batch size
                                f.add_done_callback(lambda _: batch_progress_bar.update(task_batch_size))

                            # wait for the futures to complete, with a timeout
                            # get all the results
                            exception_counter = Counter()
                            try:
                                await asyncio.wait_for(get_results(task_futures, batch_tasks_lookup, timeout, max_task_tries, tasks_progress_bar, exception_counter), timeout=timeout)
                            except Exception as e:
                                # cancel all the unfinished tasks, and add the exceptions to the task
                                for f in task_futures:
                                    await process_future(f, batch_tasks_lookup, timeout, max_task_tries, tasks_progress_bar, exception_counter, cancel_if_unfinished=True)

                            dataset.update_tasks(all_batch_tasks)
                            num_exceptions_for_current_ips += exception_counter.count

                            # check if we need to recreate workers
                            if dataset.num_left() > 0 and num_reqs_for_current_ips >= reqs_per_ip * num_actual_workers:
                                # recreate workers to get new IPs
                                if cluster_type == 'fargate':
                                    cluster.scale(0)
                                    wait_until(lambda: len(client.scheduler_info()['workers']) == 0, timeout=120)
                                    num_reqs_for_current_ips = 0
                                    cluster.adapt(minimum=1, maximum=num_workers)
                                    client.wait_for_workers(1, timeout=120)
                                elif cluster_type == 'ssh' or cluster_type == 'slurm':
                                    # reset mac address of raspberry pis and rescan for the new assigned IPs
                                    print("Changing worker IPs...")
                                    # moving these before so that even if we get exception, we know we tried
                                    num_reqs_for_current_ips = 0
                                    num_exceptions_for_current_ips = 0
                                    # await cluster_manager.change_mac_addresses()
                                    res = await client.run(
                                        change_mac_address, 
                                        wlan_username, 
                                        wlan_password,
                                        password=slurm_account_password,
                                        on_error='return',
                                        workers=list(cluster.scheduler.workers.keys())
                                    )
                                    num_res = len(res)
                                    num_success = len([r for addr, r in res.items() if r is None])
                                    print(f"Changed {num_success} out of {num_res} worker MAC addresses")
                        # catch exceptions that are recoverable without restarting the cluster
                        except RestartClusterException:
                            raise
                        except Exception as e:
                            if client.scheduler_info(): # client is still connected
                                print(f"Batch Error: {e}, Stacktrace: {traceback.format_exc()}")
                                continue
                            else:
                                raise
                                
        except Exception as ex:
            print(f"Cluster Restart Error: {ex}, Stacktrace: {traceback.format_exc()}")
            num_cluster_errors += 1
            if num_cluster_errors > max_cluster_errors:
                raise
            continue
    tasks_progress_bar.close()
    batch_progress_bar.close()

    return

class InvalidResponseException(Exception):
    pass

class NotFoundException(Exception):
    pass

class RestartClusterException(Exception):
    pass


def get_headers():
    # TODO different user agent results in different html encoding, need to update process video class for each user agent
    # software_names = [rug_params.SoftwareName.CHROME.value, rug_params.SoftwareName.FIREFOX.value]
    # operating_systems = [rug_params.OperatingSystem.WINDOWS.value, rug_params.OperatingSystem.ANDROID.value, rug_params.OperatingSystem.IOS.value, rug_params.OperatingSystem.MAC_OS_X.value]   
    
    # user_agent_rotator = rug_user_agent.UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

    # # Get Random User Agent String.
    # user_agent = user_agent_rotator.get_random_user_agent()
    
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-CA',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
    }
    return headers

class ProcessVideo:
    def __init__(self, headers=None):
        self.text = ""
        self.headers = headers
        self.start = -1
        self.json_start = '"webapp.video-detail":'
        self.json_start_len = len(self.json_start)
        self.end = -1
        self.json_end = ',"webapp.a-b":'
    
    def process_chunk(self, text_chunk):
        self.text += text_chunk
        if len(self.text) < self.json_start_len:
            return 'continue'
        if self.start == -1:
            self.start = self.text.find(self.json_start)
            if self.start != -1:
                self.text = self.text[self.start + self.json_start_len:]
                self.start = 0
        if self.start != -1:
            self.end = self.text.find(self.json_end)
            if self.end != -1:
                self.text = self.text[:self.end]
                return 'break'
        return 'continue'
            
    def process_response(self):
        if self.start == -1 or self.end == -1:
            err_data = {'text': self.text}
            if self.headers:
                err_data['headers'] = self.headers
            raise InvalidResponseException(
                "Could not find normal JSON section in returned HTML."
            )
        video_detail = json.loads(self.text)
        if video_detail.get("statusCode", 0) != 0: # assume 0 if not present
            # TODO retry when status indicates server error
            return video_detail
        video_info = video_detail.get("itemInfo", {}).get("itemStruct")
        if video_info is None:
            raise InvalidResponseException(
                video_detail, "TikTok JSON did not contain expected JSON."
            )
        return video_info

def process_video(text, headers=None):
    video_processor = ProcessVideo(headers=headers)
    video_processor.process_chunk(text)
    return video_processor.process_response()

class PyCurlResponse:
    status_code: int
    headers: dict
    content: bytes
    text: str

class PyCurlClient:
    def __init__(self, share=None, network_interface=None):
        self.c = pycurl.Curl()
        self.response_headers = {}
        self.network_interface = network_interface
        self.share = share

    def _setup(self, url, headers):
        self.buffer = io.BytesIO()
        self.c.setopt(pycurl.URL, url)
        self.c.setopt(pycurl.HTTPHEADER, [f"{key}: {value}" for key, value in headers.items()])
        self.c.setopt(pycurl.TIMEOUT, 10)
        self.c.setopt(pycurl.WRITEFUNCTION, self.buffer.write)
        self.c.setopt(pycurl.HEADERFUNCTION, self._header_function)
        self.c.setopt(pycurl.CAINFO, certifi.where())
        self.c.setopt(pycurl.IPRESOLVE, pycurl.IPRESOLVE_V4)
        if self.network_interface:
            self.c.setopt(pycurl.INTERFACE, self.network_interface)
        if self.share:
            self.c.setopt(pycurl.SHARE, self.share)
    
    def get(self, url, headers={}):
        self._setup(url, headers)
        self.c.perform()
        return self._get_response()

    def _get_response(self):
        resp = PyCurlResponse()
        resp.status_code = self.c.getinfo(pycurl.HTTP_CODE)
        resp.headers = self.response_headers

        # Json response
        resp_bytes = self.buffer.getvalue()

        if 'content-encoding' in self.response_headers:
            if self.response_headers['content-encoding'] == 'br':
                resp.content = brotli.decompress(resp_bytes)
            else:
                raise NotImplementedError("Unknown content encoding")
        else:
            resp.content = resp_bytes

        resp.text = resp.content.decode('utf-8')

        self.buffer.close()

        return resp

    def _header_function(self, header_line):
        header_line = header_line.decode('iso-8859-1')

        if ':' not in header_line:
            return

        name, value = header_line.split(':', 1)

        name = name.strip()
        value = value.strip()

        name = name.lower()

        self.response_headers[name] = value

    def close(self):
        self.c.close()


def get_video(video_id, network_interface):
    url = f"https://www.tiktok.com/@/video/{video_id}"
    headers = get_headers()
    
    client = PyCurlClient(network_interface=network_interface)
    resp = client.get(url, headers=headers)
    client.close()

    if resp.status_code >= 300:
        raise InvalidResponseException(f"Status code: {resp.status_code}")

    resp_html = resp.text

    # cannot get resolved user, tiktok doesn't redirect when video is hidden
    video_processor = ProcessVideo(headers=resp.headers)
    video_processor.process_chunk(resp_html)
    return video_processor.process_response()

class DaskFunc:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        
        pre_time = datetime.datetime.now()
        try:
            res = self.func(*args)
            exception = None
        except Exception as e:
            res = None
            exception = {'ex': e}
        post_time = datetime.datetime.now()

        return {
            'return': res,
            'exception': exception,
            'pre_time': pre_time,
            'post_time': post_time,
        }
    
class BatchNetworkInterfaceFunc:
    def __init__(self, func, network_interface=None, task_nthreads=1):
        self.func = func
        self.network_interface = network_interface
        self.task_nthreads = task_nthreads

    def __call__(self, batch_args):
        return thread_map(batch_args, itertools.repeat(self.network_interface), function=self.func, num_workers=self.task_nthreads)
        
class MultiNetworkInterfaceFunc:
    def __init__(self, func, network_interfaces=[], ratios=[], task_nthreads=1):
        assert len(network_interfaces) == len(ratios), "Number of network interfaces must match number of ratios"
        assert len(network_interfaces) > 0, "Must have at least one network interface"
        self.func = func
        self.task_nthreads = task_nthreads
        self.network_interfaces = network_interfaces
        self.ratios = ratios

    def __call__(self, batch_args):
        # TODO use httpx again for default interface
        executor = ThreadPoolExecutor(max_workers=len(self.network_interfaces))
        all_func_args = []
        cum_ratios = [sum(self.ratios[:i]) for i in range(len(self.ratios) + 1)]
        for i in range(len(self.ratios)):
            all_func_args.append(batch_args[int(len(batch_args) * cum_ratios[i]):int(len(batch_args) * cum_ratios[i+1])])
        assert sum(len(args) for args in all_func_args) == len(batch_args), "Number of batch args must match number of all batch args"
        
        futures = []
        # TODO add httpx option back for network interface that doesn't need it
        for network_interface, ratio, func_args in zip(self.network_interfaces, self.ratios, all_func_args):
            func_nthreads = int(self.task_nthreads * ratio)
            network_interface_func = BatchNetworkInterfaceFunc(self.func, network_interface=network_interface, task_nthreads=func_nthreads)
            future = executor.submit(network_interface_func, func_args)
            futures.append(future)

        results = [future.result() for future in futures]
        return [r for result in results for r in result]
        
    
class DaskTask:
    def __init__(self, args):
        self.args = args
        self.exceptions = []
        self.result = None
        self.completed = False
    

async def get_random_sample(
        generation_strategy,
        start_time,
        num_time,
        time_unit,
        num_workers,
        reqs_per_ip,
        batch_size,
        task_batch_size,
        task_nthreads,
        task_timeout,
        max_task_tries,
        worker_cpu,
        worker_mem,
        cluster_type,
        method
    ):
    print(f"Getting random sample at {start_time} for {num_time} {time_unit}")
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    with open(os.path.join(this_dir_path, '..', '..', 'figs', 'all_videos', f'{generation_strategy}_two_segments_combinations.json'), 'r') as file:
        data = json.load(file)

    # get bits of non timestamp sections of ID
    # order dict according to interval
    data = [(tuple(map(int, interval.strip('()').split(', '))), vals) for interval, vals in data.items()]
    data = sorted(data, key=lambda x: x[0][0])
    # get rid of millisecond bits
    data = [t for t in data if t[0] != (0,9)]
    interval_bits = []
    intervals = [d[0] for d in data]
    for interval, vals in data:
        # format ints to binary
        num_bits = interval[1] - interval[0] + 1
        bits = [format(i, f'0{num_bits}b') for i in vals]
        interval_bits.append(bits)
    other_bit_sequences = itertools.product(*interval_bits)
    other_bit_sequences = [''.join(bits) for bits in other_bit_sequences]

    # get all videos in 1 millisecond
    
    unit_map = {
        'ms': 'milliseconds',
        's': 'seconds',
        'm': 'minutes',
    }
    time_delta = datetime.timedelta(**{unit_map[time_unit]: num_time})
    
    end_time = start_time + time_delta
    c_time = start_time
    all_timestamp_bits = []
    while c_time < end_time:
        unix_timestamp_bits = format(int(c_time.timestamp()), '032b')
        milliseconds = int(format(c_time.timestamp(), '.3f').split('.')[1])
        milliseconds_bits = format(milliseconds, '010b')
        timestamp_bits = unix_timestamp_bits + milliseconds_bits
        all_timestamp_bits.append(timestamp_bits)
        c_time += datetime.timedelta(milliseconds=1)

    potential_video_bits = itertools.product(all_timestamp_bits, other_bit_sequences)
    potential_video_bits = [''.join(bits) for bits in potential_video_bits]
    potential_video_ids = [int(bits, 2) for bits in potential_video_bits]

    date_dir = start_time.strftime('%Y_%m_%d')
    results_dir_path = os.path.join(this_dir_path, '..', 'data', 'results', date_dir, 'hours', str(start_time.hour), str(start_time.minute), str(start_time.second))
    
    if os.path.exists(results_dir_path) and os.path.exists(os.path.join(results_dir_path, 'results.parquet.gzip')):
        # remove ids that have already been collected
        try:
            existing_df = pl.read_parquet(os.path.join(results_dir_path, 'results.parquet.gzip'))
        except Exception as e:
            print(f"Error reading existing results: {e}")
            dataset = TaskDataset()
            dataset.add_potential_ids(potential_video_ids)
        else:
            dataset = TaskDataset()
            dataset.load_existing_df(existing_df)

            # add ids that haven't been collected
            existing_ids = set(existing_df['args'])
            potential_video_ids = [i for i in potential_video_ids if i not in existing_ids]
            dataset.add_potential_ids(potential_video_ids)

            if dataset.num_left() == 0:
                print("All potential video IDs have been collected")
                return
    else:
        dataset = TaskDataset()
        dataset.add_potential_ids(potential_video_ids)

    if method == 'async':
        raise NotImplementedError("Async method not implemented")
        dataset = await async_map(async_get_video, dataset, num_workers=num_workers)
    elif method == 'dask':
        try:
            await asyncio.wait_for(dask_map(
                get_video, 
                dataset, 
                num_workers=num_workers, 
                reqs_per_ip=reqs_per_ip, 
                batch_size=batch_size,
                task_batch_size=task_batch_size,
                task_timeout=task_timeout,
                task_nthreads=task_nthreads, 
                max_task_tries=max_task_tries,
                worker_cpu=worker_cpu, 
                worker_mem=worker_mem,
                cluster_type=cluster_type
            ), timeout=60 * 60)
        except asyncio.exceptions.TimeoutError:
            print(f"Ran out of time to complete task, saving current results")
    else:
        raise ValueError("Invalid method")
    results = dataset.tasks
    num_hits = len(results.filter(pl.col('result').map_elements(lambda x: x is not None and x['return'] is not None and 'id' in x['return'] and x['return']['id'] is not None, pl.Boolean)))
    num_valid = len(results.filter(pl.col('result').map_elements(lambda x: x is not None and x['return'] is not None, pl.Boolean)))
    print(f"Num hits: {num_hits}, Num valid: {num_valid}, Num potential video IDs: {len(dataset)}")
    if num_valid == 0:
        raise ValueError("No valid results")
    print(f"Fraction hits: {num_hits / num_valid}")
    print(f"Fraction valid: {num_valid / len(dataset)}")

    date_dir = start_time.strftime('%Y_%m_%d')
    results_dir_path = os.path.join(this_dir_path, '..', '..', 'data', 'results', date_dir, 'hours', str(start_time.hour), str(start_time.minute), str(start_time.second))
    os.makedirs(results_dir_path, exist_ok=True)

    params = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'num_time': num_time,
        'time_unit': time_unit,
        'num_workers': num_workers,
        'reqs_per_ip': reqs_per_ip,
        'batch_size': batch_size,
        'task_batch_size': task_batch_size,
        'task_timeout': task_timeout,
        'task_nthreads': task_nthreads,
        'max_task_tries': max_task_tries,
        'worker_cpu': worker_cpu,
        'worker_mem': worker_mem,
        'cluster_type': cluster_type,
        'generation_strategy': generation_strategy,
        'intervals': intervals,
    }

    with open(os.path.join(results_dir_path, 'parameters.json'), 'w') as f:
        json.dump(params, f)

    df = dataset.tasks
    df = df.with_columns(pl.col('exceptions').map_elements(lambda exs: [{'exception': str(e['exception']), 'pre_time': e['pre_time'], 'post_time': e['post_time']} for e in exs], pl.List))
    df.write_parquet(os.path.join(results_dir_path, 'results.parquet.gzip'), compression='gzip')

    
async def run_random_sample():
    generation_strategy = 'all'
    # TODO run at persistent time after collection, i.e. if collection takes an hour, run after 24s after post time
    start_time = datetime.datetime(2024, 4, 10, 19, 59, 45)
    num_time = 1
    time_unit = 'h'
    num_workers = 36
    reqs_per_ip = 200000
    batch_size = 10000
    task_batch_size = 50
    task_nthreads = 2
    task_timeout = 60
    max_task_tries = 10
    worker_cpu = 256
    worker_mem = 512
    cluster_type = 'local'
    method = 'dask'
    if (num_time > 1 and time_unit == 's') or (time_unit == 'm') or (time_unit == 'h'):
        if time_unit == 's':
            num_seconds = num_time
        elif time_unit == 'm':
            num_seconds = num_time * 60
        elif time_unit == 'h':
            num_seconds = num_time * 3600
        else:
            raise ValueError("Invalid time unit")
        if num_seconds > 60 and cluster_type == 'fargate':
            raise ValueError("Too expensive to run for more than 60 seconds on Fargate")
        actual_num_time = 1
        actual_time_unit = 's'
        actual_start_times = []
        for i in range(num_seconds):
            actual_start_time = start_time + datetime.timedelta(seconds=i)
            actual_start_times.append(actual_start_time)
    else:
        actual_num_time = num_time
        actual_time_unit = time_unit
        actual_start_times = [start_time]

    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    date_dir = start_time.strftime('%Y_%m_%d')
    for actual_start_time in actual_start_times:
        await get_random_sample(
            generation_strategy,
            actual_start_time,
            actual_num_time,
            actual_time_unit,
            num_workers,
            reqs_per_ip,
            batch_size,
            task_batch_size,
            task_nthreads,
            task_timeout,
            max_task_tries,
            worker_cpu,
            worker_mem,
        )


async def run_min_each_hour_sample():
    generation_strategy = 'all'
    # TODO run at persistent time after collection, i.e. if collection takes an hour, run after 24s after post time
    num_workers = 36
    reqs_per_ip = 200000
    batch_size = 10000
    task_batch_size = 50
    task_nthreads = 2
    task_timeout = 10
    max_task_tries = 10
    worker_cpu = 256
    worker_mem = 512
    cluster_type = 'slurm'
    method = 'dask'
    actual_start_times = []
    actual_num_time = 1
    actual_time_unit = 's'
    min_time = datetime.datetime(2024, 4, 10, 23, 42, 59)
    for h in range(24):
        for s in range(60):
            s_time = datetime.datetime(2024, 4, 10, h, 42, s)
            if s_time < min_time:
                continue
            actual_start_times.append(s_time)

    for actual_start_time in actual_start_times:
        await get_random_sample(
            generation_strategy,
            actual_start_time,
            actual_num_time,
            actual_time_unit,
            num_workers,
            reqs_per_ip,
            batch_size,
            task_batch_size,
            task_nthreads,
            task_timeout,
            max_task_tries,
            worker_cpu,
            worker_mem,
            cluster_type,
            method
        )

def get_ip(_, network_interface):
    client = PyCurlClient(network_interface=network_interface)
    resp = client.get('https://ifconfig.me/ip').content.decode('utf-8')
    client.close()
    return resp

async def run_get_ips():
    results = await dask_map(
        get_ip, 
        TaskDataset(range(100)), 
        reqs_per_ip=5, 
        batch_size=5,
        task_batch_size=1,
        task_timeout=10,
        task_nthreads=5,
        cluster_type='slurm'
    )
    ips = [r.result['return'] for r in results.tasks if r.result is not None]
    print(collections.Counter(ips))
    print(f"Num unique IPs: {len(set(ips))}")
    
async def main():
    # logging.basicConfig(level=logging.DEBUG)
    dotenv.load_dotenv()
    await run_random_sample()
    # await run_min_each_hour_sample()
    # await run_get_ips()

if __name__ == '__main__':
    asyncio.run(main())
