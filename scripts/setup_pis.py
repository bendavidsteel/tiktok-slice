import asyncio
import datetime
import json
import os
import re
import subprocess
import traceback

import asyncssh
import dotenv
import randmac
import tqdm

from map_funcs import async_amap

async def setup_pi(conn, connect_options, reqs='', ip_host_map={}, munge_key_path=None):

    r = await conn.run('pwd', check=True)
    if r.stdout.strip() != f'/home/{connect_options["username"]}':
        raise Exception(f"Need to reset this pi: {r.stdout}")

    r = await conn.run('sudo apt update', check=True)
    if r.returncode != 0:
        if "you must manually run 'sudo dpkg --configure -a' to correct the problem" in r.stderr:
            r = await conn.run('sudo dpkg --configure -a', check=True)
            r = await conn.run('sudo apt upgrade -y', check=True)
        elif "Try 'apt --fix-broken install' with no packages" in r.stderr:
            r = await conn.run('sudo apt --fix-broken install', check=True)
            r = await conn.run('sudo apt upgrade -y', check=True)
        else:
            print(f"Failed to upgrade: {r.stderr}")

    r = await conn.run('sudo apt list --installed', check=True)
    packages = r.stdout.split('\n')

    # install required packages
    apt_reqs = ['libcurl4-openssl-dev', 'libssl-dev', 'slurm-wlm', 'libpmix-dev', 'munge', 'libmunge2', 'libmunge-dev', 'cgroup-tools']
    reqs_to_install = [r for r in apt_reqs if not any(r in p for p in packages)]
    if reqs_to_install:
        r = await conn.run(f"sudo apt install -y {' '.join(reqs_to_install)}", check=False)
        if "Try 'apt --fix-broken install' with no packages" in r.stderr:
            r = await conn.run('sudo apt --fix-broken install', check=True)
            r = await conn.run(f"sudo apt install -y {' '.join(reqs_to_install)}", check=True)

    munge_info_r = await conn.run('id munge', check=False)
    if r.returncode != 0:
        await conn.run('sudo apt install -y munge libmunge2 libmunge-dev', check=True)
        munge_info_r = await conn.run('id munge', check=True)

    munge_key_worker_path = '/etc/munge/munge.key'
    r = await conn.run(f'sudo ls -lh {munge_key_worker_path}')
    if r.returncode != 0 or 'munge 0' in r.stdout: # munge key doesn't exist or is empty
        temp_path = '~/munge.key'
        munge_key_host_path = './config/munge.key'
        await asyncssh.scp(munge_key_host_path, (conn, temp_path))
        try:
            await conn.run(f'sudo mv {temp_path} {munge_key_worker_path}', check=True)
        except asyncssh.process.ProcessError as e:
            if 'No such file or directory' in e.stderr:
                await conn.run('sudo mkdir -p /etc/munge', check=True)
                await conn.run(f'sudo mv {temp_path} {munge_key_worker_path}', check=True)

    r = await conn.run(f'sudo ls -lh {munge_key_worker_path}')
    if '-rw-------' not in r.stdout or 'munge munge' not in r.stdout:
        await conn.run(f'sudo chown munge:munge {munge_key_worker_path}', check=True)
        await conn.run(f'sudo chmod 600 {munge_key_worker_path}', check=True)
        await conn.run('sudo systemctl restart munge', check=True)

    # check munge is running
    r = await conn.run('munge -n | unmunge | grep STATUS', check=False)
    if r.returncode != 0:
        raise Exception("Munge is not running")
        
    # set correct munge permissions
    r = await conn.run('sudo chown -R munge: /etc/munge/ /var/log/munge/ /var/lib/munge/ /run/munge/', check=True)
    r = await conn.run('sudo chmod 0700 /etc/munge/ /var/log/munge/ /var/lib/munge/', check=True)
    r = await conn.run('sudo chmod 0755 /run/munge/', check=True)
    r = await conn.run('sudo chmod 0700 /etc/munge/munge.key', check=True)
    r = await conn.run('sudo chown -R munge: /etc/munge/munge.key', check=True)

    # r = await conn.run('sudo systemctl enable munge', check=True)
    # r = await conn.run('sudo systemctl restart munge', check=True)

    slurm_conf_worker_path = '/etc/slurm/slurm.conf'
    # r = await conn.run(f'sudo ls {slurm_conf_worker_path}', check=False)
    # if r.returncode != 0:
    slurm_conf_host_path = './config/slurm.conf'
    temp_path = '~/slurm.conf'
    await asyncssh.scp(slurm_conf_host_path, (conn, temp_path))
    try:
        await conn.run(f'sudo mv {temp_path} {slurm_conf_worker_path}', check=True)
    except asyncssh.process.ProcessError as e:
        if 'No such file or directory' in e.stderr:
            await conn.run('sudo mkdir -p /etc/slurm', check=True)
            await conn.run(f'sudo mv {temp_path} {slurm_conf_worker_path}', check=True)

    r = await conn.run('hostname', check=True)
    if r.stdout != connect_options['username']:
        r = await conn.run(f'sudo hostnamectl set-hostname {connect_options["username"]}', check=True)

    # make a new user with name slurm-user
    user_r = await conn.run('id slurm', check=False)
    if user_r.returncode != 0:
        r = await conn.run('sudo adduser --disabled-password --gecos "" -u 64030 slurm', check=True)
    elif 'uid=64030(slurm)' not in user_r.stdout:
        r = await conn.run('sudo userdel slurm', check=True)
        r = await conn.run('sudo adduser --disabled-password --gecos "" -u 64030 slurm', check=True)
        await conn.run('sudo usermod -aG sudo slurm', check=True)

    r = await conn.run('ls -lh /var/log/slurm', check=False)
    if r.returncode != 0:
        await conn.run('sudo mkdir -p /var/log/slurm', check=True)
        await conn.run('sudo chown -R slurm:slurm /var/log/slurm', check=True)
        await conn.run('sudo chmod -R 755 /var/log/slurm', check=True)
    elif 'slurm:slurm' not in r.stdout:
        await conn.run('sudo chown -R slurm:slurm /var/log/slurm', check=True)
        await conn.run('sudo chmod -R 755 /var/log/slurm', check=True)

    r = await conn.run('cat /etc/hosts', check=True)
    hosts_file = r.stdout

    # Check and remove incorrect lines
    for ip, host in ip_host_map.items():
        for line in hosts_file.splitlines():
            if host in line and line != f"{ip} {host}":
                # Remove incorrect lines
                await conn.run(f"sudo sed -i '/{line}/d' /etc/hosts", check=True)

    # Get the updated /etc/hosts file
    r = await conn.run('cat /etc/hosts', check=True)
    hosts_file = r.stdout

    # Add correct lines if they are not already present
    for ip, host in ip_host_map.items():
        if f"{ip} {host}" not in hosts_file:
            await conn.run(f"echo '{ip} {host}' | sudo tee -a /etc/hosts", check=True)

    # r = await conn.run('sudo systemctl enable slurmd', check=False)
    if 'slurmd.service does not exist' in r.stderr:
        await conn.run('sudo apt install -y slurm-wlm', check=True)
        await conn.run('sudo systemctl enable slurmd', check=True)
    await conn.run('sudo systemctl restart slurmd', check=True)

    # install python
    r = await conn.run('python3 --version', check=True)
    if r.stdout.strip() != 'Python 3.10.12':
        await conn.run('sudo apt update', check=True)
        await conn.run('sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget', check=True)
        await conn.run('cd /tmp && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz', check=True)
        await conn.run('cd /tmp && tar -xf Python-3.10.12.tgz', check=True)
        await conn.run('cd /tmp/Python-3.10.12 && ./configure --enable-optimizations', check=True)
        await conn.run('cd /tmp/Python-3.10.12 && sudo make install', check=True)
        r = await conn.run('python3 --version', check=True)
        assert r.stdout.strip() == 'Python 3.10.12'

    # setup virtual environment
    r = await conn.run('ls ~/ben/tiktok/venv', check=False)
    if r.returncode != 0:
        r = await conn.run('python3 -m venv ~/ben/tiktok/venv', check=True)
    else:
        r = await conn.run('~/ben/tiktok/venv/bin/python --version', check=False)
        if r.stdout.strip() != 'Python 3.10.12':
            r = await conn.run('rm -rf ~/ben/tiktok/venv', check=True)
            r = await conn.run('python3 -m venv ~/ben/tiktok/venv', check=True)

    # install requirements
    r = await conn.run('~/ben/tiktok/venv/bin/pip list', check=True)
    installed_packages = r.stdout.split('\n')
    installed_packages = ['=='.join([e for e in p.split(' ') if e]) for p in installed_packages[2:-1]]
    required_packages = reqs.split()
    not_installed_packages = [req for req in required_packages if req not in installed_packages]
    if not_installed_packages:
        r = await conn.run(f"~/ben/tiktok/venv/bin/pip install {' '.join(not_installed_packages)}", check=True)


async def vpn_via_service(conn):
    # check if openvpn service file exists
    r_file_exists = await conn.run('ls /etc/systemd/system/openvpn.service', check=False)
    
    # make an openvpn service file
    username = conn.get_extra_info('username')
    service_file_contents = f"""
        [Unit]
        Description=OpenVPN connection to Server
        After=network.target

        [Service]
        Type=simple
        ExecStart=sudo /usr/sbin/openvpn --config /home/{username}/tum.eduvpn.lrz.de_tum-full-ov_20240415_tarjan@raspi.ovpn
        Restart=on-failure

        [Install]
        WantedBy=multi-user.target"""
    r_file_contents = await conn.run('cat /etc/systemd/system/openvpn.service', check=False)
    if r_file_exists.returncode != 0 or service_file_contents not in r_file_contents.stdout:
        await conn.run(f'echo "{service_file_contents}" | sudo tee /etc/systemd/system/openvpn.service', check=True)
        await conn.run("sudo systemctl daemon-reload", check=True)

    await conn.run("sudo systemctl disable openvpn", check=True) # ensure it doesn't start on boot
    await conn.run("sudo systemctl start openvpn", check=True)

    # check that the service is running
    r = await conn.run("sudo systemctl status openvpn", check=True)
    
    if 'active (running)' not in r.stdout:
        raise Exception("Failed to start openvpn service")

async def vpn_via_command(conn):
    # check if openvpn process already exists
    r = await conn.run("ps aux | grep openvpn", check=True)
    if 'openvpn --config' not in r.stdout:
        # delete current nohup file
        r = await conn.run("rm -f nohup.out", check=True)
        # start the vpn service
        r = await conn.create_process("nohup sudo openvpn --config ~/tum.eduvpn.lrz.de_tum-full-ov_20240415_tarjan@raspi.ovpn &")
        # check it's running
        r = await conn.run("ps aux | grep openvpn", check=True)
        if 'openvpn --config' not in r.stdout:
            raise Exception("Failed to start openvpn")
        await asyncio.sleep(5)
        r = await conn.run("ls nohup.out", check=False)
        if r.returncode != 0:
            raise Exception("Failed to start openvpn")
        r = await conn.run("cat nohup.out", check=True)
        if 'exited' in r.stdout:
            raise Exception("Failed to start openvpn")

async def stop_vpn(conn):
    r = await conn.run("sudo killall openvpn", check=False)

async def start_vpn(conn):
    # check if ip table already exists
    local_ip = conn._host
    r = await conn.run("/usr/sbin/route -n", check=True)
    table_lines = r.stdout.split('\n')
    if not any(local_ip in line for line in table_lines):
        # add ip table to route traffic from public ip to into server
        rasp_pi_ip = conn._host
        rasp_pi_subnet = '.'.join(rasp_pi_ip.split('.')[:-1]) + '.0/24'
        await conn.run(f"sudo ip rule add table 128 from {rasp_pi_ip}", check=True)
        r = await conn.run(f"sudo ip route add table 128 to {rasp_pi_subnet} dev eth0")
        if r.returncode != 0:
            if 'File exists' not in r.stderr:
                raise Exception(f"Failed to add route table: {r.stderr}")
        r = await conn.run("ip route | awk '/default/ {print $3; exit}'", check=True)
        gateway_ip = r.stdout.strip()
        r = await conn.run(f"sudo ip route add table 128 default via {gateway_ip}")
        if r.returncode != 0:
            if 'File exists' not in r.stderr:
                raise Exception(f"Failed to add route table: {r.stderr}")

    await vpn_via_command(conn)

async def ensure_wifi_connection(conn, connect_options, force_start=False):
    num_tries = 0
    max_tries = 3
    # TODO use google dns64 server to allow ipv6 connections
    # test if we need password for sudo
    r = await conn.run("sudo -n true", check=False)
    if r.returncode == 1:
        password = connect_options['password']
        prepend = f"echo '{password}' | sudo -S"
    else:
        prepend = "sudo"

    async def create_wifi_connection():
        username = os.environ['EDUROAM_USERNAME']
        password = os.environ['EDUROAM_PASSWORD']
        command = f'{prepend} nmcli con add type wifi con-name "eduroam" ifname "wlan0" ssid "eduroam" wifi-sec.key-mgmt "wpa-eap" 802-1x.identity "{username}" 802-1x.password "{password}" 802-1x.system-ca-certs "yes" 802-1x.eap "peap" 802-1x.phase2-auth "mschapv2"'
        r = await conn.run(command, check=True)

    async def get_wifi_connections():
        try:
            r = await conn.run("nmcli con", check=True)
        except Exception as e:
            if e.stderr == "Error: NetworkManager is not running.":
                r = await conn.run("sudo systemctl start NetworkManager", check=True)
                r = await conn.run("nmcli con", check=True)
            else:
                raise
        connections = r.stdout.split('\n')
        return connections

    while num_tries < max_tries:
        num_tries += 1
        try:
            # check if wifi connection exists
            connections = await get_wifi_connections()
            
            eduroam_lines = [c for c in connections if c.startswith('eduroam')]
            if len(eduroam_lines) == 0:
                # create wifi connection
                await create_wifi_connection()
                await asyncio.sleep(3)
                connections = await get_wifi_connections()
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
                        r = await conn.run(f"{prepend} nmcli con delete {uuid}", check=True)
                else:
                    # delete all connections
                    for eduroam_line in eduroam_lines:
                        uuid = eduroam_line.split(' ')[1]
                        r = await conn.run(f"{prepend} nmcli con delete {uuid}", check=True)
                    # create wifi connection
                    await create_wifi_connection()
                connections = await get_wifi_connections()

            elif len(eduroam_lines) == 1:
                eduroam_line = eduroam_lines[0]
                reqs = [' wifi ', ' wlan0 ']
                if not all(req in eduroam_line for req in reqs):
                    # delete existing connection
                    r = await conn.run(f"{prepend} nmcli con delete eduroam", check=True)
                    # create wifi connection
                    await create_wifi_connection()
                    connections = await get_wifi_connections()
            assert any(c.startswith('eduroam') for c in connections), "No eduroam connection"
            eduroam_lines = [c for c in connections if c.startswith('eduroam')]
            assert len(eduroam_lines) == 1, "Duplicate eduroam connections"
            eduroam_line = eduroam_lines[0]
            assert 'wifi' in eduroam_line and 'wlan0' in eduroam_line, "Eduroam connection not setup correctly"
            # check if connection is up
            r = await conn.run("nmcli -f GENERAL.STATE con show eduroam", check=True)
            if 'activated' not in r.stdout or force_start:
                # start wifi connection
                r = await conn.run(f"{prepend} nmcli connection up eduroam", check=True)
        except asyncssh.misc.ChannelOpenError:
            raise
        except Exception as e:
            # delete connection and try again
            r = await conn.run(f"{prepend} nmcli con delete eduroam", check=True)
        else:
            break
    else:
        raise Exception("Failed to create wifi connection")
    
async def start_wifi_connections(hosts, connect_options, progress_bar=True, timeout=10, num_workers=4):
    iterable = list(zip(hosts, connect_options))
    async def run_start_wifi(args):
        host, co  = args
        try:
            conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=timeout)
            await asyncio.wait_for(ensure_wifi_connection(conn, co), timeout=120)
        except asyncssh.process.ProcessError as e:
            ex = asyncssh.misc.Error(e.code, e.stderr)
            print(f"Failed to start wifi connection on {host} (username: {co['username']}): {ex}")
            return None, None
        except asyncio.exceptions.TimeoutError:
            ex = asyncio.exceptions.TimeoutError(f"Timed out starting wifi connection")
            print(f"Failed to start wifi connection on {host} (username: {co['username']}): {ex}")
            return None, None
        except Exception as e:
            print(f"Failed to start wifi connection on {host} (username: {co['username']}): {e}")
            return None, None
        else:
            return host, co

    res = await async_amap(run_start_wifi, iterable, num_workers=num_workers, progress_bar=progress_bar, pbar_desc="Starting wifi connections")
    working_hosts = [host for host, co in res if host]
    working_connect_options = [co for host, co in res if host]
    return working_hosts, working_connect_options

async def check_connection(hosts, usernames, progress_bar=False, timeout=10, num_workers=4):
    assert len(hosts) == len(usernames), "Hosts and usernames must be the same length"
    iterable = list(zip(hosts, usernames))
    async def run_connect(args):
        host, username = args
        try:
            await asyncio.wait_for(asyncssh.connect(host, username=username, password='rp145', known_hosts=None), timeout=timeout)
        except Exception as e:
            return False
        else:
            return True
    results = await async_amap(run_connect, iterable, num_workers=num_workers, pbar_desc="Checking connections", progress_bar=progress_bar)
    working_hosts = [host for host, res in zip(hosts, results) if res]
    working_usernames = [username for username, res in zip(usernames, results) if res]
    return working_hosts, working_usernames

async def scan_for_pis(possible_usernames, ignore_hosts=[], progress_bar=False, password=None, timeout=10):
    if not password:
        password = os.environ['RASPI_PASSWORD']

    ip_cidrs = ['10.157.115.0/24', '10.162.42.0/23']
    all_reports = []
    for ip_cidr in ip_cidrs:
        r = subprocess.run(f'nmap -p 22 --open -sV {ip_cidr}', shell=True, capture_output=True)

        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, 'nmap', r.stderr)

        stdout = r.stdout.decode()
        result_lines = stdout.split('\n')
        
        report = None
        for i in range(len(result_lines)):
            if 'Nmap scan report' in result_lines[i]:
                if report:
                    all_reports.append(report)
                report = []
                report.append(result_lines[i])
            elif result_lines[i] != '':
                if report:
                    report.append(result_lines[i])

    concurrent = False

    if concurrent:
        async def run_test_connect(report):
            ip = re.search(r'[0-9]+(?:\.[0-9]+){3}', report[0]).group()
            if ip in ignore_hosts:
                return None, None
            table_headers = [i for i, l in enumerate(report) if 'PORT' in l]
            if not table_headers:
                return None, None
            table_header_line = table_headers[0]
            ports = []
            for row_idx in range(table_header_line + 1, len(report)):
                ports.append(report[row_idx])
            ports = [port for port in ports if '22/tcp' in port]
            if not ports:
                return None, None
            port = ports[0]

            for username in possible_usernames:
                try:
                    await asyncio.wait_for(asyncssh.connect(ip, username=username, password=password, known_hosts=None), timeout=timeout)
                except asyncssh.TimeoutError:
                    continue
                except ConnectionResetError:
                    break
                except asyncssh.misc.ConnectionLost:
                    continue
                except asyncssh.misc.PermissionDenied:
                    break
                except asyncssh.misc.DisconnectError:
                    continue
                except asyncio.exceptions.TimeoutError:
                    continue
                except OSError:
                    continue
                except Exception as e:
                    continue
                else:
                    return ip, username
            else:
                return None, None
            
        results = await async_amap(run_test_connect, all_reports, num_workers=4, progress_bar=progress_bar, pbar_desc="Scanning for Pis")
        hosts, usernames = [host for host, username in results if host], [username for host, username in results if host]
        
    else:
        hosts = []
        usernames = []
        remaining_usernames = possible_usernames.copy()
        if progress_bar:
            all_reports = tqdm.tqdm(all_reports, desc="Scanning for Pis")
        for report in all_reports:
            ip = re.search(r'[0-9]+(?:\.[0-9]+){3}', report[0]).group()
            if ip in ignore_hosts:
                continue
            table_headers = [i for i, l in enumerate(report) if 'PORT' in l]
            if not table_headers:
                continue
            table_header_line = table_headers[0]
            port_line = None
            for row_idx in range(table_header_line + 1, len(report)):
                if 'open' in report[row_idx] and '22/tcp' in report[row_idx]:
                    port_line = report[row_idx]
                    break
            else:
                continue

            working_username = None
            if 'Mocana NanoSSH' in port_line:
                continue

            if 'Ubuntu' in port_line:
                continue

            for username in remaining_usernames:
                try:
                    await asyncio.wait_for(asyncssh.connect(ip, username=username, password=password, known_hosts=None), timeout=timeout)
                    working_username = username
                    break
                except asyncssh.misc.PermissionDenied:
                    continue
                except Exception as e:
                    continue

            if working_username:
                hosts.append(ip)
                usernames.append(working_username)
                remaining_usernames.remove(working_username)

    # if username shows up for multiple hosts, remove the duplicates
    unique_usernames = []
    unique_hosts = []
    for host, username in zip(hosts, usernames):
        if username not in unique_usernames:
            unique_usernames.append(username)
            unique_hosts.append(host)
    return unique_hosts, unique_usernames

def generate_random_mac():
    return randmac.RandMac()

async def change_mac_address(conn, connect_options, interface='eth0'):
    # write bash script to change mac address on eth0 network interface
    random_mac = generate_random_mac()
    if interface == 'eth0':
        commands = f"""
            sudo ip link set dev eth0 down
            sudo ip link set dev eth0 address {random_mac}
            sudo ip link set dev eth0 up
            sleep 5  # Wait for a few seconds before reboot to ensure commands are processed
            sudo reboot
        """
        r = await conn.run(f'echo "{commands}" > ~/change_mac.sh', check=True)
        r = await conn.run('chmod +x ~/change_mac.sh', check=True)
        await conn.create_process('nohup ~/change_mac.sh &') # no need to check since it will disconnect the ssh connection
    elif interface == 'wlan0':
        # ensure eduroam connection exists
        await ensure_wifi_connection(conn, connect_options)
        r = await conn.run(f"sudo nmcli con modify --temporary eduroam 802-11-wireless.cloned-mac-address {random_mac}", check=True)
        if r.returncode != 0 or r.stdout or r.stderr:
            raise asyncssh.misc.Error(r.returncode, f"Failed to change MAC address: {r.stdout}, {r.stderr}")
        await ensure_wifi_connection(conn, connect_options)
        r = await conn.run(f"sudo nmcli connection up eduroam", check=True)

async def killall_python(conn):
    r = await conn.run('killall ~/ben/tiktok/venv/bin/python', check=False)
    if r.returncode != 0:
        stderr = r.stderr
        if 'no process found' in stderr:
            pass
        else:
            print(f'Failed to stop on {conn._host}, {conn.get_extra_info("username")}: {stderr}')

async def kill_workers(hosts, connect_options):
    async def run_killall(args):
        host, co = args
        conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
        await killall_python(conn)

    args = zip(hosts, connect_options)
    await async_amap(run_killall, args, num_workers=len(hosts))

async def stop_stale_workers(hosts, connect_options, timeout=10):
    async def run_killall(args):
        host, co = args
        try:
            conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=timeout)
            await asyncio.wait_for(killall_python(conn), timeout=120)
        except:
            return None, None
        else:
            return host, co
    args = zip(hosts, connect_options)
    results = await async_amap(run_killall, args, num_workers=len(hosts))
    working_hosts = [host for host, co in results if host]
    working_connect_options = [co for host, co in results if host]
    return working_hosts, working_connect_options

async def reboot_workers(hosts, connect_options, timeout=10):
    async def run_reboot(args):
        host, co = args
        try:
            conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=timeout)
            commands = f"""
                sudo reboot
            """
            r = await conn.run(f'echo "{commands}" > ~/reboot.sh', check=True)
            r = await conn.run('chmod +x ~/reboot.sh', check=True)
            await conn.create_process('nohup ~/change_mac.sh &') # no need to check since it will disconnect the ssh connection
        except:
            return None, None
        else:
            return host, co

    args = zip(hosts, connect_options)
    results = await async_amap(run_reboot, args, num_workers=len(hosts))
    return [host for host, co in results if host], [co for host, co in results if host]

async def get_connect_latency(hosts, connect_options):
    async def run_connect_latency(args):
        host, co = args
        try:
            start_time = datetime.datetime.now()
            await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
            end_time = datetime.datetime.now()
            latency = (end_time - start_time).total_seconds()
            return host, co, latency
        except:
            return host, co, 999999

    args = zip(hosts, connect_options)
    results = await async_amap(run_connect_latency, args, num_workers=len(hosts))
    hosts, connect_options, latencies = zip(*results)
    return hosts, connect_options, latencies

async def change_mac_addresses(hosts, connect_options, progress_bar=False, **kwargs):
    async def run_change_mac_address(args):
        host, co = args
        tries = 0
        max_tries = 3
        exceptions = []
        while tries < max_tries:
            tries += 1
            conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
            try:
                await asyncio.wait_for(change_mac_address(conn, co, **kwargs), timeout=120)
            except asyncssh.process.ProcessError as e:
                ex = asyncssh.misc.Error(e.code, e.stderr)
                exceptions.append(ex)
            except asyncssh.misc.Error as e:
                exceptions.append(e)
            except asyncio.exceptions.TimeoutError as ex:
                exceptions.append(ex)
            else:
                return
        else:
            raise Exception(f"Failed to change MAC address on {host} (username: {co['username']}) after {max_tries} tries: {exceptions}")
    args = list(zip(hosts, connect_options))
    await async_amap(run_change_mac_address, args, num_workers=len(hosts), progress_bar=progress_bar, pbar_desc="Changing MAC addresses...")

async def run_on_pis(hosts, connect_options, func, timeout=2400, **kwargs):
    async def run_func(args):
        host, connect_option, func, kwargs = args
        print(f"Connecting to {host}, {connect_option['username']}...")
        tries = 0
        max_tries = 1
        exceptions = []
        while tries < max_tries:
            tries += 1
            try:
                connection_options = {'connect_timeout': '20s', 'keepalive_interval': '10s', 'keepalive_count_max': 10}
                connection_options = asyncssh.SSHClientConnectionOptions(**connection_options)
                conn = await asyncio.wait_for(asyncssh.connect(host, **connect_option, known_hosts=None, options=connection_options), timeout=20)
            except Exception as e:
                exceptions.append(e)
            else:
                async with conn:
                    try:
                        r = await asyncio.wait_for(func(conn, connect_option, **kwargs), timeout=timeout)
                    except asyncssh.process.ProcessError as e:
                        e.args = (e.args[0], e.stderr, traceback.format_exc())
                        exceptions.append(e)
                    except Exception as e:
                        e.args = (e.args, traceback.format_exc())
                        exceptions.append(e)
                    else:
                        return r, host, connect_option
        else:
            print(f"Failed to run on {host} (username: {connect_option['username']}) after {max_tries} tries: {exceptions}")
            return None, host, connect_option

    args = list(zip(hosts, connect_options, [func] * len(hosts), [kwargs] * len(hosts)))
    results = await async_amap(run_func, args, num_workers=len(hosts), progress_bar=True)
    return results

async def get_hosts(usernames):
    try:
        # load cached hosts and usernames
        with open('hosts.json', 'r') as file:
            user_hosts = json.load(file)
        hosts_users = [(host, un) for un, host in user_hosts.items()]
        hosts, found_usernames = zip(*hosts_users)
        hosts, found_usernames = list(hosts), list(found_usernames)
        assert len(found_usernames) == len(usernames), "Cached usernames do not match expected usernames"
        await check_connection(hosts, found_usernames)
    except Exception as ex:
        hosts, found_usernames = await scan_for_pis(usernames)
        if len(found_usernames) != len(usernames):
            missing_usernames = set(usernames) - set(found_usernames)
            print(f"Unable to find: {missing_usernames}")
        user_hosts = {un: host for un, host in zip(found_usernames, hosts)}
        with open('hosts.json', 'w') as file:
            json.dump(user_hosts, file)

    return hosts, found_usernames

async def get_hosts_with_retries(usernames, max_tries=2, progress_bar=False, timeout=10, scan=False):
    hosts = []
    found_usernames = []
    try:
        # load cached hosts and usernames
        with open('hosts.json', 'r') as file:
            user_hosts = json.load(file)
        hosts_users = [(host, un) for un, host in user_hosts.items()]
        file_hosts, file_usernames = zip(*hosts_users)
        file_hosts, file_usernames = list(file_hosts), list(file_usernames)
        file_hosts = [h for h, un in zip(file_hosts, file_usernames) if un in usernames]
        file_usernames = [un for un in file_usernames if un in usernames]
        print("Attempting to load cached connection data")
        num_tries = 0
        while num_tries < max_tries:
            num_tries += 1
            print(f"Attempt {num_tries} to find all hosts from file...")
            try_hosts, try_found_usernames = await check_connection(file_hosts, file_usernames, progress_bar=progress_bar, timeout=timeout)
            hosts.extend(try_hosts)
            found_usernames.extend(try_found_usernames)
            if len(found_usernames) == len(usernames):
                break
            else:
                print(f"Found {len(found_usernames)} out of {len(usernames)} hosts, trying again...")
                file_usernames = list(set(file_usernames) - set(found_usernames))
                file_hosts = [user_hosts[un] for un in file_usernames]
        else:
            raise Exception("Unable to find all hosts from file")
    except Exception as ex:
        if scan:
            num_tries = 0
            non_found_usernames = list(set(usernames) - set(found_usernames))
            while num_tries < max_tries:
                num_tries += 1
                print(f"Attempt {num_tries} to find all hosts from network scan...")
                try_hosts, try_found_usernames = await scan_for_pis(non_found_usernames, ignore_hosts=hosts, progress_bar=progress_bar, timeout=timeout)
                hosts.extend(try_hosts)
                found_usernames.extend(try_found_usernames)
                non_found_usernames = list(set(non_found_usernames) - set(try_found_usernames))
                if len(found_usernames) == len(usernames):
                    break
                else:
                    print(f"Found {len(found_usernames)} out of {len(usernames)} hosts, unable to find: {', '.join(set(usernames) - set(found_usernames))}, trying again...")
            else:
                print("Unable to find all hosts after multiple tries, returning found hosts")
            user_hosts = {un: host for un, host in zip(found_usernames, hosts)}
            with open('hosts.json', 'w') as file:
                json.dump(user_hosts, file)

    return hosts, found_usernames

async def get_ip(conn, co, interface='eth0'):
    r = await conn.run(f'curl --interface {interface} ifconfig.me', check=True)
    return r.stdout.strip()

async def main():
    dotenv.load_dotenv()
    potential_usernames = [
        # most reliable batch
        'hoare', 'miacli', 'fred',
        'geoffrey', 'edmund',
        'cook', 'goldwasser', 'milner',
        'hemming', 'lee', 'turing',
        'juris', 'floyd',
        'neumann', 'beauvoir', 'satoshi', 'putnam', 
        'shannon', 'chowning', 
        'tegmark', 'hanson', 'chomsky', 'keynes',
        # next most reliable
        'edward', 'buterin',
        'arendt', 'chan', 'sutskever', 'herbert',
        'mordvintsev',
        # to set up
        'marvin', 'frances', 'barbara', 'conway', 'tarjan', 'lovelace', 'edwin', 'edsger', 'fernando', 'rivest',
        # unreliable
        'ivan'
    ]
    hosts, usernames = await get_hosts_with_retries(potential_usernames, progress_bar=True)
    connect_options = [{'username': username, 'password': 'rp145'} for username in usernames]

    todo = 'setup'

    if todo == 'setup':
        with open('worker_requirements.txt', 'r') as f:
            reqs = f.readlines()
        reqs = ' '.join([req.strip() for req in reqs])
        ip_host_map = {host: username for host, username in zip(hosts, usernames)}
        ip_host_map['10.162.43.238'] = 'pfeffer-HP-ProDesk-405-G4-Desktop-Mini'
        r = await run_on_pis(hosts, connect_options, setup_pi, timeout=600, reqs=reqs, ip_host_map=ip_host_map)
        usernames, hosts = zip(*[(co['username'], host) for ret, host, co in r if ret is None])
        print(f"Successfully set up {usernames} on {hosts}")
    elif todo == 'ping':
        async def ping(conn):
            r = await conn.run('ls', check=True)
        await run_on_pis(hosts, connect_options, ping)
    elif todo == 'get_ip':
        async def print_ip(conn):
            r = await conn.run('curl ifconfig.me', check=True)
            print(f'Public IP: {r.stdout.strip()}')
        run_on_pis(hosts, connect_options, print_ip)
    elif todo == 'change_ip':
        interface = 'wlan0'
        initial_ips = await run_on_pis(hosts, connect_options, get_ip)
        print(f"Initial IPs: {initial_ips}")
        await run_on_pis(hosts, connect_options, ensure_wifi_connection)
        vpn_ips = await run_on_pis(hosts, connect_options, get_ip, interface=interface)
        print(f"WIFI IPs: {vpn_ips}")
        start_time = datetime.datetime.now()
        print("Changing MAC addresses...")
        await run_on_pis(hosts, connect_options, change_mac_address, interface=interface)
        if interface == 'eth0':
            num_tries = 0
            max_tries = 10
            while num_tries < max_tries:
                num_tries += 1
                print(f"Attempt {num_tries} to find all hosts...")
                found_hosts, found_usernames = await get_hosts(usernames)
                if len(found_usernames) == len(usernames):
                    end_time = datetime.datetime.now()
                    print(f"Found all hosts after {num_tries} tries in {end_time - start_time}")
                    break
                else:
                    print(f"Found {len(found_usernames)} out of {len(usernames)} hosts")
            else:
                print(f"Failed to find all hosts after {max_tries} tries")
                return
            
            found_connect_options = [{'username': un, 'password': 'rp145'} for un in found_usernames]
            hosts = found_hosts
            connect_options = found_connect_options

        print("Getting new IPs...")
        new_ips = await run_on_pis(hosts, connect_options, get_ip, interface='wlan0')
        print(f"New IPs: {new_ips}")
    elif todo == 'stop':
        await run_on_pis(hosts, connect_options, killall_python)
    elif todo == 'get_os':
        async def get_os(conn, co):
            # if startx is installed, then it's a desktop
            r = await conn.run('ls -lh /usr/bin | grep startx', check=False)
            if 'startx' in r.stdout:
                os_type = 'desktop'
            else:
                os_type = 'server'
            return {'os_type': os_type, 'username': co['username']}
        os_types = await run_on_pis(hosts, connect_options, get_os)
        print(f"OS Types: {os_types}")

    else:
        raise ValueError(f"Unknown todo: {todo}")

        
            
if __name__ == "__main__":
    asyncio.run(main())