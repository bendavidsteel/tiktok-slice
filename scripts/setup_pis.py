import asyncio
import datetime
import json
import os
import random
import subprocess
import traceback

import asyncssh
import dotenv
import randmac
import tqdm

from map_funcs import async_amap

async def setup_pi(conn, connect_options, reqs=''):
    # install python
    # TODO use more lightweight raspberry pi image
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
    if not all(req in r.stdout for req in reqs.split()):
        r = await conn.run(f'~/ben/tiktok/venv/bin/pip install {reqs}', check=True)

    # setup vpn
    # r = await conn.run('sudo apt install -y openvpn', check=True)
    # username = conn.get_extra_info('username')
    # this_dir_path = os.path.dirname(os.path.realpath(__file__))
    # root_dir_path = os.path.dirname(this_dir_path)
    # rasp_pi_openvpn_file_path = os.path.join(root_dir_path, "config", f'tum.eduvpn.lrz.de_tum-full-ov_20240415_tarjan@raspi.ovpn')
    # await asyncssh.scp(rasp_pi_openvpn_file_path, (conn, '~/'))

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
    
async def start_wifi_connections(hosts, connect_options, progress_bar=True):
    iterable = list(zip(hosts, connect_options))
    async def run_start_wifi(args):
        host, co  = args
        conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
        try:
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

    res = await async_amap(run_start_wifi, iterable, num_workers=len(hosts), progress_bar=progress_bar, pbar_desc="Starting wifi connections")
    working_hosts = [host for host, co in res if host]
    working_connect_options = [co for host, co in res if host]
    return working_hosts, working_connect_options

async def check_connection(hosts, usernames, progress_bar=False):
    assert len(hosts) == len(usernames), "Hosts and usernames must be the same length"
    iterable = zip(hosts, usernames)
    async def run_connect(args):
        host, username = args
        try:
            await asyncio.wait_for(asyncssh.connect(host, username=username, password='rp145', known_hosts=None), timeout=10)
        except Exception as e:
            return False
        else:
            return True
    results = await async_amap(run_connect, iterable, num_workers=len(hosts), pbar_desc="Checking connections")
    working_hosts = [host for host, res in zip(hosts, results) if res]
    working_usernames = [username for username, res in zip(usernames, results) if res]
    return working_hosts, working_usernames

async def scan_for_pis(possible_usernames, ignore_hosts=[], progress_bar=False, password=None):
    if not password:
        password = os.environ['RASPI_PASSWORD']

    r = subprocess.run('nmap 10.157.115.0/24', shell=True, capture_output=True)

    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, 'nmap', r.stderr)

    stdout = r.stdout.decode()
    lines = stdout.split('\n')
    result_lines = lines[1:-2]
    i = 0
    all_reports = []
    report = []
    while i < len(result_lines):
        if result_lines[i] == '':
            all_reports.append(report)
            report = []
        else:
            report.append(result_lines[i])
        i += 1

    concurrent = True

    if concurrent:
        async def run_test_connect(report):
            ip = report[0].split(' ')[4]
            if ip in ignore_hosts:
                return None, None
            table_headers = [i for i, l in enumerate(report) if 'PORT' in l]
            if not table_headers:
                return None, None
            table_header_line = table_headers[0]
            open_ports = []
            for row_idx in range(table_header_line + 1, len(report)):
                if 'open' in report[row_idx]:
                    open_ports.append(report[row_idx].split(' ')[0])
                else:
                    break

            if '22/tcp' in open_ports:
                for username in possible_usernames:
                    try:
                        await asyncio.wait_for(asyncssh.connect(ip, username=username, password=password, known_hosts=None), timeout=10)
                    except Exception as e:
                        continue
                    else:
                        return ip, username
                else:
                    return None, None
            else:
                return None, None
            
        results = await async_amap(run_test_connect, all_reports, num_workers=len(all_reports), progress_bar=progress_bar, pbar_desc="Scanning for Pis")
        hosts, usernames = [host for host, username in results if host], [username for host, username in results if host]
        
    else:
        hosts = []
        usernames = []
        remaining_usernames = possible_usernames.copy()
        if progress_bar:
            all_reports = tqdm.tqdm(all_reports, desc="Scanning for Pis")
        for report in all_reports:
            ip = report[0].split(' ')[4]
            if ip in ignore_hosts:
                continue
            table_headers = [i for i, l in enumerate(report) if 'PORT' in l]
            if not table_headers:
                continue
            table_header_line = table_headers[0]
            open_ports = []
            for row_idx in range(table_header_line + 1, len(report)):
                if 'open' in report[row_idx]:
                    open_ports.append(report[row_idx].split(' ')[0])
                else:
                    break

            working_username = None
            if '22/tcp' in open_ports:
                for username in remaining_usernames:
                    try:
                        conn = await asyncio.wait_for(asyncssh.connect(ip, username=username, password=password, known_hosts=None), timeout=10)
                        working_username = username
                        break
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
    try:
        r = await conn.run('killall ~/ben/tiktok/venv/bin/python', check=True)
    except Exception as e:
        if hasattr(e, 'stderr') and 'no process found' in e.stderr:
            pass
        else:
            print(f'Failed to stop: {e}')

async def kill_workers(hosts, connect_options):
    async def run_killall(args):
        host, co = args
        conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
        await killall_python(conn)

    args = zip(hosts, connect_options)
    await async_amap(run_killall, args, num_workers=len(hosts))

async def stop_stale_workers(hosts, connect_options):
    async def run_killall(args):
        host, co = args
        try:
            conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
            await killall_python(conn)
        except:
            return None, None
        else:
            return host, co
    args = zip(hosts, connect_options)
    results = await async_amap(run_killall, args, num_workers=len(hosts))
    working_hosts = [host for host, co in results if host]
    working_connect_options = [co for host, co in results if host]
    return working_hosts, working_connect_options

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

async def run_on_pis(hosts, connect_options, func, **kwargs):
    async def run_func(args):
        host, connect_option, func, kwargs = args
        print(f"Connecting to {host}, {connect_option['username']}...")
        tries = 0
        max_tries = 3
        exceptions = []
        while tries < max_tries:
            tries += 1
            try:
                conn = await asyncio.wait_for(asyncssh.connect(host, **connect_option, known_hosts=None), timeout=10)
            except Exception as e:
                print(f'Failed to connect to {host}: {e}')
                continue
            else:
                async with conn:
                    try:
                        r = await func(conn, connect_option, **kwargs)
                    except asyncssh.process.ProcessError as e:
                        e.args = (e.args[0], e.stderr, traceback.format_exc())
                        exceptions.append(e)
                    except Exception as e:
                        exceptions.append(e)
                    else:
                        return r
        else:
            print(f"Failed to run on {host} (username: {connect_option['username']}) after {max_tries} tries: {exceptions}")
            return None

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

async def get_hosts_with_retries(usernames, max_tries=2, progress_bar=False):
    hosts = []
    found_usernames = []
    try:
        # load cached hosts and usernames
        with open('hosts.json', 'r') as file:
            user_hosts = json.load(file)
        hosts_users = [(host, un) for un, host in user_hosts.items()]
        file_hosts, file_usernames = zip(*hosts_users)
        file_hosts, file_usernames = list(file_hosts), list(file_usernames)
        print("Attempting to load cached connection data")
        num_tries = 0
        while num_tries < max_tries:
            num_tries += 1
            print(f"Attempt {num_tries} to find all hosts from file...")
            try_hosts, try_found_usernames = await check_connection(file_hosts, file_usernames, progress_bar=progress_bar)
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
        num_tries = 0
        non_found_usernames = list(set(usernames) - set(found_usernames))
        while num_tries < max_tries:
            num_tries += 1
            print(f"Attempt {num_tries} to find all hosts from network scan...")
            try_hosts, try_found_usernames = await scan_for_pis(non_found_usernames, ignore_hosts=hosts, progress_bar=progress_bar)
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
        'hoare', 'tarjan', 'miacli', 'fred',
        'geoffrey', 'rivest', 'edmund', 'ivan',
        'cook', 'barbara', 'goldwasser', 'milner',
        'hemming', 'frances', 'lee', 'turing',
        'floyd', 'juris', 'marvin',
        'conway', 'fernando', 'edward', 'edwin', 
        'satoshi', 'buterin', 'lovelace',
        'putnam', 'beauvoir',
        'arendt', 'mordvintsev', 'chan', 'sutskever', 'neumann',
        'edsger', 'herbert'
    ]
    hosts, usernames = await get_hosts_with_retries(potential_usernames, progress_bar=True)
    connect_options = [{'username': username, 'password': 'rp145'} for username in usernames]

    # TODO add more pis to network
    todo = 'get_os'

    if todo == 'setup':
        with open('worker_requirements.txt', 'r') as f:
            reqs = f.readlines()
        reqs = ' '.join([req.strip() for req in reqs])
        await run_on_pis(hosts, connect_options, setup_pi, reqs=reqs)
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