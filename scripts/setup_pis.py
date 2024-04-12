import asyncio
import datetime
import json
import os
import random
import subprocess

import asyncssh

async def setup_pi(conn, reqs=''):
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
    r = await conn.run('~/ben/tiktok/venv/bin/python --version', check=True)
    if r.stdout.strip() != 'Python 3.10.12':
        r = await conn.run('rm -rf ~/ben/tiktok/venv', check=True)
        r = await conn.run('python3 -m venv ~/ben/tiktok/venv', check=True)
    r = await conn.run(f'~/ben/tiktok/venv/bin/pip install {reqs}', check=True)

async def check_connection(hosts, usernames):
    for host, username in zip(hosts, usernames):
        await asyncio.wait_for(asyncssh.connect(host, username=username, password='rp145', known_hosts=None), timeout=10)

async def scan_for_pis(possible_usernames):
    pi_password = 'rp145'

    r = subprocess.run('nmap 10.157.115.0/24', shell=True, capture_output=True)
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

    hosts = []
    usernames = []
    for report in all_reports:
        ip = report[0].split(' ')[4]
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
            for username in possible_usernames:
                try:
                    conn = await asyncio.wait_for(asyncssh.connect(ip, username=username, password=pi_password, known_hosts=None), timeout=10)
                    working_username = username
                    break
                except Exception as e:
                    continue

        if working_username:
            hosts.append(ip)
            usernames.append(working_username)

    return hosts, usernames

def generate_random_mac():
    # Create a list of 6 hex values, each 00 to FF
    mac = [random.randint(0, 255) for _ in range(6)]
    # Format the MAC address in the standard format with colon separation
    mac_address = ':'.join(f'{value:02X}' for value in mac)
    return mac_address

async def change_mac_address(conn):
    # write bash script to change mac address on eth0 network interface
    random_mac = generate_random_mac()
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

async def killall_python(conn):
    try:
        r = await conn.run('killall ~/ben/tiktok/venv/bin/python', check=True)
    except Exception as e:
        if 'no process found' in e.stderr:
            pass
        else:
            print(f'Failed to stop: {e}')

async def kill_workers(hosts, connect_options):
    for host, co in zip(hosts, connect_options):
        conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
        await killall_python(conn)

async def change_mac_addresses(hosts, connect_options):
    for host, co in zip(hosts, connect_options):
        conn = await asyncio.wait_for(asyncssh.connect(host, **co), timeout=10)
        await change_mac_address(conn)

async def run_on_pis(hosts, connect_options, func, **kwargs):
    results = []
    for host, connect_option in zip(hosts, connect_options):
        print(f'Connecting to {host}...')
        try:
            conn = await asyncio.wait_for(asyncssh.connect(host, **connect_option, known_hosts=None), timeout=10)
        except Exception as e:
            print(f'Failed to connect to {host}: {e}')
            continue
        else:
            async with conn:
                r = await func(conn, **kwargs)
                results.append(r)
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

async def get_hosts_with_retries(usernames, max_tries=10):
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
        num_tries = 0
        while num_tries < max_tries:
            hosts, found_usernames = await scan_for_pis(usernames)
            if len(found_usernames) == len(usernames):
                break
        else:
            raise Exception(f"Failed to find all hosts after {max_tries} tries")
        user_hosts = {un: host for un, host in zip(found_usernames, hosts)}
        with open('hosts.json', 'w') as file:
            json.dump(user_hosts, file)

    return hosts, found_usernames

async def main():
    potential_usernames = [
        'hoare',
        'tarjan',
        'miacli',
        'fred',
        'geoffrey',
        'rivest',
        'edmund',
        # 'ivan',
        # 'cook',
        # 'barbara',
        # 'goldwasser',
        # 'milner',
        # 'hemming',
        # 'frances',
        # 'lee'
    ]
    hosts, usernames = await get_hosts(potential_usernames)
    connect_options = [{'username': username, 'password': 'rp145'} for username in usernames]

    # TODO look into connecting pis to tum vpn for larger network range
    # TODO add more pis to network
    todo = 'change_ip'

    if todo == 'setup':
        with open('worker_requirements.txt', 'r') as f:
            reqs = f.readlines()
        reqs = ' '.join([req.strip() for req in reqs])
        run_on_pis(hosts, connect_options, setup_pi, reqs=reqs)
    elif todo == 'ping':
        async def ping(conn):
            r = await conn.run('ls', check=True)
        run_on_pis(hosts, connect_options, ping)
    elif todo == 'get_ip':
        async def print_ip(conn):
            r = await conn.run('curl ifconfig.me', check=True)
            print(f'Public IP: {r.stdout.strip()}')
        run_on_pis(hosts, connect_options, print_ip)
    elif todo == 'change_ip':
        async def get_ip(conn):
            r = await conn.run('curl ifconfig.me', check=True)
            return r.stdout.strip()
        initial_ips = await run_on_pis(hosts, connect_options, get_ip)
        print(f"Initial IPs: {initial_ips}")
        start_time = datetime.datetime.now()
        print("Changing MAC addresses...")
        await run_on_pis(hosts, connect_options, change_mac_address)
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

        print("Getting new IPs...")
        new_ips = await run_on_pis(found_hosts, found_connect_options, get_ip)
        print(f"New IPs: {new_ips}")
    elif todo == 'stop':
        run_on_pis(hosts, connect_options, killall_python)

        
            
if __name__ == "__main__":
    asyncio.run(main())