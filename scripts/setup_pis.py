import asyncio
import sys

import asyncssh

async def main():
    hosts=[
        '10.157.115.214',
        '10.157.115.244',
        '10.157.115.234',
        '10.157.115.143',
        '10.157.115.198',
        '10.157.115.24',
        '10.157.115.213'
    ]
    connect_options=[
        { 'username': 'hoare', 'password': 'rp145' },
        { 'username': 'tarjan', 'password': 'rp145' },
        { 'username': 'miacli', 'password': 'rp145' },
        { 'username': 'fred', 'password': 'rp145' },
        { 'username': 'geoffrey', 'password': 'rp145' },
        { 'username': 'rivest', 'password': 'rp145' },
        { 'username': 'edmund', 'password': 'rp145' },
    ]
    with open('worker_requirements.txt', 'r') as f:
        reqs = f.readlines()
    reqs = ' '.join([req.strip() for req in reqs])

    for host, connect_option in zip(hosts, connect_options):
        print(f'Connecting to {host}...')
        async with asyncssh.connect(host, **connect_option, known_hosts=None) as conn:
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
            assert r.exit_status == 0
            r = await conn.run(f'~/ben/tiktok/venv/bin/pip install {reqs}', check=True)
            assert r.exit_status == 0

if __name__ == "__main__":
    asyncio.run(main())