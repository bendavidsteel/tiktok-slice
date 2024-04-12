import asyncio
import os

from setup_pis import scan_for_pis

async def main():
    passwd = os.environ['PASSWORD']
    possible_usernames = [
        'hoare', 'tarjan', 'miacli', 'fred', 'geoffrey', 'rivest', 'edmund', 'pi',
        'frances', 'ivan', 'milner', 'cook', 'lee', 'barbara', 'goldwasser', 'gold', 'hemming'
    ]
    
    hosts, usernames = await scan_for_pis(possible_usernames)

    hosts_users = {host: user for host, user in zip(hosts, usernames)}
    print(hosts_users)

if __name__ == "__main__":
    asyncio.run(main())