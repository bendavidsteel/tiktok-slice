import asyncio
import os

from setup_pis import scan_for_pis

async def main():
    hello = 1
    f"{hello} {{hi}}"
    possible_usernames = [
        'hoare', 'tarjan', 'miacli', 'fred', 'geoffrey', 'rivest', 'edmund', 'juris',
        'frances', 'ivan', 'milner', 'cook', 'lee', 'barbara', 'goldwasser', 'hemming',
        'floyd', 'turing'
    ]
    
    hosts, usernames = await scan_for_pis(possible_usernames)

    hosts_users = {host: user for host, user in zip(hosts, usernames)}
    print(hosts_users)

    print(f"Couldn't find: {set(possible_usernames) - set(usernames)}")

if __name__ == "__main__":
    asyncio.run(main())