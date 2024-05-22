import asyncio

import dotenv

from setup_pis import get_hosts_with_retries

async def main():
    dotenv.load_dotenv()

    possible_usernames = [
        # most reliable batch
        'hoare', 'tarjan', 'miacli', 'fred',
        'geoffrey', 'rivest', 'edmund', 'ivan',
        'cook', 'barbara', 'goldwasser', 'milner',
        'hemming', 'frances', 'lee', 'turing',
        'marvin', 'juris', 'floyd', 'edsger',
        'neumann', 'beauvoir', 'satoshi', 'putnam', 
        'fernando', 'edwin'
        # next most reliable
        # 'conway', 'edward',
        # 'buterin', 'lovelace',
        # 'arendt', 'chan', 'sutskever',
        # 'herbert',
        # 'mordvintsev'
        # not setup yet
        "shannon", "chowning", "tegmark", "hanson"
        "chomsky", "keynes"
    ]
    
    hosts, usernames = await get_hosts_with_retries(possible_usernames, 2, progress_bar=True)

    hosts_users = {host: user for host, user in zip(hosts, usernames)}
    print(hosts_users)

    print(f"Couldn't find: {set(possible_usernames) - set(usernames)}")

if __name__ == "__main__":
    asyncio.run(main())