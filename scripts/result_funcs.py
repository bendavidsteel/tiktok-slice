import os

def get_result_paths(result_dir_path, result_filename='results.parquet.gzip'):

    for dir_path, dir_names, filenames in os.walk(result_dir_path):
        for filename in filenames:
            if filename == result_filename:
                result_path = os.path.join(dir_path, filename)
                yield result_path

async def get_remote_result_paths(conn, result_filename='results.parquet.gzip'):
    r = await conn.run('ls -R ~/repos/what-for-where/data/results/')
    output = r.stdout
    lines = output.split('\n')
    dir_files = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.endswith(':'):
            dir_path = line[:-1]
            dir_files[dir_path] = []
            i += 1
            while i < len(lines) and lines[i] and not lines[i].endswith(':'):
                dir_files[dir_path].append(lines[i])
                i += 1
        i += 1
    result_paths = []
    for dir_path, files in dir_files.items():
        for file in files:
            if file == result_filename:
                result_path = os.path.join(dir_path, file)
                result_paths.append(result_path)

    return result_paths