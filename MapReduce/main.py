import collections
import os
import re
import string

import numpy as np
from mpi4py import MPI

input_path = "input"
map_path = "map"
reduce_path = "reduce"
output_path = "output"


def mapper(filename):
    words = []
    with open(input_path + '/' + filename, 'r', errors="ignore") as f:
        for line in f:
            for word in line.split():
                word = re.sub(r'[^A-Za-z]', '', word).lower()
                if word == '':
                    continue
                words.append([word, filename[:-4]])
    f.close()
    return words


def reducer(letters_list):
    pairs = []
    for filename in os.listdir(map_path):
        with open(map_path + '/' + filename, 'r') as f:
            for line in f:
                words = line.split()
                if words[0][0] in letters_list:
                    temp = (words[0], int(words[1]))
                    pairs.append(temp)
        f.close()
    ordered_dict = collections.OrderedDict(sorted(collections.Counter(pairs).items()))
    data = {}
    for key in ordered_dict:
        if key[0] not in data.keys():
            data[key[0]] = {key[1]: ordered_dict[key]}
        else:
            data[key[0]][key[1]] = ordered_dict[key]
    return data


def divide_to_processes(no_processes, data):
    return [list(x) for x in np.array_split(data, no_processes)]


if __name__ == '__main__':
    letters = list(string.ascii_lowercase)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    no_mappers = (size - 1) // 2 + (size - 1) % 2
    no_reducers = (size - 1) // 2
    if rank == 0:
        map_data = divide_to_processes(no_mappers, os.listdir(input_path))
        reduce_data = divide_to_processes(no_reducers, letters)
        for i in range(len(map_data)):
            comm.send(map_data[i], dest=i + 1, tag=99)
        for i in range(len(reduce_data)):
            comm.send(reduce_data[i], dest=i + no_mappers + 1, tag=99)
        comm.barrier()
        for p in range(no_mappers + 1, size):
            comm.recv(source=p, tag=90)
        text = ''
        for filename in os.listdir(reduce_path):
            with open(reduce_path + '/' + filename, 'r') as f:
                text += f.read()
        with open(f'{output_path}/output.txt', 'w') as f:
            f.write(text)
    elif 1 <= rank <= no_mappers:
        content_list = []
        map_data = comm.recv(source=0, tag=99)
        for filename in map_data:
            words = mapper(filename)
            content_list += words
        with open(f'{map_path}/mapper{rank}.txt', 'w') as f:
            for pair in content_list:
                f.write(f'{pair[0]} {pair[1]}\n')
        comm.barrier()
    else:
        comm.barrier()
        reduce_data = comm.recv(source=0, tag=99)
        content = reducer(reduce_data)
        with open(f'{reduce_path}/reducer{rank}.txt', 'w') as f:
            for key in content.keys():
                f.write(f'{key} -> {content[key]}\n')
        comm.send("Ok", dest=0, tag=90)
