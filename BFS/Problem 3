from queue import Queue
import numpy as np

myFile = open("input3.txt", "r")
nodes = int(myFile.readline())
totalconnection = int(myFile.readline())
graph = np.zeros((nodes, nodes), dtype="int")
for i in range(totalconnection):
    line = myFile.readline().split(' ')
    x = int(line[0])
    y = int(line[1])
    graph[y][x] = 1

visited = np.empty(nodes, dtype="int")
parent = np.empty(nodes, dtype="object")
distance = np.zeros(nodes, dtype="int")

lina_in_position = int(myFile.readline())
number_of_participants = int(myFile.readline())
position_of_first_participant = int(myFile.readline())
position_of_second_participant = int(myFile.readline())
position_of_third_participant = int(myFile.readline())
position_of_fourth_participant = int(myFile.readline())
position_of_fifth_participant = int(myFile.readline())

qu = Queue()


def bfs(pos):
    visited[:] = 0
    parent[:]
    distance[:] = 10000
    visited[pos] = 1
    distance[pos] = 0
    qu.put(pos)
    while not qu.empty():
        u = qu.get()
        for v in range(0, nodes):
            if graph[u][v] == 1:
                if visited[v] == 0:
                    visited[v] = 1
                    distance[v] = distance[u]+1
                    parent[v] = u
                    qu.put(v)
        visited[u] = 2
    return


bfs(lina_in_position)
dis1 = distance[position_of_first_participant]
dis2 = distance[position_of_second_participant]
dis3 = distance[position_of_third_participant]
dis4 = distance[position_of_fourth_participant]
dis5 = distance[position_of_fifth_participant]
print(min(dis1,dis2,dis3,dis4,dis5))
