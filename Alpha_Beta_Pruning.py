import re

def alpha_beta(pos, depth, alpha, beta, m_player, n, comp):
    if depth == 0:
        return n[pos], comp

    result = 0
    if m_player:
        maxv = alpha
        for i in range(0, branches):
            res, comp = alpha_beta(pos * branches + i, depth - 1, alpha, beta, False, n, comp)
            maxv = max(maxv, res)
            alpha = max(alpha, maxv)
            if beta <= alpha:
                comp += 1
                break

        return maxv, comp
    else:
        minv = beta
        for i in range(0, branches):
            res, comp = alpha_beta(pos * branches + i, depth - 1, alpha, beta, True, n, comp)

            minv = min(minv, res)
            beta = min(beta, minv)
            if beta <= alpha:
                comp += 1
                break

        return minv, comp


def minimax(pos, depth, alpha, beta, m_player, n, comp):
    if depth == 0:
        comp += 1
        return n[pos], comp

    result = 0
    if m_player:
        maxv = alpha
        for i in range(0, branches):
            res, comp = minimax(pos * branches + i, depth - 1, alpha, beta, False, n, comp)
            maxv = max(maxv, res)

        return maxv, comp
    else:
        minv = beta
        for i in range(0, branches):
            res, comp = minimax(pos * branches + i, depth - 1, alpha, beta, True, n, comp)

            minv = min(minv, res)

        return minv, comp



myFile = open("input.txt", "r")
file = myFile.read()
list1 = re.split("\s", file)
leaf_nodes = []
for x in range(len(list1)):
    y = int(list1[x])
    leaf_nodes.append(y)

turn = int(input("Number of Turns:"))
branches = int(input("Branches per node:"))
minvalue=input("minimum value for leaf nodes: ")
maxvalue=input("miximum value for leaf nodes: ")
minimum = int(minvalue)
maximum = int(maxvalue)
depth = 2 * turn
total_leaf_nodes = branches ** depth
print("Depth:", depth)
print("Branch:", branches)
print("Terminal States (Leaf Nodes):", total_leaf_nodes)

comp = 0
mini_witout, count = minimax(0, depth, -1000, 1000, True, leaf_nodes, comp)
mxamnt, cmp = alpha_beta(0, depth, -1000, 1000, True, leaf_nodes, comp)
print("Maximum ammount :", mxamnt)
print("Comparisons 1:", count)

print("Comparisons 2: ", count - cmp)
