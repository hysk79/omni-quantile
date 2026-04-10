import matplotlib.pyplot as plt

def color_func(total_n, idx):
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / (1.25*(total_n-1))) for i in range(2*(total_n-1))]
    return colors[idx]