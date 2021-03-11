import sys
import argparse
import pickle
import pygame
from pygame.locals import *
import math
from scipy.spatial import distance
import copy

'''Sample Run Code
python NetworkVisualization.py --o ../Outputs/Final --e dietadj
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--o', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--e', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--nodep', dest='node_power', type=int, default=3)
    parser.add_argument('--edgep', dest='edge_power', type=int, default=3)
    parser.add_argument('--nodes', dest='node_maxsize', type=int, default=50) #radius
    parser.add_argument('--edges', dest='edge_maxsize', type=int, default=30)
    parser.add_argument('--labelshow', dest='show_labels', type=int, default=0)
    parser.add_argument('--labels', dest='label_maxsize', type=int, default=50)
    parser.add_argument('--fromsave', dest='from_save', type=int, default=1)
    options = parser.parse_args(argv[1:])

    output_path = options.output_path
    experiment_name = options.experiment_name
    experiment_path = output_path + '/' + experiment_name
    edge_power = options.edge_power
    node_power = options.node_power
    edge_maxsize = options.edge_maxsize
    node_maxsize = options.node_maxsize
    show_labels = options.show_labels == 1
    label_maxsize = options.label_maxsize
    from_save = options.from_save == 1

    # Extract Network Data
    save_opened = False
    if from_save:
        try:
            file = open(experiment_path + '/Composite/rulepop/savedpygame', 'rb')
            info = pickle.load(file)
            file.close()
            save_opened = True
        except:
            file = open(experiment_path + '/Composite/rulepop/networkpickle', 'rb')
            info = pickle.load(file)
            file.close()
    else:
        file = open(experiment_path + '/Composite/rulepop/networkpickle', 'rb')
        info = pickle.load(file)
        file.close()

    acc_spec_dict = info[0]
    edge_list = info[1]
    weight_list = info[2]
    pos = info[3]

    # Pygame
    pygame.init()
    main_clock = pygame.time.Clock()
    window_width = 1000
    window_height = 1000
    window_surface = pygame.display.set_mode((window_width, window_height))
    BLACK = (0, 0, 0)
    WHITE = (255,255,255)
    NODECOLOR = (255,51,119)
    EDGECOLOR = (224,184,255)

    originalasd = copy.deepcopy(acc_spec_dict)
    originalel = copy.deepcopy(edge_list)
    originalwl = copy.deepcopy(weight_list)

    #Normalize Sizes
    max_node_value = max(acc_spec_dict.values())
    for i in acc_spec_dict:
        acc_spec_dict[i] = math.pow(acc_spec_dict[i] / max_node_value, node_power) * node_maxsize  # Cubic Node Size Function

    max_weight_value = max(weight_list)
    for i in range(len(weight_list)):
        weight_list[i] = math.pow(weight_list[i] / max_weight_value, edge_power) * edge_maxsize  # Cubic Weight Function

    # Set up nodes' initial positions:
    nodes = {}
    for nodename in pos:
        x = int(int(pos[nodename][0] * window_width * 0.4) + window_width / 2)
        y = int(int(pos[nodename][1] * window_height * 0.4) * -1 + window_height / 2)
        nodes[nodename] = [acc_spec_dict[nodename],x,y]

    # Set up initial labels
    if not save_opened:
        label_display = {}
        for nodename in acc_spec_dict:
            font = pygame.font.SysFont("monospace", max(20,int(acc_spec_dict[nodename]/node_maxsize*label_maxsize)))
            label_display[nodename] = [show_labels,font]
    else:
        label_display = {}
        for nodename in acc_spec_dict:
            font = pygame.font.SysFont("monospace", max(20,int(acc_spec_dict[nodename] / node_maxsize * label_maxsize))) #font minsize is 10
            label_display[nodename] = [info[4][nodename], font]

    dragged_nodes = []

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                newpos = {}
                labels = {}
                for nodename in nodes:
                    tx = (nodes[nodename][1]-window_width/2)/(window_width*0.4)
                    ty = (nodes[nodename][2]-window_height/2)/(window_height*0.4)*-1
                    newpos[nodename] = [tx,ty]
                    labels[nodename] = label_display[nodename][0]
                to_save = [originalasd,originalel,originalwl,newpos,labels]
                outfile = open(experiment_path + '/Composite/rulepop/savedpygame', 'wb')
                pickle.dump(to_save, outfile)
                outfile.close()

                pygame.image.save(window_surface,experiment_path + '/Composite/rulepop/saved.png')

                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == ord('l'):
                    mx,my = pygame.mouse.get_pos()
                    for nodename in nodes:
                        dist = distance.euclidean((nodes[nodename][1],nodes[nodename][2]),(mx,my))
                        if dist < nodes[nodename][0]:
                            label_display[nodename][0] = not label_display[nodename][0]

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for nodename in nodes:
                    dist = distance.euclidean((nodes[nodename][1], nodes[nodename][2]), (mx, my))
                    if dist < nodes[nodename][0]:
                        if not nodename in dragged_nodes:
                            dragged_nodes.append(nodename)

            if event.type == pygame.MOUSEBUTTONUP:
                dragged_nodes = []

        window_surface.fill(WHITE)

        mx, my = pygame.mouse.get_pos()
        for nodename in dragged_nodes:
            nodes[nodename][1] = mx
            nodes[nodename][2] = my

        edgecounter = 0
        for n1,n2 in edge_list:
            weight = weight_list[edgecounter]
            pygame.draw.line(window_surface,EDGECOLOR,(nodes[n1][1],nodes[n1][2]),(nodes[n2][1],nodes[n2][2]),int(weight))
            edgecounter += 1

        for nodename in nodes:
            x = nodes[nodename][1]
            y = nodes[nodename][2]
            pygame.draw.circle(window_surface,NODECOLOR,(x,y),int(nodes[nodename][0]))
            if label_display[nodename][0]:
                label = label_display[nodename][1].render(nodename,1,BLACK)
                size = label_display[nodename][1].size(nodename)
                window_surface.blit(label,(x-size[0]/2,y-size[1]/2))

        pygame.display.update()
        main_clock.tick()

if __name__ == '__main__':
    sys.exit(main(sys.argv))