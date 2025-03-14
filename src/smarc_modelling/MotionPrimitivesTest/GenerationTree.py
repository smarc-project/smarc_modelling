import heapq
import math
from smarc_modelling.MotionPrimitivesTest.MapGeneration import *
from smarc_modelling.MotionPrimitivesTest.MotionPrimitives import SAM_PRIMITIVES

class Node:
    def __init__(self, state, cost=0):
        self.state = state
        self.cost = cost  # g(n) + h(n)

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.state[0] == other.state[0] and self.state[1] == other.state[1] and self.state[2] == other.state[2]

    def __hash__(self):
        return hash((self.state[0], self.state[1], self.state[2]))  # Make Node hashable by its state
    
def arrived(current):
    x = current.state[0]
    y = current.state[1]
    z = current.state[2]

    arrivalx_min = TILESIZE * goalCellx
    arrivalx_max = arrivalx_min + TILESIZE
    arrivaly_min = randomPositionGoal * TILESIZE
    arrivaly_max = arrivaly_min + TILESIZE
    #Add also for z

    if (x<arrivalx_min or x>arrivalx_max) or (y<arrivaly_min or y>arrivaly_max) :
        return False
    return True

def reconstruct_path(current, parents_dict):
    final_path = []
    while current is not None:
        final_path.append(current.state)
        #current = parents_dict.get(current, (None, None))[0]  # Get parent node
        current = parents_dict[current]
    return final_path[::-1]  # Return reversed path

def get_neighbors(current, sim):
    max_input = 7
    step_input = 0.1
    reached_states = []
    last_states = []
    for input_u in np.arange(-max_input, max_input + step_input, step_input):
        data, cost = sim.curvePrimitives(current.state, input_u)
        if len(data) == 0:
            continue
        reached_states.append(data) #The complete sequence of states
        #find cost of this path
        last_states.append((data[:,-1],cost))  #Only the last state for each input
    return reached_states, last_states
    


def heuristic(state, goal_p):
    return math.sqrt((state[0] - goal_p[0]) ** 2 + (state[1] - goal_p[1]) ** 2)

def a_star_search(x0, DISPLAY):
    sim = SAM_PRIMITIVES()
    start = Node(x0)
    open_set = []   #(cost_node, Node)
    parents_dictionary = {} #(Node_current: Node_parent)
    g_cost = {start: 0} #keeps track of costs of nodes, useful to understand more convenient paths
    heapq.heappush(open_set, (0,start))
    parents_dictionary[start] = None

    flag = 0

    while (open_set):
        flag = flag + 1
        print("Iteration number:")
        print(flag)
        _, current_node = heapq.heappop(open_set)   #removes and returns the node with lowest f value
        current_g = g_cost[current_node]

        # Check if we reached the goal
        if arrived(current_node) or flag>100000:
            print("--> we try to get the final path!")
            return reconstruct_path(current_node, parents_dictionary)
        
        #Move current node from open to closed list
        # Already done inherently by using heapq
        
        # Check new neighbors using primitives
        reached_states, last_states = get_neighbors(current_node, sim)

                # Draw the lines for the primitives in Pygame
        for sequence_states in reached_states:
            for i in range(1, sequence_states.shape[1]):
                # Draw from state i-1 to state i
                pygame.draw.line(
                    DISPLAY, (0, 0, 255),
                    (sequence_states[0, i - 1], sequence_states[1, i - 1]),
                    (sequence_states[0, i], sequence_states[1, i]),
                    2
                )
        pygame.display.update()  # Update display for each batch of primitives

        for neighbor, cost_path in last_states:
            print("-> Evaluating neighbor")
            # Skip neighbor if already evaluated
            #not sure if I want to update the costs??

            # Calculate tentative g score
            # Find the corresponding sequence of states for this neighbor
            tentative_g_cost = current_g + cost_path   #current + distance between current and neighbor

            if Node(neighbor) not in g_cost or tentative_g_cost < g_cost[neighbor]:
                g_cost[Node(neighbor)] = tentative_g_cost     #add BACK cost to g_cost
                f_cost = tentative_g_cost + heuristic(neighbor, (map["goal_pixel"][0], map["goal_pixel"][1]))
                #I should add this node to the grid
                parents_dictionary[Node(neighbor)] = (current_node)     #add node dependency on current node
                heapq.heappush(open_set, (f_cost, Node(neighbor)))    #nodes are evaluated based on f


    return [] # No path found 



    
