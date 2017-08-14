import math


def get_degree(x1, y1, x2, y2):
    radians = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(radians)


def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_position(degree, distance, x1, y1):
    theta = math.pi / 2 - math.radians(degree)
    return x1 + distance * math.sin(theta), y1 + distance * math.cos(theta)

# I can't understand the negtive operation.
def get_position2(degree, distance, x1, y1):
    theta = math.radians(degree) # 0-360 not -1->1
    return x1 + distance * math.cos(theta), y1 + distance * math.sin(theta)



def print_progress(episodes, wins):
    print "Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
        episodes, wins, wins / (episodes + 1E-6))
