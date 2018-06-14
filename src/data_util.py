

def write_fen(fen_string, fname):
    fname = fname[:-4]
    print("Writing {}".format(fname + "_fen.txt"))
    with open(fname + "_fen.txt", "w") as f:
        f.write(fen_string)
        


def extract_squares(board):
    ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    files = ["1", "2", "3", "4", "5", "6", "7", "8"]
    squares = []
    names = []
    ww, hh = board.shape
    w = int(ww / 8)
    h = int(hh / 8)

    for i in range(8):
        for j in range(8):
            squares.append(board[i*w:(i+1)*w, j*h:(j+1)*h])
            names.append(ranks[j]+files[7-i]) 
    return squares, names