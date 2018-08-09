#info depth 10 seldepth 21 multipv 1 score cp 1176 nodes 27704 nps 1319238 tbhits 0 time 21 pv e5c4 d6c6 c4a5 c6d6 h5e2 b7b6 d2d4 b6a5 d4c5 d6c6 d5c3 d8e7 f7d5 c6c5 c1e3 e7e3 e2e3 c5b4 d5a8

info = ["info", "depth", "10", "score", "cp", "1176"]

score = float(info[info.index("cp") + 1]) / 100.

print(score)